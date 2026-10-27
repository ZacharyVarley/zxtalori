// Stream only the symmetry images in an angular shell around the closed FZ.
struct ShellParams{n:u32,capacity:u32,order:u32,planes:u32,fence:u32,zero:u32,pad0:u32,pad1:u32,config:vec4<f32>,padding:vec4<f32>}
@group(0) @binding(0) var<uniform> u:ShellParams;
@group(0) @binding(1) var<storage,read_write> sites:array<vec4<f32>>;
@group(0) @binding(11) var<storage,read> halfspaces:array<vec4<f32>>;
@group(0) @binding(12) var<storage,read> ops:array<vec4<f32>>;
@group(0) @binding(14) var<storage,read_write> info:array<atomic<u32>>;
@group(0) @binding(15) var<storage,read> core:array<vec4<f32>>;
@group(0) @binding(16) var<storage,read_write> origin:array<u32>; // source + n*(2*op+negative)
@group(0) @binding(17) var<storage,read_write> blocks:array<u32>; // per workgroup: shell, fence
fn mul(a:vec4<f32>,b:vec4<f32>)->vec4<f32>{return vec4<f32>(a.x*b.x-dot(a.yzw,b.yzw),a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));}
fn ordered(x:f32)->u32{let bits=bitcast<u32>(x);return select(bits^0x80000000u,~bits,(bits>>31u)!=0u);}
// Images within the halfspace margin s of the FZ, of sources near its boundary.
fn near_zone(original:vec4<f32>,q:vec4<f32>,s:f32)->bool{
    var rim=false;for(var k=0u;k<u.planes;k++){if(dot(original,halfspaces[k])<=s+2e-6){rim=true;break;}}if(!rim){return false;}
    for(var k=0u;k<u.planes;k++){if(dot(q,halfspaces[k])< -s-2e-6){return false;}}return true;
}
// Image `index` = source + n*(2*op+negative): kind 0 dropped, 1 core, 2 shell (margin
// config.x), 3 fence (beyond it, within config.y, when u.fence is set). Fence sites
// are real sites that are never seeds of the DT. They only bound the empty spheres
// of outer shell sites; every cap through one lies farther than the margin from the
// FZ, so no counted cap changes.
struct ShellImage{kind:u32,q:vec4<f32>}
fn classify(index:u32)->ShellImage{
    if(index>=u.n*u.order*2u){return ShellImage(0u,vec4<f32>(0.0));}
    let source=index%u.n;let image=index/u.n;let op=image/2u;let negative=(image&1u)!=0u;
    let original=normalize(core[source]);var q=normalize(mul(original,normalize(ops[op])));if(negative){q=-q;}
    var kind=0u;
    if(op==0u&&!negative){kind=1u;}else if(near_zone(original,q,u.config.x)){kind=2u;}else if(u.fence!=0u&&near_zone(original,q,u.config.y)){kind=3u;}
    // A south-pole point cannot define a bounded cap centred in the FZ with
    // radius < pi/2. Near-pole samples otherwise retain their original bits.
    if(q.x<=-0.9999999){kind=0u;}
    return ShellImage(kind,q);
}
// Stable compaction in index order, so site ids (and the DT's symbolic tie-breaking)
// do not depend on scheduling: equal ids across passes, and on lattices far fewer
// exact predicates than scheduling order gives (T h=8: 1.3k against about 6.7k).
var<workgroup> scan:array<u32,128>;
var<workgroup> scan_fence:array<u32,128>;
// Exclusive workgroup prefix of `value` (shell count | fence count << 16); scan[127] holds the total.
fn block_prefix(lane:u32,value:u32)->u32{
    scan[lane]=value;workgroupBarrier();
    for(var stride=1u;stride<128u;stride*=2u){var add=0u;if(lane>=stride){add=scan[lane-stride];}workgroupBarrier();scan[lane]+=add;workgroupBarrier();}
    return scan[lane]-value;
}
fn packed(kind:u32)->u32{return select(0u,1u,kind==2u)|select(0u,1u<<16u,kind==3u);}
@compute @workgroup_size(128)
fn shell_count(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>){
    let index=id.x+id.y*groups.x*128u;let block=group.x+group.y*groups.x;
    let kind=classify(index).kind;let before=block_prefix(lane,packed(kind));
    if(lane==127u){let total=before+packed(kind);blocks[2u*block]=total&0xffffu;blocks[2u*block+1u]=total>>16u;}
}
// One workgroup: block counts become exclusive offsets; info[0] = n + shell, info[13] = fence.
@compute @workgroup_size(128)
fn shell_scan(@builtin(local_invocation_index) lane:u32){
    let count=(u.n*u.order*2u+127u)/128u;let width=(count+127u)/128u;
    let begin=min(lane*width,count);let end=min(begin+width,count);
    var shell=0u;var fence=0u;for(var b=begin;b<end;b++){shell+=blocks[2u*b];fence+=blocks[2u*b+1u];}
    scan[lane]=shell;scan_fence[lane]=fence;workgroupBarrier();
    for(var stride=1u;stride<128u;stride*=2u){
        var add=0u;var add_fence=0u;if(lane>=stride){add=scan[lane-stride];add_fence=scan_fence[lane-stride];}workgroupBarrier();
        scan[lane]+=add;scan_fence[lane]+=add_fence;workgroupBarrier();
    }
    var at=scan[lane]-shell;var at_fence=scan_fence[lane]-fence;
    for(var b=begin;b<end;b++){let s=blocks[2u*b];let f=blocks[2u*b+1u];blocks[2u*b]=at;blocks[2u*b+1u]=at_fence;at+=s;at_fence+=f;}
    if(lane==127u){atomicStore(&info[0],u.n+scan[127]);atomicStore(&info[13],scan_fence[127]);}
}
// Core sites keep their source index; shell sites follow from n, then the fence.
@compute @workgroup_size(128)
fn shell_build(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>){
    let index=id.x+id.y*groups.x*128u;let block=group.x+group.y*groups.x;
    let image=classify(index);let before=block_prefix(lane,packed(image.kind));
    if(image.kind==0u){return;}
    var at=index;
    if(image.kind==2u){at=u.n+blocks[2u*block]+(before&0xffffu);}
    else if(image.kind==3u){at=atomicLoad(&info[0])+blocks[2u*block+1u]+(before>>16u);}
    if(at>=u.capacity){atomicStore(&info[1],1u);return;}
    let q=image.q;let position=q.yzw/(1.0+q.x);sites[at]=vec4<f32>(position,0.0);origin[at]=index;
    for(var k=0u;k<3u;k++){atomicMin(&info[2u+k],ordered(position[k]));atomicMax(&info[5u+k],ordered(position[k]));}
}
// Shell images (no fence) of every source, or with pad0 set only of moved sources
// (moved[source] != 0), appended as (index, stereographic coordinate bits) for the CPU
// triangulation (live_dt.zig). Appended past pad1 entries only counts them.
@group(0) @binding(18) var<storage,read> moved:array<u32>;
@group(0) @binding(19) var<storage,read_write> images:array<vec4<u32>>;
@group(0) @binding(20) var<storage,read_write> image_count:array<atomic<u32>>;
@compute @workgroup_size(128)
fn shell_images(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>){
    let index=id.x+id.y*groups.x*128u;if(index>=u.n*u.order*2u){return;}
    if(u.pad0!=0u&&moved[index%u.n]==0u){return;}
    let image=classify(index);if(image.kind==0u){return;}
    let at=atomicAdd(&image_count[0],1u);if(at>=u.pad1){return;}
    let q=image.q;images[at]=vec4<u32>(index,bitcast<vec3<u32>>(q.yzw/(1.0+q.x)));
}
// Live covering moves (covering.Live.beginMove; layout in live.wgsl): one thread per
// moved source rewrites the slots of its images that are still shell or fence images,
// takes free slots for images that entered, frees the slots of images that left, and
// records each change. u.capacity is the slot capacity; the box the positions must stay
// in is config.zw, padding.x (lower) and padding.yzw (upper).
@group(0) @binding(21) var<storage,read_write> live_coordinates:array<vec4<u32>>;
@group(0) @binding(22) var<storage,read_write> live_origin:array<u32>;
@group(0) @binding(23) var<storage,read_write> live_links:array<atomic<u32>>;
@group(0) @binding(24) var<storage,read_write> live_scratch:array<atomic<u32>>;
const LIVE_NONE=0xffffffffu;
const LIVE_HIGH=0x80000000u;
const LIVE_IMAGES=120u; // 2 |K| for the icosahedral group
// flags: 1 old position, 2 new position, 4 a seed now, 8 the position changed.
fn live_change(slot:u32,flags:u32,before:vec3<u32>,after:vec3<u32>){
    let at=atomicAdd(&live_scratch[2],1u);if(at>=u.capacity){atomicOr(&live_scratch[3],4u);return;}
    let b=4u+2u*u.capacity+8u*at;
    atomicStore(&live_scratch[b],slot);atomicStore(&live_scratch[b+1u],flags);
    for(var k=0u;k<3u;k++){atomicStore(&live_scratch[b+2u+k],before[k]);atomicStore(&live_scratch[b+5u+k],after[k]);}
}
var<workgroup> live_kind:array<u32,LIVE_IMAGES>;
var<workgroup> live_q:array<vec4<f32>,LIVE_IMAGES>;
// One workgroup per moved source (`moved` here is the list live_moved makes: [0] count,
// then sources): the lanes classify its images, lane 0 keeps its slots.
@compute @workgroup_size(64)
fn shell_live_images(@builtin(workgroup_id) wid:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let item=wid.x+wid.y*groups.x;if(item>=moved[0]){return;}
    let s=moved[1u+item];
    for(var image=lane;image<u.order*2u;image+=64u){let im=classify(s+u.n*image);live_kind[image]=im.kind;live_q[image]=im.q;}
    workgroupBarrier();
    if(lane!=0u){return;}
    var kept:array<u32,LIVE_IMAGES>;var count=0u;
    var link=atomicLoad(&live_links[s]);
    while(link!=0u&&count<LIVE_IMAGES){kept[count]=link-1u;count++;link=atomicLoad(&live_links[u.n+link-1u]);}
    var retained:array<bool,LIVE_IMAGES>;
    let lower=vec3<f32>(u.config.z,u.config.w,u.padding.x);let upper=u.padding.yzw;
    var chain=0u;
    for(var image=0u;image<u.order*2u;image++){
        let index=s+u.n*image;
        var slot=LIVE_NONE;
        for(var k=0u;k<count;k++){if(live_origin[kept[k]]==index){slot=kept[k];retained[k]=true;}}
        let im=ShellImage(live_kind[image],live_q[image]);
        if(im.kind==0u){
            if(slot!=LIVE_NONE){for(var k=0u;k<count;k++){if(kept[k]==slot){retained[k]=false;}}}
            continue;
        }
        let q=im.q;let position=q.yzw/(1.0+q.x);let bits=bitcast<vec3<u32>>(position);
        if(any(position<lower)||any(position>upper)){atomicOr(&live_scratch[3],2u);}
        let fence=select(0u,LIVE_HIGH,im.kind==3u);let seed=select(4u,0u,im.kind==3u);
        if(slot!=LIVE_NONE){
            let old=live_coordinates[slot];
            let moved_position=any(old.xyz!=bits);
            if(moved_position||(old.w&LIVE_HIGH)!=fence){
                live_coordinates[slot]=vec4<u32>(bits,slot|fence);
                live_change(slot,3u|seed|select(0u,8u,moved_position),old.xyz,bits);
            }
        }else{
            let top=atomicSub(&live_scratch[0],1u);
            if(top==0u||top>u.capacity){atomicAdd(&live_scratch[0],1u);atomicOr(&live_scratch[3],1u);continue;}
            slot=atomicLoad(&live_scratch[4u+top-1u]);
            live_coordinates[slot]=vec4<u32>(bits,slot|fence);live_origin[slot]=index;
            live_change(slot,2u|seed|8u,vec3<u32>(0u),bits);
        }
        atomicStore(&live_links[u.n+slot],chain);chain=slot+1u;
    }
    // Images that left both bands: their slots are freed (after this pass).
    for(var k=0u;k<count;k++){
        if(retained[k]){continue;}
        let slot=kept[k];let old=live_coordinates[slot];
        live_coordinates[slot]=vec4<u32>(old.xyz,LIVE_NONE);live_origin[slot]=LIVE_NONE;
        live_change(slot,1u|8u,old.xyz,vec3<u32>(0u));
        let at=atomicAdd(&live_scratch[1],1u);atomicStore(&live_scratch[4u+u.capacity+at],slot);
    }
    atomicStore(&live_links[s],chain);
}
