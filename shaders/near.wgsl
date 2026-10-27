// Near-field Quey forces (src/near.zig): counting sorts of points into a uniform grid of
// the stereographic chart, deterministic within each cell. They sort the shell images
// within the cutoff of the zone (sites, from shell_build) in row-major cell order, and the
// sources themselves in Morton cell order; the forces are summed by near_force in
// pairs.wgsl.
// flags: bit 0 the count is the shell build's; bit 1 Morton order (dim a power of two, at most 1024).
struct SortParams { count:u32, capacity:u32, dim:u32, flags:u32, lower:vec4<f32> } // lower.w: cells per unit
@group(0) @binding(0) var<uniform> u:SortParams;
@group(0) @binding(1) var<storage,read> input:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> offsets:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> sorted:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> cursors:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> info:array<u32>;
@group(0) @binding(6) var<storage,read> quats:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> points:array<vec4<f32>>;
struct ImageParams { n:u32, capacity:u32, pad0:u32, pad1:u32 }
@group(0) @binding(8) var<uniform> v:ImageParams;
@group(0) @binding(9) var<storage,read> operators:array<vec4<f32>>;
@group(0) @binding(10) var<storage,read> origin:array<u32>;                 // per site: source + n*(2*op+negative)
@group(0) @binding(11) var<storage,read_write> images:array<vec4<f32>>;     // by sorted site: its quaternion
@group(0) @binding(12) var<storage,read_write> codes:array<u32>;            // by sorted site: source, high bit its identity image
// The number of points: u.count, or the shell build's site count (clamped to capacity).
fn items()->u32{if((u.flags&1u)!=0u){return min(info[0],u.capacity);}return u.count;}
fn spread(v:u32)->u32{var x=v&0x3ffu;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;return (x|(x<<2u))&0x09249249u;}
fn cell_of(x:vec3<f32>)->u32{
    let c=vec3<u32>(clamp(floor((x-u.lower.xyz)*u.lower.w),vec3<f32>(0.0),vec3<f32>(f32(u.dim-1u))));
    if((u.flags&2u)!=0u){return spread(c.x)|(spread(c.y)<<1u)|(spread(c.z)<<2u);}
    return c.x+u.dim*(c.y+u.dim*c.z);
}
// Each source's stereographic position, its index in w (as the shell build places it).
@compute @workgroup_size(128)
fn near_points(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if(i>=u.count){return;}let q=quats[i];points[i]=vec4<f32>(q.yzw/(1.0+q.x),bitcast<f32>(i));
}
@compute @workgroup_size(128)
fn near_clear(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<=u.dim*u.dim*u.dim){atomicStore(&offsets[id.x],0u);}}
@compute @workgroup_size(128)
fn near_count(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<items()){atomicAdd(&offsets[cell_of(input[id.x].xyz)],1u);}}
var<workgroup> prefix:array<u32,256>;
var<workgroup> total:u32;
// One workgroup: counts become exclusive offsets (and the scatter cursors).
@compute @workgroup_size(256)
fn near_prefix(@builtin(local_invocation_index) lane:u32){
    if(lane==0u){total=0u;}workgroupBarrier();let count=u.dim*u.dim*u.dim;
    for(var base=0u;base<count;base+=256u){
        let i=base+lane;var value=0u;if(i<count){value=atomicLoad(&offsets[i]);}prefix[lane]=value;workgroupBarrier();
        for(var stride=1u;stride<256u;stride*=2u){var add=0u;if(lane>=stride){add=prefix[lane-stride];}workgroupBarrier();prefix[lane]+=add;workgroupBarrier();}
        let begin=total+prefix[lane]-value;if(i<count){atomicStore(&offsets[i],begin);atomicStore(&cursors[i],begin);}workgroupBarrier();
        if(lane==0u){total+=prefix[255];}workgroupBarrier();
    }
    if(lane==0u){atomicStore(&offsets[count],total);}
}
// The point's index goes in w.
@compute @workgroup_size(128)
fn near_scatter(@builtin(global_invocation_id) id:vec3<u32>){
    if(id.x>=items()){return;}let x=input[id.x];
    sorted[atomicAdd(&cursors[cell_of(x.xyz)],1u)]=vec4<f32>(x.xyz,bitcast<f32>(id.x));
}
fn qmul(a:vec4<f32>,b:vec4<f32>)->vec4<f32>{return vec4<f32>(a.x*b.x-dot(a.yzw,b.yzw),a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));}
// The sorted sites' quaternions, rebuilt from their codes exactly as quey_partial builds
// images, and sources, laid out in sorted order so near_force reads them in runs.
@compute @workgroup_size(128)
fn near_images(@builtin(global_invocation_id) id:vec3<u32>){
    let k=id.x;if(k>=min(info[0],v.capacity)){return;}
    let code=origin[bitcast<u32>(sorted[k].w)];let source=code%v.n;let image=code/v.n;let op=image/2u;
    var q=select(normalize(qmul(quats[source],operators[op])),quats[source],op==0u);
    if((image&1u)!=0u){q=-q;}
    images[k]=q;codes[k]=source|select(0u,0x80000000u,image==0u);
}
// Index order within each cell, so sums run in the same order every pass (crowded
// cells, rare, keep the scatter's order).
@compute @workgroup_size(128)
fn near_order(@builtin(global_invocation_id) id:vec3<u32>){
    if(id.x>=u.dim*u.dim*u.dim){return;}
    let begin=atomicLoad(&offsets[id.x]);let end=atomicLoad(&offsets[id.x+1u]);
    if(end-begin>128u){return;}
    for(var at=begin+1u;at<end;at++){
        let value=sorted[at];var before=at;
        while(before>begin&&bitcast<u32>(sorted[before-1u].w)>bitcast<u32>(value.w)){sorted[before]=sorted[before-1u];before--;}
        sorted[before]=value;
    }
}
