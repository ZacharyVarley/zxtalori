// Live covering updates (covering.Live): the shell and fence images keep stable slots in
// the triangulation's coordinates; each slot carries its orbit code, and each source
// lists its slots. A move rewrites, adds or frees the slots of the moved sources' images
// (shell_live_images in shell.wgsl), the grid is listed again from current positions,
// and the stars those changes can reach are recomputed (gao_mark.wgsl, gao_live.wgsl).
//
// links: per source its first slot + 1 (0: none), then per slot the next slot + 1.
// scratch: [0] free slots, [1] released slots, [2] changes, [3] overflow bits; then the
// free stack, the released list (capacity each) and the changes (8 words each:
// slot, flags, old position, new position).
struct LiveParams { n:u32, total:u32, capacity:u32, count:u32, dim:u32, pad0:u32, pad1:u32, pad2:u32, lower:vec4<f32> } // lower.w: index cells per unit
@group(0) @binding(0) var<uniform> u:LiveParams;
@group(0) @binding(2) var<storage,read_write> coordinates:array<vec4<u32>>;
@group(0) @binding(10) var<storage,read_write> pending:array<atomic<u32>>; // [0]: count, then slots
@group(0) @binding(11) var<storage,read_write> launch:array<u32>;
@group(0) @binding(16) var<storage,read> site_origin:array<u32>;   // by input site, at the build
@group(0) @binding(17) var<storage,read_write> origin:array<u32>;  // by slot
@group(0) @binding(18) var<storage,read_write> links:array<atomic<u32>>;
@group(0) @binding(19) var<storage,read_write> scratch:array<atomic<u32>>;
@group(0) @binding(20) var<storage,read_write> index:array<atomic<u32>>;
@group(0) @binding(21) var<storage,read> moved:array<u32>;              // per source: nonzero if it moved
@group(0) @binding(22) var<storage,read_write> moved_list:array<atomic<u32>>; // [0] count, then sources
const NONE=0xffffffffu;
const HIGH=0x80000000u;
// The sorted build's slots become stable: coordinates' w is the slot (high bit: a fence
// site, no star), origin its code, and each source lists its slots; unused slots are
// free.
@compute @workgroup_size(128)
fn live_adopt(@builtin(global_invocation_id) id:vec3<u32>){
    let slot=id.x;if(slot>=u.capacity){return;}
    if(slot>=u.total){
        coordinates[slot]=vec4<u32>(0u,0u,0u,NONE);origin[slot]=NONE;
        atomicStore(&scratch[4u+slot-u.total],slot);return;
    }
    let site=coordinates[slot].w;let code=site_origin[site];origin[slot]=code;
    coordinates[slot].w=slot|select(0u,HIGH,site>=u.count);
    let previous=atomicExchange(&links[code%u.n],slot+1u);atomicStore(&links[u.n+slot],previous);
}
// Slots freed by the last move return to the free stack (one workgroup).
@compute @workgroup_size(256)
fn live_release(@builtin(local_invocation_index) lane:u32){
    let count=atomicLoad(&scratch[1]);let top=atomicLoad(&scratch[0]);
    for(var i=lane;i<count;i+=256u){atomicStore(&scratch[4u+top+i],atomicLoad(&scratch[4u+u.capacity+i]));}
    workgroupBarrier();
    if(lane==0u){atomicStore(&scratch[0],top+count);atomicStore(&scratch[1],0u);}
}
// The moved sources, listed.
@compute @workgroup_size(128)
fn live_moved(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<u.n&&moved[id.x]!=0u){atomicStore(&moved_list[1u+atomicAdd(&moved_list[0],1u)],id.x);}}
@compute @workgroup_size(1)
fn live_launch_moved(){let count=atomicLoad(&moved_list[0]);launch[8]=min(count,32768u);launch[9]=max(1u,(count+32767u)/32768u);launch[10]=1u;}
// Indirect sizes: words 0..2 one workgroup per pending star, words 4..6 one per change,
// words 8..10 one per moved source.
@compute @workgroup_size(1)
fn live_launch(){let count=atomicLoad(&pending[0]);launch[0]=min(count,32768u);launch[1]=max(1u,(count+32767u)/32768u);launch[2]=1u;}
@compute @workgroup_size(1)
fn live_launch_changes(){let count=min(atomicLoad(&scratch[2]),u.capacity);launch[4]=min(count,32768u);launch[5]=max(1u,(count+32767u)/32768u);launch[6]=1u;}
// The jobs' shell grid (shaders/local.wgsl): heads per cell, next per slot, then each
// slot's code; free slots are listed nowhere.
@compute @workgroup_size(128)
fn live_index_clear(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<u.dim*u.dim*u.dim){atomicStore(&index[id.x],NONE);}}
@compute @workgroup_size(128)
fn live_index(@builtin(global_invocation_id) id:vec3<u32>){
    let slot=id.x;if(slot>=u.capacity){return;}
    let cells=u.dim*u.dim*u.dim;let c=coordinates[slot];
    atomicStore(&index[cells+u.capacity+slot],select(origin[slot],NONE,c.w==NONE));
    if(c.w==NONE){return;}
    let b=vec3<u32>(clamp(floor((bitcast<vec3<f32>>(c.xyz)-u.lower.xyz)*u.lower.w),vec3<f32>(0.0),vec3<f32>(f32(u.dim-1u))));
    let previous=atomicExchange(&index[b.x+u.dim*(b.y+u.dim*b.z)],slot);
    atomicStore(&index[cells+slot],previous);
}
