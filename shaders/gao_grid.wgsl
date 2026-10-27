// Physically sort sites by uniform-grid bucket, as Gao's SORTING_POINTS path.
struct GaoParams{n:u32,dim:u32,zero:u32,reserved:u32,lower:vec4<f32>,upper:vec3<f32>,seeds:u32,scale:vec4<f32>}
@group(0) @binding(0) var<uniform> p:GaoParams;
@group(0) @binding(1) var<storage,read> input:array<vec4<u32>>;
@group(0) @binding(2) var<storage,read_write> offsets:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> sorted:array<vec4<u32>>;
@group(0) @binding(4) var<storage,read_write> cursors:array<atomic<u32>>;
var<workgroup> prefix:array<u32,128>;
var<workgroup> total:u32;
var<workgroup> occupied:atomic<u32>;
fn bucket(i:u32)->u32{let q=bitcast<vec4<f32>>(input[i]).xyz;let c=vec3<u32>(clamp(floor((q-p.lower.xyz)*p.scale.xyz),vec3<f32>(0),vec3<f32>(f32(p.dim-1u))));return c.x+p.dim*(c.y+p.dim*c.z);}
@compute @workgroup_size(128)
fn gao_grid_clear(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<=p.dim*p.dim*p.dim){atomicStore(&offsets[id.x],0u);}}
@compute @workgroup_size(128)
fn gao_grid_count(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<p.n){atomicAdd(&offsets[bucket(id.x)],1u);}}
@compute @workgroup_size(128)
fn gao_grid_prefix(@builtin(local_invocation_index) lane:u32){
    if(lane==0u){total=0u;atomicStore(&occupied,0u);}workgroupBarrier();let count=p.dim*p.dim*p.dim;
    for(var base=0u;base<count;base+=128u){
        let i=base+lane;var value=0u;if(i<count){value=atomicLoad(&offsets[i]);}prefix[lane]=value;if(value>0u){atomicAdd(&occupied,1u);}workgroupBarrier();
        for(var stride=1u;stride<128u;stride*=2u){var add=0u;if(lane>=stride){add=prefix[lane-stride];}workgroupBarrier();prefix[lane]+=add;workgroupBarrier();}
        let begin=total+prefix[lane]-value;if(i<count){atomicStore(&offsets[i],begin);atomicStore(&cursors[i],begin);}workgroupBarrier();
        if(lane==0u){total+=prefix[127];}workgroupBarrier();
    }
    // cursors[count]: occupied cells, which size the next grid.
    if(lane==0u){atomicStore(&offsets[count],total);atomicStore(&cursors[count],atomicLoad(&occupied));}
}
// Slot of each site in the sorted coordinates (live updates rewrite moved sites in place).
@group(0) @binding(5) var<storage,read_write> slots:array<u32>;
@compute @workgroup_size(128)
fn gao_grid_slots(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<p.n){slots[sorted[id.x].w]=id.x;}}
@compute @workgroup_size(128)
fn gao_grid_scatter(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<p.n){let at=atomicAdd(&cursors[bucket(id.x)],1u);sorted[at]=vec4<u32>(input[id.x].xyz,id.x);}}
fn gao_grid_less(a:vec4<u32>,b:vec4<u32>)->bool{
    let x=bitcast<vec3<f32>>(a.xyz);let y=bitcast<vec3<f32>>(b.xyz);
    return x.x<y.x||(x.x==y.x&&(x.y<y.y||(x.y==y.y&&(x.z<y.z||(x.z==y.z&&a.w<b.w)))));
}
// Stable spatial order inside ordinary buckets removes scatter-scheduling
// variation from the main initialization. Dense exceptional buckets remain
// valid without this optional locality optimization; avoid quadratic work.
@compute @workgroup_size(128)
fn gao_grid_order(@builtin(global_invocation_id) id:vec3<u32>){
    if(id.x>=p.dim*p.dim*p.dim){return;}
    let begin=atomicLoad(&offsets[id.x]);let end=atomicLoad(&offsets[id.x+1u]);
    if(end-begin>64u){return;}
    for(var at=begin+1u;at<end;at++){
        let value=sorted[at];var before=at;
        while(before>begin&&gao_grid_less(value,sorted[before-1u])){sorted[before]=sorted[before-1u];before--;}
        sorted[before]=value;
    }
}
// Listed grid (GAO_LISTED, live updates): slots keep their place in `input` (the
// coordinates themselves); each live slot (w != NONE) is listed after the cell offsets,
// by the cell of its current position. gao_grid_clear and gao_grid_prefix run between.
@compute @workgroup_size(128)
fn gao_list_count(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<p.n&&input[id.x].w!=0xffffffffu){atomicAdd(&offsets[bucket(id.x)],1u);}}
@compute @workgroup_size(128)
fn gao_list_scatter(@builtin(global_invocation_id) id:vec3<u32>){
    if(id.x>=p.n||input[id.x].w==0xffffffffu){return;}
    let at=atomicAdd(&cursors[bucket(id.x)],1u);atomicStore(&offsets[p.dim*p.dim*p.dim+1u+at],id.x);
}
