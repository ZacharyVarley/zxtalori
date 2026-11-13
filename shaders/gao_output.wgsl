// Count only completed, real, positive-volume tetrahedra. Symbolic ties may
// create exactly flat cells: omit those cells without perturbing coordinates.
@group(0) @binding(6) var<storage,read> prime_table:array<u32>;
@group(0) @binding(7) var<storage,read_write> gao_counters:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> gao_tets:array<vec4<u32>>;
fn gao_record_sign(c:Cache,sign:i32){}
fn gao_output_orientation(ids:vec3<u32>)->i32{return gao_numeric_sign(vec4<u32>(ids,gao_seed),GAO_NONE,0u);}
// The stars' summary (counts and failure flags), or with GAO_EMIT the output tetrahedra.
// One entry, so browsers compile the exact orientation sign once for both.
const GAO_EMIT=32u;
@compute @workgroup_size(128)
fn gao_output(@builtin(global_invocation_id) id_grid:vec3<u32>,@builtin(num_workgroups) id_groups:vec3<u32>){let id=vec3<u32>(id_grid.x+id_grid.y*id_groups.x*128u,0u,0u);
    if(id.x>=p.n){return;}gao_seed=id.x;let state=gao_states[gao_seed];let emit=(p.reserved&GAO_EMIT)!=0u;
    if(!emit){atomicMax(&gao_counters[3],state.z);atomicMax(&gao_counters[4],state.w);}
    if(state.x!=0u){if(!emit){atomicAdd(&gao_counters[1],1u);atomicOr(&gao_counters[2],1u<<state.x);}return;}
    if(!emit&&state.w<40u){atomicAdd(&gao_counters[5],1u);}
    for(var i=0u;i<state.z;i++){
        let t=gao_triangle(i);if((t.w&GAO_OUTPUT)==0u){continue;}
        let ids=gao_global_triangle(t);if(!emit&&any(ids>=vec3<u32>(p.n))){atomicOr(&gao_counters[2],1u<<6u);continue;}
        let orientation=gao_output_orientation(ids);
        if(emit){
            if(orientation>0){let at=atomicAdd(&gao_counters[6],1u);gao_tets[at]=vec4<u32>(coordinates[ids.x].w,coordinates[ids.y].w,coordinates[ids.z].w,coordinates[gao_seed].w)&vec4<u32>(0x7fffffffu);}
        } else if(orientation>0){atomicAdd(&gao_counters[0],1u);}else if(orientation<0){atomicOr(&gao_counters[2],1u<<7u);}
    }
    if(!emit&&gao_predicate_overflow){atomicOr(&gao_counters[2],1u<<8u);}
}
