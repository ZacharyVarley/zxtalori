// Count only completed, real, positive-volume tetrahedra. Symbolic ties may
// create exactly flat cells: omit those cells without perturbing coordinates.
@group(0) @binding(6) var<storage,read> prime_table:array<u32>;
@group(0) @binding(7) var<storage,read_write> gao_counters:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> gao_tets:array<vec4<u32>>;
fn gao_record_sign(c:Cache,sign:i32){}
fn gao_output_orientation(ids:vec3<u32>)->i32{return gao_numeric_sign(vec4<u32>(ids,gao_seed),GAO_NONE,0u);}
@compute @workgroup_size(128)
fn gao_summary(@builtin(global_invocation_id) id:vec3<u32>){
    if(id.x>=p.n){return;}gao_seed=id.x;let state=gao_states[gao_seed];atomicMax(&gao_counters[3],state.z);atomicMax(&gao_counters[4],state.w);
    if(state.x!=0u){atomicAdd(&gao_counters[1],1u);atomicOr(&gao_counters[2],1u<<state.x);return;}
    if(state.w<40u){atomicAdd(&gao_counters[5],1u);}
    for(var i=0u;i<state.z;i++){
        let t=gao_triangle(i);if((t.w&GAO_OUTPUT)==0u){continue;}
        let ids=gao_global_triangle(t);if(any(ids>=vec3<u32>(p.n))){atomicOr(&gao_counters[2],1u<<6u);continue;}
        let orientation=gao_output_orientation(ids);if(orientation>0){atomicAdd(&gao_counters[0],1u);}else if(orientation<0){atomicOr(&gao_counters[2],1u<<7u);}
    }
    if(gao_predicate_overflow){atomicOr(&gao_counters[2],1u<<8u);}
}
@compute @workgroup_size(128)
fn gao_emit(@builtin(global_invocation_id) id:vec3<u32>){
    if(id.x>=p.n){return;}gao_seed=id.x;let state=gao_states[gao_seed];if(state.x!=0u){return;}
    for(var i=0u;i<state.z;i++){
        let t=gao_triangle(i);if((t.w&GAO_OUTPUT)==0u){continue;}
        let ids=gao_global_triangle(t);if(gao_output_orientation(ids)<=0){continue;}
        let at=atomicAdd(&gao_counters[6],1u);gao_tets[at]=vec4<u32>(coordinates[ids.x].w,coordinates[ids.y].w,coordinates[ids.z].w,coordinates[gao_seed].w)&vec4<u32>(0x7fffffffu);
    }
}
