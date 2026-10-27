// cudaDelaunay.cu::PostProcessing and post_processing_kernel.
// Preserve the source's 128 logical search streams. A 32-lane WebGPU group
// multiplexes four streams per lane to keep exact scratch in shared memory.
@group(0) @binding(6) var<storage,read> prime_table:array<u32>;
@group(0) @binding(7) var<storage,read_write> gao_counters:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> gao_records:array<Cache>;
fn gao_record_sign(c:Cache,sign:i32){
    if((p.reserved&2u)==0u||sign!=0){return;}let at=atomicAdd(&gao_counters[20],1u);
    if(at<arrayLength(&gao_records)){gao_records[at]=Cache(c.ids,vec4<u32>(c.more.xyz,gao_seed));}
}
var<workgroup> gao_post_meta:vec4<u32>;
var<workgroup> gao_post_active:u32;
var<workgroup> gao_post_ids:vec3<u32>;
var<workgroup> gao_post_range:GaoRange;
var<workgroup> gao_post_sphere:GaoSphere;
var<workgroup> gao_post_double:GaoDoubleSphere;
var<workgroup> gao_post_candidates:array<u32,POST_LANES>;
var<workgroup> gao_post_distances:array<f32,POST_LANES>;
var<workgroup> gao_post_remove:array<u32,GAO_MAX_T>;
var<workgroup> gao_post_cold:array<u32,POST_LANES>;
var<private> gao_search_tests:u32;
var<private> gao_classification_tests:u32;
var<private> gao_search_queries:u32;
var<private> gao_search_full:u32;
var<private> gao_search_largest:u32;
var<private> gao_search_uncertain:u32;
var<private> gao_search_double_uncertain:u32;
fn gao_any_deferred()->bool{
    gao_post_cold[gao_lane]=u32(gao_defer_exact);workgroupBarrier();
    for(var stride=POST_LANES/2u;stride>0u;stride/=2u){
        if(gao_lane<stride){gao_post_cold[gao_lane]|=gao_post_cold[gao_lane+stride];}workgroupBarrier();
    }
    return workgroupUniformLoad(&gao_post_cold[0])!=0u;
}
fn gao_mark_deferred(face:u32){
    if(gao_lane==0u){var t=gao_triangle(face);t.w|=GAO_INEXACT;gao_write_triangle(face,t);gao_status=5u;}
}
fn gao_post_active_triangle()->u32{
    for(var i=0u;i<gao_nt;i++){var t=gao_triangle(i);if((t.w&GAO_PROCESSED)==0u||(t.w&GAO_INEXACT)!=0u){t.w=(t.w|GAO_PROCESSED)&~GAO_INEXACT;gao_write_triangle(i,t);return i;}}
    return GAO_NONE;
}
fn gao_halfspace_post(id:u32){
    let vertex=gao_add_vertex(gao_encode(id));if(gao_status!=0u){return;}gao_nr=0u;
    var i=0u;while(i<gao_nt){if(gao_post_remove[i]!=0u){gao_remove_triangle(i);gao_post_remove[i]=gao_post_remove[gao_nt];}else{i++;}}
    let first=gao_compute_boundary();if(first==GAO_NONE){gao_status=2u;return;}
    var current=first;var allowance=GAO_MAX_P;
    loop {
        if(allowance==0u||current>=gao_nv){gao_status=2u;return;}allowance--;
        let next=gao_boundary_get(current);if(next==GAO_NONE){gao_status=2u;return;}
        gao_add_triangle(vertex,current,next,p.n);if(gao_status!=0u){return;}current=next;if(current==first){break;}
    }
}
@compute @workgroup_size(POST_LANES)
fn gao_post(@builtin(workgroup_id) wid:vec3<u32>,@builtin(num_workgroups) grid:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let seed=wid.x+wid.y*grid.x;
    if(seed>=p.n){return;}gao_seed=seed;gao_lane=lane;gao_full=(p.reserved&GAO_FULL)!=0u;let state=gao_states[gao_seed];
    if(lane==0u){
        gao_post_meta.z=state.x;
        // A filtered pass can exhaust this same 64-vertex capacity. Preserve
        // its complete pending mesh for the larger retry instead of attempting
        // a second update in the already-exhausted buffer.
        if(!GAO_FILTERED_ONLY&&state.x==6u){gao_states[gao_seed].x=1u;}
    }
    workgroupBarrier();let incoming=workgroupUniformLoad(&gao_post_meta).z;
    if(incoming==0u||(!GAO_FILTERED_ONLY&&incoming==6u)){return;}
    gao_origin=gao_point(gao_seed);
    for(var i=lane;i<state.z;i+=POST_LANES){gao_shared_triangles[i]=gao_triangles[i*p.n+gao_seed];}
    for(var i=lane;i<state.y;i+=POST_LANES){gao_shared_indices[i]=gao_indices[i*p.n+gao_seed];}
    if(lane==0u){gao_nv=state.y;gao_nt=state.z;gao_nr=0u;gao_status=0u;gao_noutput=0u;gao_post_meta=vec4<u32>(gao_nv,gao_nt,0u,0u);}
    workgroupBarrier();
    loop {
        if(lane==0u){
            gao_post_active=gao_post_active_triangle();
            if(gao_post_active!=GAO_NONE&&gao_status==0u&&gao_post_meta.w<500u){gao_post_ids=gao_global_triangle(gao_triangle(gao_post_active));gao_post_sphere=gao_prepare(gao_post_ids);gao_post_double=gao_prepare_double(gao_post_ids,gao_post_sphere);gao_post_sphere=gao_refined_sphere(gao_post_double);gao_post_range=gao_range(gao_post_sphere);}
            else{gao_post_active=GAO_NONE;}
        }
        workgroupBarrier();let face=workgroupUniformLoad(&gao_post_active);if(face==GAO_NONE){break;}
        let low=gao_post_range.low;let high=gao_post_range.high;let extent=high-low+vec3<i32>(1);
        // Source chooses the longest axis as its contiguous scanline axis.
        var axis=0u;if(extent.y>extent.x){axis=1u;}if(extent.z>extent[axis]){axis=2u;}
        let j=(axis+1u)%3u;let k=(axis+2u)%3u;let plane=extent[axis]*extent[j];let count=plane*extent[k];
        if(lane==0u){gao_search_queries++;gao_search_largest=max(gao_search_largest,u32(count));if(count==i32(p.dim*p.dim*p.dim)){gao_search_full++;}}
        var candidate=GAO_NONE;var distance=bitcast<f32>(0x7f800000u^p.zero);var found=0u;
        let streams=128u/POST_LANES;var valid_streams=0u;
        if(u32(count)>lane){valid_streams=min(streams,(u32(count)-1u-lane)/POST_LANES+1u);}
        let all_found=(1u<<valid_streams)-1u;
        for(var at=i32(lane);at<count;at+=i32(POST_LANES)){
            let stream=(u32(at)/POST_LANES)%streams;
            if((found&(1u<<stream))!=0u){continue;}
            var cell=low;cell[k]+=at/plane;cell[j]+=(at%plane)/extent[axis];cell[axis]+=at%extent[axis];
            let bucket=gao_cell_id(cell);
            if(gao_cells[bucket]==gao_cells[bucket+1u]||gao_cell_outside(gao_post_sphere,cell)){continue;}
            for(var k=gao_cells[bucket];k<gao_cells[bucket+1u];k++){let id=gao_slot(k);
                if(id==gao_seed||any(gao_post_ids==vec3<u32>(id))){continue;}
                gao_search_tests++;
                let fast=gao_sphere_test(gao_post_sphere,id);
                var inside=fast.x;
                if(!fast.y){
                    gao_search_uncertain++;let wide_sign=gao_prepared_double_sign(gao_post_double,id);
                    if(wide_sign==2){gao_search_double_uncertain++;inside=gao_inside_exact(gao_post_ids,id);}else{inside=wide_sign<0;}
                }
                if(inside){
                    found|=1u<<stream;let delta=gao_point(id)-gao_origin;let d2=dot(delta,delta);
                    if(d2<distance||(d2==distance&&id<candidate)){candidate=id;distance=d2;}break;
                }
            }
            if(found==all_found){break;}
        }
        gao_post_candidates[lane]=candidate;gao_post_distances[lane]=distance;
        workgroupBarrier();
        for(var stride=POST_LANES/2u;stride>0u;stride/=2u){
            if(lane<stride){let b=lane+stride;let db=gao_post_distances[b];let da=gao_post_distances[lane];if(db<da||(db==da&&gao_post_candidates[b]<gao_post_candidates[lane])){gao_post_distances[lane]=db;gao_post_candidates[lane]=gao_post_candidates[b];}}
            workgroupBarrier();
        }
        let chosen=workgroupUniformLoad(&gao_post_candidates[0]);
        if(GAO_FILTERED_ONLY&&gao_any_deferred()){gao_mark_deferred(face);workgroupBarrier();break;}
        if(chosen!=GAO_NONE){
            for(var i=lane;i<gao_post_meta.y;i+=POST_LANES){gao_classification_tests++;let ids=gao_global_triangle(gao_triangle(i));gao_post_remove[i]=u32(i==face||any(ids==vec3<u32>(chosen))||gao_inside_prepared(ids,chosen));}
            workgroupBarrier();
            if(GAO_FILTERED_ONLY&&gao_any_deferred()){gao_mark_deferred(face);workgroupBarrier();break;}
            if(lane==0u){
                gao_halfspace_post(chosen);
                // The active face was marked processed before search. Keep it
                // pending if vertex capacity prevented the discovered update.
                if(gao_status==1u&&face<gao_nt){var t=gao_triangle(face);t.w|=GAO_INEXACT;gao_write_triangle(face,t);}
                gao_post_meta.w++;gao_post_meta.x=gao_nv;gao_post_meta.y=gao_nt;gao_post_meta.z=gao_status;
            }
            workgroupBarrier();
        }
    }
    if(lane==0u){
        if(gao_post_meta.w==500u){gao_status=3u;}
        if(gao_status==0u){
            // The whole star stays (see gao_finish_main); output triangles carry GAO_OUTPUT.
            for(var i=0u;i<gao_nv;i++){gao_write_index(i,gao_index(i)&~GAO_GREATER);}
        }
        if(GAO_FILTERED_ONLY&&gao_status==1u){gao_status=6u;}
        gao_post_meta=vec4<u32>(gao_nv,gao_nt,gao_status,gao_post_meta.w);gao_states[gao_seed]=vec4<u32>(gao_status,gao_nv,gao_nt,state.w+gao_post_meta.w);
    }
    workgroupBarrier();
    for(var i=lane;i<gao_post_meta.y;i+=POST_LANES){gao_triangles[i*p.n+gao_seed]=gao_shared_triangles[i];}
    for(var i=lane;i<gao_post_meta.x;i+=POST_LANES){gao_indices[i*p.n+gao_seed]=gao_shared_indices[i];}
    if(gao_exact_count!=0u){atomicAdd(&gao_counters[7],gao_exact_count);}
    if((p.reserved&2u)!=0u){
        if(gao_search_tests!=0u){atomicAdd(&gao_counters[8],gao_search_tests);}
        if(gao_classification_tests!=0u){atomicAdd(&gao_counters[9],gao_classification_tests);}
        if(lane==0u){atomicMax(&gao_counters[10],gao_search_largest);atomicAdd(&gao_counters[11],gao_search_full);atomicAdd(&gao_counters[12],gao_search_queries);}
        if(gao_search_uncertain!=0u){atomicAdd(&gao_counters[13],gao_search_uncertain);}
        if(gao_search_double_uncertain!=0u){atomicAdd(&gao_counters[14],gao_search_double_uncertain);}
        if(gao_exact_zeros!=0u){atomicAdd(&gao_counters[15],gao_exact_zeros);}
        if(gao_prime_sum!=0u){atomicAdd(&gao_counters[16],gao_prime_sum);atomicMax(&gao_counters[17],gao_prime_max);}
        if(gao_exact_orientations!=0u){atomicAdd(&gao_counters[18],gao_exact_orientations);}
        if(gao_exact_spheres!=0u){atomicAdd(&gao_counters[19],gao_exact_spheres);}
    }
    if(gao_predicate_overflow){atomicOr(&gao_counters[2],1u<<8u);}
}
