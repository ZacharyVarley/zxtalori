// cudaDelaunay.cu::dt_init, ballsearch, MainPhase and DT_kernel.
// Source commit 6e20ee49d3bf13914f1ce87b9194bdde6f2b3273 (MIT).
fn gao_dt_init(){
    gao_nv=4u;gao_nt=0u;gao_nr=0u;gao_noutput=0u;gao_status=0u;
    for(var i=0u;i<4u;i++){gao_write_index(i,gao_encode(p.n+i));}
    // Same oriented initial tetrahedron faces as CUDA. Compute ordering tags
    // rather than assuming all data coordinates are inside the unit cube.
    gao_add_triangle(0u,2u,3u,p.n);gao_add_triangle(0u,1u,2u,p.n);
    gao_add_triangle(1u,0u,3u,p.n);gao_add_triangle(2u,1u,3u,p.n);
    let origin_cell=gao_cell(gao_origin);
    let directions=array<vec3<i32>,7>(vec3<i32>(0),vec3<i32>(1,0,0),vec3<i32>(-1,0,0),vec3<i32>(0,1,0),vec3<i32>(0,-1,0),vec3<i32>(0,0,1),vec3<i32>(0,0,-1));
    for(var i=0u;i<7u;i++){
        let cell=origin_cell+directions[i];if(any(cell<vec3<i32>(0))||any(cell>=vec3<i32>(i32(p.dim)))){continue;}
        let bucket=gao_cell_id(cell);
        for(var k=gao_cells[bucket];k<gao_cells[bucket+1u];k++){let id=gao_slot(k);
            if(id==gao_seed){continue;}let changed=gao_halfspace_intersection(gao_encode(id),p.n);
            if(gao_status!=0u){return;}
        }
    }
}
struct GaoSearch{candidate:u32,trusted:bool}
fn gao_ballsearch(ids:vec3<u32>)->GaoSearch{
    let sphere=gao_prepare(ids);let range=gao_range(sphere);var lo=range.start;var hi=range.start;
    var previous_low=range.start+vec3<i32>(1);var previous_high=range.start-vec3<i32>(1);
    var cells=0u;var trusted=true;
    loop {
        for(var i=lo.x;i<=hi.x;i++){for(var j=lo.y;j<=hi.y;j++){for(var k=hi.z;k>=lo.z;k--){
            if(cells>200u){return GaoSearch(GAO_NONE,false);}
            // Cells of the previous box were searched already: jump over that block
            // of this z-column (same visiting order, without iterating it).
            let cell=vec3<i32>(i,j,k);if(all(cell>=previous_low)&&all(cell<=previous_high)){k=previous_low.z;continue;}
            let bucket=gao_cell_id(cell);let begin=gao_cells[bucket];let end=gao_cells[bucket+1u];
            if(end>begin){cells++;}
            for(var k=begin;k<end;k++){let id=gao_slot(k);
                if(id==gao_seed||any(ids==vec3<u32>(id))){continue;}
                let verdict=gao_sphere_test(sphere,id);trusted=trusted&&verdict.y;
                if(verdict.x){return GaoSearch(id,trusted);}
            }
        }}}
        previous_low=lo;previous_high=hi;lo-=vec3<i32>(1);hi+=vec3<i32>(1);
        if(all(lo<range.low)&&all(hi>range.high)){break;}
        lo=max(lo,range.low);hi=min(hi,range.high);
    }
    return GaoSearch(GAO_NONE,trusted);
}
@compute @workgroup_size(128)
fn gao_main(@builtin(global_invocation_id) id:vec3<u32>){
    if(id.x>=p.n){return;}
    if(p.scale.w>0.5&&gao_states[id.x].x!=GAO_PENDING){return;}
    if(coordinates[id.x].w>=p.seeds){gao_states[id.x]=vec4<u32>(0u);return;}
    gao_seed=id.x;gao_origin=gao_point(gao_seed);gao_full=(p.reserved&GAO_FULL)!=0u;gao_dt_init();var cycles=0u;
    if(gao_status==0u){
        while(cycles<40u){
            let face=gao_update_active();if(face==GAO_NONE){break;}
            let result=gao_ballsearch(gao_global_triangle(gao_triangle(face)));
            if(result.candidate==GAO_NONE){gao_finish_search(face,result.trusted);}
            else{
                gao_remove_triangle(face);
                if(!gao_halfspace_intersection(gao_encode(result.candidate),p.n)&&gao_status==0u){var t=gao_triangle(gao_nt-1u);t.w|=GAO_INEXACT;gao_write_triangle(gao_nt-1u,t);}
                cycles++;
            }
            if(gao_status!=0u){break;}
        }
    }
    gao_finish_main(cycles);gao_states[gao_seed]=vec4<u32>(gao_status,gao_nv,gao_nt,cycles);
}
