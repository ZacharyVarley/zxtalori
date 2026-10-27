// Live restar (covering.Live.move): one workgroup recomputes one pending star with its
// mesh in workgroup memory, as a Voronoi cell clipped by its nearby sites.
//
// The sites within radius r of the seed are gathered from the grid and inserted
// nearest first (Gao's topology and exact conflict test, lanes classifying faces in
// parallel). The seed's Voronoi cell is the star's dual; its vertices are the face
// circumcentres. A face's sphere passes through the seed, so a face whose circumcentre
// is within r/2 of the seed has its sphere inside the gathered ball and is final once
// every candidate is in. Insertion stops at the first candidate farther than twice the
// largest such distance (it cannot cut the cell). Faces with larger spheres are then
// searched like gao_post: the grid cells their sphere meets, in parallel, exactly; a
// site found is inserted, and the search goes on until every face is certified. A star
// that overflows the workgroup mesh stays GAO_PENDING for gao_main.
// The pending list ([0] count, then slots) is bound in gao_records' place (binding 8),
// which keeps the stage within WebGPU's eight storage buffers; the dispatch clears the
// statistics bit, so gao_record_sign never writes it.
fn live_pending(j:u32)->u32{let c=gao_records[j/8u];let w=j%8u;return select(c.more[w%4u],c.ids[w%4u],w<4u);}
const LIVE_CANDIDATES=256u;
const LIVE_SEARCHES=96u;
var<workgroup> live_ids:array<u32,LIVE_CANDIDATES>;
var<workgroup> live_distance:array<f32,LIVE_CANDIDATES>;
var<workgroup> live_order:array<u32,LIVE_CANDIDATES>;
var<workgroup> live_count:atomic<u32>;
var<workgroup> live_hits:atomic<u32>;
var<workgroup> live_found:atomic<u32>;
var<workgroup> live_reach:array<f32,POST_LANES>;
var<workgroup> live_verdict:u32;
var<workgroup> live_uniform:vec2<u32>;
// The distance from the seed to face i's circumcentre, rounded up; inf when the face is
// unbounded or ill-conditioned.
fn live_face_reach(i:u32)->f32{
    let s=gao_prepare(gao_global_triangle(gao_triangle(i)));
    let co=vec4<f32>(s.coefficient.z,-s.coefficient.y,s.coefficient.x,s.coefficient.w);
    let error=4e-6*max(max(s.bound.x,s.bound.y),max(s.bound.z,s.bound.w));
    if(!(abs(co.w)>2.0*error)){return bitcast<f32>(0x7f800000u^p.zero);}
    return length(co.xyz/(2.0*co.w))*(1.0+1e-3)+1e-30;
}
// The largest face reach over the star.
fn live_rho(lane:u32,nt:u32)->f32{
    var largest=0.0;
    for(var i=lane;i<nt;i+=POST_LANES){largest=max(largest,live_face_reach(i));}
    live_reach[lane]=largest;workgroupBarrier();
    for(var stride=POST_LANES/2u;stride>0u;stride/=2u){
        if(lane<stride){live_reach[lane]=max(live_reach[lane],live_reach[lane+stride]);}workgroupBarrier();
    }
    return workgroupUniformLoad(&live_reach[0]);
}
// Inserts `candidate` if its halfspace cuts the cell. Returns 0 (no cut), 1 (inserted)
// or 2 (the mesh failed: capacity or topology).
fn live_insert(lane:u32,candidate:u32)->u32{
    let nt=workgroupUniformLoad(&gao_post_meta).y;
    if(lane==0u){atomicStore(&live_hits,0u);}
    workgroupBarrier();
    for(var i=lane;i<nt;i+=POST_LANES){
        let inside=gao_inside_prepared(gao_global_triangle(gao_triangle(i)),candidate);
        gao_post_remove[i]=u32(inside);if(inside){atomicAdd(&live_hits,1u);}
    }
    workgroupBarrier();
    if(lane==0u){live_verdict=atomicLoad(&live_hits);}
    if(workgroupUniformLoad(&live_verdict)==0u){return 0u;}
    if(lane==0u){
        gao_halfspace_post(candidate);
        gao_post_meta=vec4<u32>(gao_nv,gao_nt,gao_status,0u);
    }
    return select(1u,2u,workgroupUniformLoad(&gao_post_meta).z!=0u);
}
// The first face not yet certified: unsearched and its sphere not inside the ball of
// radius `radius` around the seed. NONE when every face is certified.
fn live_open_face(lane:u32,radius:f32)->u32{
    let nt=workgroupUniformLoad(&gao_post_meta).y;
    if(lane==0u){atomicStore(&live_found,GAO_NONE);}
    workgroupBarrier();
    for(var i=lane;i<nt;i+=POST_LANES){
        if((gao_triangle(i).w&GAO_PROCESSED)==0u&&!(2.0*live_face_reach(i)<=radius)){atomicMin(&live_found,i);}
    }
    workgroupBarrier();
    if(lane==0u){live_verdict=atomicLoad(&live_found);}
    return workgroupUniformLoad(&live_verdict);
}
// The site nearest the seed strictly inside face f's sphere, or NONE: the grid cells
// the sphere meets, in parallel.
var<workgroup> live_best:array<vec2<u32>,POST_LANES>;
fn live_search(lane:u32,f:u32)->u32{
    let ids=gao_global_triangle(gao_triangle(f));let sphere=gao_prepare(ids);let range=gao_range(sphere);
    var best=vec2<u32>(0xffffffffu,GAO_NONE);
    let extent=vec3<u32>(range.high-range.low)+vec3<u32>(1u);let cells=extent.x*extent.y*extent.z;
    for(var t=lane;t<cells;t+=POST_LANES){
        let cell=range.low+vec3<i32>(vec3<u32>(t%extent.x,(t/extent.x)%extent.y,t/(extent.x*extent.y)));
        let bucket=gao_cell_id(cell);
        if(gao_cells[bucket]==gao_cells[bucket+1u]||gao_cell_outside(sphere,cell)){continue;}
        for(var k=gao_cells[bucket];k<gao_cells[bucket+1u];k++){let id=gao_slot(k);
            if(id==gao_seed||any(ids==vec3<u32>(id))){continue;}
            let fast=gao_sphere_test(sphere,id);
            if(select(gao_inside_prepared(ids,id),fast.x,fast.y)){
                let d=gao_point(id)-gao_origin;let key=vec2<u32>(bitcast<u32>(dot(d,d)),id);
                if(key.x<best.x||(key.x==best.x&&key.y<best.y)){best=key;}
            }
        }
    }
    live_best[lane]=best;workgroupBarrier();
    for(var stride=POST_LANES/2u;stride>0u;stride/=2u){
        if(lane<stride){let o=live_best[lane+stride];let m=live_best[lane];if(o.x<m.x||(o.x==m.x&&o.y<m.y)){live_best[lane]=o;}}workgroupBarrier();
    }
    return workgroupUniformLoad(&live_best[0]).y;
}
@compute @workgroup_size(POST_LANES)
fn gao_live_star(@builtin(workgroup_id) wid:vec3<u32>,@builtin(num_workgroups) grid:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let item=wid.x+wid.y*grid.x;
    if(lane==0u){live_uniform.x=live_pending(0u);}
    if(item>=workgroupUniformLoad(&live_uniform).x){return;}
    gao_seed=live_pending(1u+item);gao_lane=lane;gao_full=true;gao_origin=gao_point(gao_seed);
    // Radius: twice the old star's reach (a fresh seed's: two cells).
    let old=gao_states[gao_seed];
    if(lane==0u){gao_post_meta=vec4<u32>(old.y,old.z,0u,0u);}
    for(var i=lane;i<old.z;i+=POST_LANES){gao_shared_triangles[i]=gao_triangles[i*p.n+gao_seed];}
    for(var i=lane;i<old.y;i+=POST_LANES){gao_shared_indices[i]=gao_indices[i*p.n+gao_seed];}
    workgroupBarrier();
    // Between one and three grid cells (about 4 sites each): an old star with a
    // nearly flat face can reach far, and the rounds below grow the ball as needed.
    let cell_width=1.0/max(p.scale.x,1e-30);
    var radius=2.0*live_rho(lane,workgroupUniformLoad(&gao_post_meta).y);
    radius=select(2.0*cell_width,clamp(radius,cell_width,3.0*cell_width),radius>0.0&&radius<3e37);
    // The initial tetrahedron of the guards, as gao_dt_init.
    if(lane==0u){
        gao_nv=4u;gao_nt=0u;gao_nr=0u;gao_noutput=0u;gao_status=0u;
        for(var i=0u;i<4u;i++){gao_write_index(i,gao_encode(p.n+i));}
        gao_add_triangle(0u,2u,3u,p.n);gao_add_triangle(0u,1u,2u,p.n);
        gao_add_triangle(1u,0u,3u,p.n);gao_add_triangle(2u,1u,3u,p.n);
        gao_post_meta=vec4<u32>(gao_nv,gao_nt,0u,0u);
    }
    workgroupBarrier();
    var rho=bitcast<f32>(0x7f800000u^p.zero);
    var steps=0u;var failed=false;
    // Gather rounds: the sites with covered < d <= radius, inserted nearest first; then
    // the ball grows to twice the cell's reach until it holds every face's sphere. A
    // crowded shell shrinks (the face searches below certify what the ball misses).
    var covered=0.0;
    for(var round=0u;round<4u&&!failed;round++){
        var count=LIVE_CANDIDATES+1u;
        for(var attempt=0u;attempt<8u&&count>LIVE_CANDIDATES;attempt++){
            if(lane==0u){atomicStore(&live_count,0u);}
            workgroupBarrier();
            let lo=gao_cell(gao_origin-vec3<f32>(radius));let hi=gao_cell(gao_origin+vec3<f32>(radius));
            let extent=vec3<u32>(hi-lo)+vec3<u32>(1u);let cells=extent.x*extent.y*extent.z;
            for(var t=lane;t<cells;t+=POST_LANES){
                let cell=lo+vec3<i32>(vec3<u32>(t%extent.x,(t/extent.x)%extent.y,t/(extent.x*extent.y)));
                let bucket=gao_cell_id(cell);
                for(var k=gao_cells[bucket];k<gao_cells[bucket+1u];k++){let id=gao_slot(k);
                    if(id==gao_seed){continue;}
                    let d=gao_point(id)-gao_origin;let d2=dot(d,d);
                    if(d2<=radius*radius&&(covered==0.0||d2>covered*covered)){
                        let at=atomicAdd(&live_count,1u);
                        if(at<LIVE_CANDIDATES){live_ids[at]=id;live_distance[at]=d2;}
                    }
                }
            }
            workgroupBarrier();
            if(lane==0u){live_uniform.y=atomicLoad(&live_count);}
            count=workgroupUniformLoad(&live_uniform).y;
            if(count>LIVE_CANDIDATES){radius=covered+0.5*(radius-covered);}
        }
        if(count>LIVE_CANDIDATES){break;}
        // Nearest first: each candidate's rank among the others (ties by slot).
        for(var i=lane;i<count;i+=POST_LANES){
            var rank=0u;let di=live_distance[i];
            for(var j=0u;j<count;j++){let dj=live_distance[j];if(dj<di||(dj==di&&live_ids[j]<live_ids[i])){rank++;}}
            live_order[rank]=i;
        }
        workgroupBarrier();
        for(var k=0u;k<count;k++){
            let at=workgroupUniformLoad(&live_order[k]);
            // Past twice the cell's reach no site can cut it.
            if(4.0*rho*rho<workgroupUniformLoad(&live_distance[at])){break;}
            let result=live_insert(lane,live_ids[at]);
            if(result==2u){failed=true;break;}
            if(result==1u){steps++;rho=live_rho(lane,workgroupUniformLoad(&gao_post_meta).y);}
        }
        covered=radius;
        if(2.0*rho<=covered){break;}
        radius=select(2.0*covered,2.0*rho*1.001,rho<3e37);
    }
    radius=covered;
    // Faces whose sphere leaves the gathered ball: searched one by one.
    for(var search=0u;search<LIVE_SEARCHES&&!failed;search++){
        let f=live_open_face(lane,radius);
        if(f==GAO_NONE){break;}
        let found=live_search(lane,f);
        if(found==GAO_NONE){
            if(lane==0u){var t=gao_triangle(f);t.w|=GAO_PROCESSED;gao_write_triangle(f,t);}
            workgroupBarrier();
            continue;
        }
        let result=live_insert(lane,found);
        if(result!=1u){failed=true;}
        steps++;
    }
    if(lane==0u){
        var done=!failed&&live_open_face_serial(radius);
        for(var i=0u;i<gao_nt&&done;i++){if(any(gao_global_triangle(gao_triangle(i))>=vec3<u32>(p.n))){done=false;}}
        live_verdict=select(0u,1u,done);
    }
    if(workgroupUniformLoad(&live_verdict)==0u){
        // gao_main recomputes it (the state is still GAO_PENDING).
        if(lane==0u){atomicAdd(&gao_counters[select(23u,20u,failed)],1u);
}
        return;
    }
    // Write the star back as gao_finish_main leaves it.
    let state=workgroupUniformLoad(&gao_post_meta);
    if(lane==0u){
        for(var i=0u;i<gao_nv;i++){gao_write_index(i,gao_index(i)&~GAO_GREATER);}
        gao_states[gao_seed]=vec4<u32>(0u,gao_nv,gao_nt,steps);
        atomicAdd(&gao_counters[19],1u);
    }
    workgroupBarrier();
    for(var i=lane;i<state.y;i+=POST_LANES){gao_triangles[i*p.n+gao_seed]=gao_shared_triangles[i];}
    for(var i=lane;i<state.x;i+=POST_LANES){gao_indices[i*p.n+gao_seed]=gao_shared_indices[i];}
}
// Every face certified (lane 0 only, after the searches).
fn live_open_face_serial(radius:f32)->bool{
    for(var i=0u;i<gao_nt;i++){if((gao_triangle(i).w&GAO_PROCESSED)==0u&&!(2.0*live_face_reach(i)<=radius)){return false;}}
    return true;
}
