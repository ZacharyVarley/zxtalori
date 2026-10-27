// Live covering moves (covering.Live.beginMove): which stars a step's changes can alter.
//
// A star changes exactly when one of its tetrahedra is destroyed or a new tetrahedron
// joins it. A destroyed one has a changed vertex (its old position left) or its sphere
// holds a new position; a new one has a changed vertex, or its sphere held an old
// position (then the star had that site as a vertex), or it only exists once another
// changed site left (then the star had that site as a vertex). So each seed is marked
// when a changed site was a vertex of its star or a new position lies in one of its
// spheres (the conservative float test also marks undecided cases). Both are within the
// seed's reach, twice its largest face circumradius, of the position: the grid is
// scanned within the largest reach (R*), and seeds whose reach is unbounded or beyond
// the cap are kept in a list that every change tests.
//
// flags: per slot the mark bits (2 marked, 4 wide); then per slot its reach (f32 bits);
// then R*, the wide count, and the wide list.
@group(0) @binding(6) var<storage,read_write> mark_flags:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read_write> mark_pending:array<atomic<u32>>; // [0]: count, then slots
@group(0) @binding(8) var<storage,read> mark_changes:array<u32>;              // live.wgsl's scratch
const MARK_WIDE=1024u;
fn mark_star(v:u32){
    if(coordinates[v].w>=p.seeds){return;}
    if((atomicOr(&mark_flags[v],2u)&2u)==0u){gao_states[v].x=GAO_PENDING;atomicStore(&mark_pending[1u+atomicAdd(&mark_pending[0],1u)],v);}
}
fn mark_reach(v:u32)->f32{return bitcast<f32>(atomicLoad(&mark_flags[p.n+v]));}
// Whether the star of seed w has `slot` among its vertices (its vertex list may also
// hold vertices it lost while forming: a superset).
fn mark_has(w:u32,slot:u32)->bool{
    let nv=gao_states[w].y;
    for(var i=0u;i<nv;i++){if((gao_indices[i*p.n+w]&~GAO_GREATER)==slot){return true;}}
    return false;
}
// Whether the site in `slot` may lie in a sphere of the star of seed w.
fn mark_holds(w:u32,slot:u32)->bool{
    gao_seed=w;gao_origin=gao_point(w);
    let nt=gao_states[w].z;
    for(var i=0u;i<nt;i++){
        let ids=gao_global_triangle(gao_triangle(i));if(any(ids==vec3<u32>(slot))){continue;}
        let verdict=gao_sphere_test(gao_prepare(ids),slot);
        if(verdict.x||!verdict.y){return true;}
    }
    return false;
}
fn mark_test(w:u32,slot:u32,centre:vec3<f32>,holds:bool,near:bool){
    if(w==slot||coordinates[w].w>=p.seeds||gao_states[w].x==GAO_PENDING){return;}
    if(near){let d=gao_point(w)-centre;let r=mark_reach(w);if(!(dot(d,d)<=r*r)){return;}}
    if(select(mark_has(w,slot),mark_holds(w,slot),holds)){mark_star(w);}
}
// The seeds a changed position can concern: the grid within R* of it (lanes split the
// cells), then the wide list.
fn mark_scan(lane:u32,centre:vec3<f32>,slot:u32,holds:bool){
    let reach=bitcast<f32>(atomicLoad(&mark_flags[2u*p.n]));
    let lo=gao_cell(centre-vec3<f32>(reach));let hi=gao_cell(centre+vec3<f32>(reach));
    let extent=vec3<u32>(hi-lo)+vec3<u32>(1u);let cells=extent.x*extent.y*extent.z;
    for(var t=lane;t<cells;t+=64u){
        let cell=lo+vec3<i32>(vec3<u32>(t%extent.x,(t/extent.x)%extent.y,t/(extent.x*extent.y)));
        let bucket=gao_cell_id(cell);
        for(var at=gao_cells[bucket];at<gao_cells[bucket+1u];at++){mark_test(gao_slot(at),slot,centre,holds,true);}
    }
    let wide=min(atomicLoad(&mark_flags[2u*p.n+1u]),MARK_WIDE);
    for(var at=lane;at<wide;at+=64u){
        let w=atomicLoad(&mark_flags[2u*p.n+2u+at]);
        if((atomicLoad(&mark_flags[w])&4u)!=0u){mark_test(w,slot,centre,holds,false);}
    }
}
// One workgroup per change (flat over x and y).
@compute @workgroup_size(64)
fn gao_mark(@builtin(workgroup_id) wid:vec3<u32>,@builtin(num_workgroups) grid:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let change=wid.x+wid.y*grid.x;if(change>=min(mark_changes[2],p.n)){return;}
    let b=4u+2u*p.n+8u*change;
    let slot=mark_changes[b];let flags=mark_changes[b+1u];
    // The changed site's own star: recomputed if it is a seed, gone otherwise.
    if(lane==0u){if((flags&4u)!=0u){mark_star(slot);}else{gao_states[slot]=vec4<u32>(0u);}}
    if((flags&8u)==0u){return;}
    if((flags&1u)!=0u){mark_scan(lane,bitcast<vec3<f32>>(vec3<u32>(mark_changes[b+2u],mark_changes[b+3u],mark_changes[b+4u])),slot,false);}
    if((flags&2u)!=0u){mark_scan(lane,gao_point(slot),slot,true);}
}
// Reach of each recomputed star, whose mark it also clears (one workgroup per pending
// star, or with scale.w > 1.5 one per slot): twice its largest face circumcentre distance, rounded up;
// beyond the cap (lower.w) or unbounded, the star is listed as wide.
@compute @workgroup_size(1)
fn gao_reach(@builtin(workgroup_id) wid:vec3<u32>,@builtin(num_workgroups) grid:vec3<u32>){
    let item=wid.x+wid.y*grid.x;
    var v=item;
    if(p.scale.w<=1.5){if(item>=atomicLoad(&mark_pending[0])){return;}v=atomicLoad(&mark_pending[1u+item]);}
    if(v>=p.n){return;}
    // The next move may mark it again.
    atomicAnd(&mark_flags[v],~2u);
    if(coordinates[v].w>=p.seeds||gao_states[v].x!=0u){atomicStore(&mark_flags[p.n+v],0u);return;}
    gao_seed=v;gao_origin=gao_point(v);
    var reach=0.0;
    for(var i=0u;i<gao_states[v].z;i++){
        let s=gao_prepare(gao_global_triangle(gao_triangle(i)));
        let co=vec4<f32>(s.coefficient.z,-s.coefficient.y,s.coefficient.x,s.coefficient.w);
        let error=4e-6*max(max(s.bound.x,s.bound.y),max(s.bound.z,s.bound.w));
        if(!(abs(co.w)>2.0*error)){reach=bitcast<f32>(0x7f800000u^p.zero);break;}
        reach=max(reach,2.0*length(co.xyz/(2.0*co.w))*(1.0+1e-3)+1e-30);
    }
    atomicStore(&mark_flags[p.n+v],bitcast<u32>(reach));
    if(reach<=p.lower.w){atomicMax(&mark_flags[2u*p.n],bitcast<u32>(reach));atomicAnd(&mark_flags[v],~4u);return;}
    if((atomicOr(&mark_flags[v],4u)&4u)==0u){
        let at=atomicAdd(&mark_flags[2u*p.n+1u],1u);
        if(at<MARK_WIDE){atomicStore(&mark_flags[2u*p.n+2u+at],v);}
    }
}
