// Source-guided port of Delaunay1Ring topology in cudaDelaunay.cu.
// Upstream MIT: WuhengGao/Parallel-3D-Delaunay-GPU, commit
// 6e20ee49d3bf13914f1ce87b9194bdde6f2b3273. See licenses/MIT-Gao.txt.
// Caller supplies gao_triangle/write_triangle and gao_index/write_index.
// Triangles contain three local vertex indices plus the original bit tags.
const GAO_NONE=0xffffffffu;
const GAO_GREATER=0x80000000u;
const GAO_PROCESSED=1u;
const GAO_INEXACT=2u;
const GAO_OUTPUT=4u;
var<private> gao_nv:u32;
var<private> gao_nt:u32;
var<private> gao_nr:u32;
var<private> gao_noutput:u32;
var<private> gao_status:u32;
// Search every face (the caller sets it from GaoParams; see GAO_FULL).
var<private> gao_full:bool;
var<private> gao_boundary_next:array<u32,GAO_MAX_P>;
var<workgroup> gao_shared_boundary_next:array<u32,GAO_MAX_P>;
fn gao_boundary_get(i:u32)->u32{if(GAO_POST){return gao_shared_boundary_next[i];}return gao_boundary_next[i];}
fn gao_boundary_put(i:u32,value:u32){if(GAO_POST){gao_shared_boundary_next[i]=value;}else{gao_boundary_next[i]=value;}}
var<private> gao_used:array<u32,(GAO_MAX_P+31u)/32u>;
fn gao_mark_used(i:u32){gao_used[i/32u]|=1u<<(i%32u);}
fn gao_clear_used(i:u32){gao_used[i/32u]&=~(1u<<(i%32u));}
fn gao_is_used(i:u32)->bool{return (gao_used[i/32u]&(1u<<(i%32u)))!=0u;}
// Delaunay1Ring::add_vertex: compact unused local vertices only on overflow.
fn gao_add_vertex(encoded_id:u32)->u32{
    if(gao_nv>=GAO_MAX_P){
        for(var i=0u;i<(GAO_MAX_P+31u)/32u;i++){gao_used[i]=0u;}
        for(var i=0u;i<gao_nt+gao_nr;i++){let t=gao_triangle(i);gao_mark_used(t.x);gao_mark_used(t.y);gao_mark_used(t.z);}
        var i=0u;
        while(i<gao_nv){
            if(!gao_is_used(i)){
                gao_nv--;if(gao_nv==i){break;}gao_write_index(i,gao_index(gao_nv));
                if(gao_is_used(gao_nv)){
                    gao_mark_used(i);
                    for(var j=0u;j<gao_nt+gao_nr;j++){var t=gao_triangle(j);for(var k=0u;k<3u;k++){if(t[k]==gao_nv){t[k]=i;}}gao_write_triangle(j,t);}
                }else{gao_clear_used(i);}
            }else{i++;}
        }
        if(gao_nv>=GAO_MAX_P){gao_status=1u;return GAO_NONE;}
    }
    let slot=gao_nv;gao_write_index(slot,encoded_id);gao_nv++;return slot;
}
// Delaunay1Ring::remove_triangle: removed triangles occupy [nt,nt+nr).
fn gao_remove_triangle(i:u32){gao_nt--;let last=gao_triangle(gao_nt);gao_write_triangle(gao_nt,gao_triangle(i));gao_write_triangle(i,last);gao_nr++;}
fn gao_add_triangle(i:u32,j:u32,k:u32,npoints:u32){
    if(gao_nt>=GAO_MAX_T){gao_status=1u;return;}
    let a=gao_index(i);let b=gao_index(j);let c=gao_index(k);
    let order=u32((a&GAO_GREATER)!=0u)+u32((b&GAO_GREATER)!=0u)+u32((c&GAO_GREATER)!=0u);var tag=0u;
    if((order==0u||order==3u)&&!gao_full){tag|=GAO_PROCESSED;}
    else if(order==1u&&all((vec3<u32>(a,b,c)&vec3<u32>(~GAO_GREATER))<vec3<u32>(npoints))){tag|=GAO_OUTPUT;}
    gao_write_triangle(gao_nt,vec4<u32>(i,j,k,tag));gao_nt++;
}
// Delaunay1Ring::compute_boundary: merge removed triangles into one directed
// boundary, using the local vertex-indexed successor array from the CUDA code.
fn gao_compute_boundary()->u32{
    for(var i=0u;i<gao_nv;i++){gao_boundary_put(i,GAO_NONE);}
    var first=GAO_NONE;var at=gao_nt;var allowance=GAO_MAX_T*GAO_MAX_T;
    while(gao_nr>0u){
        if(allowance==0u){gao_status=2u;return GAO_NONE;}allowance--;
        let t=gao_triangle(at);var border:vec3<bool>;var opposite:vec3<bool>;
        for(var i=0u;i<3u;i++){border[i]=gao_boundary_get(t[i])!=GAO_NONE;opposite[i]=gao_boundary_get(t[(i+1u)%3u])==t[i];}
        var simple=true;
        for(var i=0u;i<3u;i++){if(!opposite[i]&&!opposite[(i+1u)%3u]&&border[(i+1u)%3u]){simple=false;}}
        if(!any(opposite)){
            if(first==GAO_NONE){for(var i=0u;i<3u;i++){gao_boundary_put(t[i],t[(i+1u)%3u]);}first=t.x;}
            else{simple=false;}
        }
        if(!simple){at++;if(at==gao_nt+gao_nr){gao_status=2u;return GAO_NONE;}continue;}
        for(var i=0u;i<3u;i++){if(!opposite[i]){gao_boundary_put(t[i],t[(i+1u)%3u]);}}
        for(var i=0u;i<3u;i++){
            let v=t[(i+1u)%3u];if(opposite[i]&&opposite[(i+1u)%3u]){if(first==v){first=gao_boundary_get(v);}gao_boundary_put(v,GAO_NONE);}
        }
        gao_write_triangle(at,gao_triangle(gao_nt+gao_nr-1u));at=gao_nt;gao_nr--;
    }
    return first;
}
// Delaunay1Ring::halfspace_intersection. The fast predicate returns
// (inside, trusted); an uncertain plane update is rolled back as in CUDA.
fn gao_halfspace_intersection(encoded_id:u32,npoints:u32)->bool{
    let vertex=gao_add_vertex(encoded_id);if(gao_status!=0u){gao_nt+=gao_nr;gao_nr=0u;return false;}
    var trusted=true;var i=0u;
    while(i<gao_nt){
        let verdict=gao_conflict(encoded_id,gao_triangle(i));trusted=trusted&&verdict.y;
        if(verdict.x){gao_remove_triangle(i);}else{i++;}
    }
    if(gao_nr==0u||!trusted){gao_nt+=gao_nr;gao_nr=0u;gao_nv--;return false;}
    let first=gao_compute_boundary();if(first==GAO_NONE){gao_nv--;return false;}
    var current=first;var allowance=GAO_MAX_P;
    loop {
        if(allowance==0u||current>=gao_nv){gao_status=2u;return false;}allowance--;
        let next=gao_boundary_get(current);if(next==GAO_NONE){gao_status=2u;return false;}
        gao_add_triangle(vertex,current,next,npoints);if(gao_status!=0u){return false;}
        current=next;if(current==first){break;}
    }
    return true;
}
// MainPhase::update_activetr, retaining the certified output prefix.
fn gao_update_active()->u32{
    for(var i=gao_noutput;i<gao_nt;i++){var t=gao_triangle(i);if((t.w&GAO_PROCESSED)==0u){t.w|=GAO_PROCESSED;gao_write_triangle(i,t);return i;}}
    return GAO_NONE;
}
// MainPhase::search's no-candidate branch.
fn gao_finish_search(active_id:u32,trusted:bool){
    var t=gao_triangle(active_id);if(!trusted){t.w|=GAO_INEXACT;gao_write_triangle(active_id,t);}
    if((t.w&GAO_OUTPUT)!=0u&&(t.w&GAO_INEXACT)==0u){let previous=gao_triangle(gao_noutput);gao_write_triangle(gao_noutput,t);gao_write_triangle(active_id,previous);gao_noutput++;}
}
// MainPhase::update_stat and the successful DT_kernel output contract.
fn gao_finish_main(cycles:u32){
    if(gao_status!=0u){return;}
    if(cycles==40u){gao_status=3u;return;}
    for(var i=0u;i<gao_nt;i++){if((gao_triangle(i).w&GAO_INEXACT)!=0u){gao_status=4u;return;}}
    // The whole star stays: its triangles give the exact neighbours (live updates); the
    // tetrahedra this seed outputs are the GAO_OUTPUT triangles, listed first.
    for(var i=0u;i<gao_nv;i++){gao_write_index(i,gao_index(i)&~GAO_GREATER);}
}
