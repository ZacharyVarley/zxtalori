// Source-guided Gao main-pass layout. See licenses/MIT-Gao.txt.
// Physically bucket-sorted coordinates, cell offsets, component-major local
// triangles and vertex IDs. A failed main pass preserves its entire local mesh.
struct GaoParams {
    // reserved: bit 2 collects exact-predicate statistics; bit GAO_FULL searches every
    // face, so each star is its exact Delaunay star (live updates read neighbours from
    // stars). Without it, as in the source, faces of tetrahedra that another seed
    // outputs are never searched and may be stale.
    n:u32, dim:u32, zero:u32, reserved:u32,
    // Sites with original id >= seeds are not seeds: no star of their own.
    // Live updates (gao_delaunay.Live): scale.w > 0.5 recomputes only the stars marked
    // GAO_PENDING. With GAO_LISTED the slots are not sorted by cell: gao_cells holds the
    // cell offsets, then the slots of each cell in that order (gao_slot).
    lower:vec4<f32>, upper:vec3<f32>, seeds:u32, scale:vec4<f32>
}
const GAO_PENDING=15u;
const GAO_FULL=4u;
const GAO_LISTED=8u;
@group(0) @binding(0) var<uniform> p:GaoParams;
@group(0) @binding(1) var<storage,read> coordinates:array<vec4<u32>>;
@group(0) @binding(2) var<storage,read> gao_cells:array<u32>;
@group(0) @binding(3) var<storage,read_write> gao_triangles:array<u32>;
@group(0) @binding(4) var<storage,read_write> gao_indices:array<u32>;
@group(0) @binding(5) var<storage,read_write> gao_states:array<vec4<u32>>;
var<private> gao_seed:u32;
var<private> gao_lane:u32;
var<private> gao_origin:vec3<f32>;
var<workgroup> gao_shared_triangles:array<u32,GAO_MAX_T>;
var<workgroup> gao_shared_indices:array<u32,GAO_MAX_P>;
// The slot at position k of the grid's cell order.
fn gao_slot(k:u32)->u32{if((p.reserved&GAO_LISTED)!=0u){return gao_cells[p.dim*p.dim*p.dim+1u+k];}return k;}
fn gao_point(i:u32)->vec3<f32>{return bitcast<vec4<f32>>(coordinates[i]).xyz;}
fn gao_triangle(i:u32)->vec4<u32>{var t:u32;if(GAO_POST){t=gao_shared_triangles[i];}else{t=gao_triangles[i*p.n+gao_seed];}return vec4<u32>(t&255u,(t>>8u)&255u,(t>>16u)&255u,t>>24u);}
fn gao_write_triangle(i:u32,t:vec4<u32>){let packed=t.x|(t.y<<8u)|(t.z<<16u)|(t.w<<24u);if(GAO_POST){gao_shared_triangles[i]=packed;}else{gao_triangles[i*p.n+gao_seed]=packed;}}
fn gao_index(i:u32)->u32{if(GAO_POST){return gao_shared_indices[i];}return gao_indices[i*p.n+gao_seed];}
fn gao_write_index(i:u32,value:u32){if(GAO_POST){gao_shared_indices[i]=value;}else{gao_indices[i*p.n+gao_seed]=value;}}
fn gao_encode(i:u32)->u32{
    let q=gao_point(i);let o=gao_origin;
    let greater=q.x>o.x||(q.x==o.x&&(q.y>o.y||(q.y==o.y&&(q.z>o.z||(q.z==o.z&&coordinates[i].w>coordinates[gao_seed].w)))));
    return i|select(0u,GAO_GREATER,greater);
}
fn gao_global_triangle(t:vec4<u32>)->vec3<u32>{return vec3<u32>(gao_index(t.x),gao_index(t.y),gao_index(t.z))&vec3<u32>(~GAO_GREATER);}
fn gao_cell(q:vec3<f32>)->vec3<i32>{return vec3<i32>(clamp(floor((q-p.lower.xyz)*p.scale.xyz),vec3<f32>(0.0),vec3<f32>(f32(p.dim-1u))));}
fn gao_cell_id(c:vec3<i32>)->u32{return u32(c.x)+p.dim*(u32(c.y)+p.dim*u32(c.z));}
