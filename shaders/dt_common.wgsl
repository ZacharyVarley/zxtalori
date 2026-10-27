// Independent one-rings following Gao & Chen's halfspace construction.
// MIT reference: WuhengGao/Parallel-3D-Delaunay-GPU (licenses/MIT-Gao.txt, docs/gpu-delaunay.md).
const NONE:u32=0xffffffffu;
struct Params { n:u32, seeds:u32, first:u32, cap:u32, dim:u32, zero:u32, budget:u32, pad:u32, lower:vec4<f32>, scale:vec4<f32> }
// phase,face count,initial-neighbour cursor,count; active face,clip cursor,new cursor,removed;
// query,initial flag,cold slot,cache cursor; cell box/cursor; ready,updates,status,cold count.
struct Star { info:vec4<u32>, cursor:vec4<u32>, query:vec4<u32>, low:vec4<u32>, high:vec4<u32>, cell:vec4<u32>, more:vec4<u32> }
struct Cache { ids:vec4<u32>, more:vec4<u32> } // fifth ID,kind,code(1-/2zero/3+/4pending),valid
struct QueryBox{lo:vec3<u32>,hi:vec3<u32>}
var<private> current_star:Star;
var<private> cached_seed:u32;
var<private> local_star_active:bool;
var<private> cooperative_query:bool;
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> coordinates:array<vec4<u32>>;
@group(0) @binding(2) var<storage,read_write> heads:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> next_site:array<u32>;
@group(0) @binding(4) var<storage,read_write> faces:array<vec4<u32>>;
@group(0) @binding(5) var<storage,read_write> stars:array<Star>;
@group(0) @binding(6) var<storage,read_write> cache:array<Cache>;
@group(0) @binding(7) var<storage,read_write> scratch:array<u32>;
@group(0) @binding(8) var<storage,read> prime_table:array<u32>;
@group(0) @binding(9) var<storage,read_write> counters:array<atomic<u32>>;
@group(0) @binding(10) var<storage,read_write> tetrahedra:array<vec4<u32>>;
fn predicate(seed:u32,v:vec4<u32>,e:u32,kind:u32)->i32 {
    var fast=orient_filter(v);if(kind==1u){fast=sphere_filter(v,e);}
    if(fast!=2){return fast;}
    for(var i=0u;i<8u;i++) {let c=cache[seed*8u+i];if(c.more.w!=0u&&all(c.ids==v)&&c.more.x==e&&c.more.y==kind){return select(i32(c.more.z)-2,2,c.more.z==4u);}}
    let wide_sign=double_filter(v,e,kind);if(wide_sign!=2){return wide_sign;}
    return queue_predicate(seed,v,e,kind);
}
fn queue_predicate(seed:u32,v:vec4<u32>,e:u32,kind:u32)->i32 {
    var slot=0u;
    if(local_star_active){slot=current_star.query.w%8u;current_star.query.w++;current_star.query.z=slot;current_star.more.w++;}
    else{slot=stars[seed].query.w%8u;stars[seed].query.w++;stars[seed].query.z=slot;stars[seed].more.w++;}
    cache[seed*8u+slot]=Cache(v,vec4<u32>(e,kind,4u,1u));
    return 2;
}
fn predicate_fast(seed:u32,v:vec4<u32>,e:u32,kind:u32)->i32{
    var sign=2i;if(kind==0u){sign=orient_quick(v);}else{sign=sphere_quick(v,e);}if(sign!=2){return sign;}
    for(var i=0u;i<8u;i++){let c=cache[seed*8u+i];if(c.more.w!=0u&&all(c.ids==v)&&c.more.x==e&&c.more.y==kind){return select(i32(c.more.z)-2,2,c.more.z==4u);}}
    return queue_predicate(seed,v,e,kind);
}
fn inside_fast(seed:u32,v:vec4<u32>,e:u32)->i32{
    if(any(v==vec4<u32>(e))){return 0;}let raw=predicate_fast(seed,v,e,1u);if(raw!=0){return raw;}
    var ids=array<u32,5>(v.x,v.y,v.z,v.w,e);
    for(var i=0u;i<5u;i++){for(var j=i+1u;j<5u;j++){if(all(signed_bits(coordinates[ids[i]].xyz)==signed_bits(coordinates[ids[j]].xyz))){return 0;}}}
    for(var rank=0u;rank<5u;rank++){
        var row=0u;for(var j=1u;j<5u;j++){if(ids[j]<ids[row]){row=j;}}
        var other:vec4<u32>;var at=0u;for(var j=0u;j<5u;j++){if(j!=row){other[at]=select(v[j%4u],e,j==4u);at++;}}
        let sign=predicate_fast(seed,other,NONE,0u);if(sign==2){return 2;}if(sign!=0){return select(sign,-sign,(row&1u)!=0u);}ids[row]=NONE;
    }return 0;
}
fn inside(seed:u32,v:vec4<u32>,e:u32)->i32 {
    if(any(v==vec4<u32>(e))){return 0;}
    let raw=predicate(seed,v,e,1u);if(raw==2||raw!=0){return raw;}
    var ids=array<u32,5>(v.x,v.y,v.z,v.w,e);
    for(var i=0u;i<5u;i++){for(var j=i+1u;j<5u;j++){if(all(signed_bits(coordinates[ids[i]].xyz)==signed_bits(coordinates[ids[j]].xyz))){return 0;}}}
    // Lift-only SoS, in global stable-ID order; no coordinate displacement.
    for(var rank=0u;rank<5u;rank++){
        var row=0u;for(var j=1u;j<5u;j++){if(ids[j]<ids[row]){row=j;}}
        var other:vec4<u32>;var at=0u;for(var j=0u;j<5u;j++){if(j!=row){other[at]=select(v[j%4u],e,j==4u);at++;}}
        let sign=predicate(seed,other,NONE,0u);if(sign==2){return 2;}if(sign!=0){return select(sign,-sign,(row&1u)!=0u);}ids[row]=NONE;
    }
    return 0;
}


