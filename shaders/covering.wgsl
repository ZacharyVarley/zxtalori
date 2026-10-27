// Delaunay circumcaps plus a global shell-completeness certificate.
@group(0) @binding(11) var<storage,read> zone_halfspaces:array<vec4<f32>>;
@group(0) @binding(12) var<storage,read> group_ops:array<vec4<f32>>;
@group(0) @binding(13) var<storage,read_write> cap_data:array<vec4<f32>>;
@group(0) @binding(14) var<storage,read_write> cover:array<atomic<u32>>;
@group(0) @binding(15) var<storage,read> original_sites:array<vec4<f32>>;
fn qmul(a:vec4<f32>,b:vec4<f32>)->vec4<f32>{return vec4<f32>(a.x*b.x-dot(a.yzw,b.yzw),a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));}
fn stereo_quaternion(index:u32)->vec4<f32>{let x=point(index);let r=dot(x,x);return vec4<f32>(1.0-r,2.0*x)/(1.0+r);}
fn ds_divide(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{let q=rounded(a.x/b.x);let error=ds_sub(a,ds_mul(b,vec2<f32>(q,0.0)));return ds_add(vec2<f32>(q,0.0),vec2<f32>(rounded(rounded(error.x+error.y)/b.x),0.0));}
fn quaternion_ds(index:u32)->array<vec2<f32>,4>{
    let x=point(index);var radius=vec2<f32>(0.0);for(var k=0u;k<3u;k++){radius=ds_add(radius,ds_mul(vec2<f32>(x[k],0.0),vec2<f32>(x[k],0.0)));}
    let denominator=ds_add(vec2<f32>(1.0,0.0),radius);var q:array<vec2<f32>,4>;q[0]=ds_divide(ds_sub(vec2<f32>(1.0,0.0),radius),denominator);
    for(var k=0u;k<3u;k++){q[k+1u]=ds_divide(vec2<f32>(2.0*x[k],0.0),denominator);}return q;
}
fn in_zone(q:vec4<f32>)->bool{if(q.x<0.0){return false;}for(var k=0u;k<p.first;k++){if(abs(dot(q,normalize(group_ops[k])))>q.x+2e-6){return false;}}return true;}
@compute @workgroup_size(128)
fn circumcaps(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if(i>=p.seeds){return;}cap_data[i*2u]=vec4<f32>(0.0);cap_data[i*2u+1u]=vec4<f32>(0.0);
    let t=tetrahedra[i];var q:array<array<vec2<f32>,4>,4>;for(var j=0u;j<4u;j++){q[j]=quaternion_ds(t[j]);}
    var edges:array<array<vec2<f32>,4>,3>;for(var j=0u;j<3u;j++){for(var k=0u;k<4u;k++){edges[j][k]=ds_sub(q[j+1u][k],q[0][k]);}}
    var centre:vec4<f32>;
    for(var k=0u;k<4u;k++){var rows:array<array<vec2<f32>,3>,3>;for(var j=0u;j<3u;j++){var at=0u;for(var l=0u;l<4u;l++){if(l!=k){rows[j][at]=edges[j][l];at++;}}}let minor=ds_det(rows[0],rows[1],rows[2]);centre[k]=rounded(minor.x+minor.y)*select(1.0,-1.0,(k&1u)!=0u);}
    let length=length(centre);if(!(length>1e-24)){atomicAdd(&cover[15],1u);return;}centre=centre/length;
    let first=vec4<f32>(q[0][0].x,q[0][1].x,q[0][2].x,q[0][3].x);var height=dot(centre,first);
    if(centre.x+height<0.0){centre=-centre;height=-height;}
    if(height<=0.0||!in_zone(centre)){return;}
    var smallest=4.0;var largest=0.0;
    for(var j=0u;j<4u;j++){var distance=vec2<f32>(0.0);for(var k=0u;k<4u;k++){let d=ds_sub(q[j][k],vec2<f32>(centre[k],0.0));distance=ds_add(distance,ds_mul(d,d));}let value=max(0.0,rounded(distance.x+distance.y));smallest=min(smallest,value);largest=max(largest,value);}
    let radius=2.0*asin(min(1.0,sqrt(largest)*0.5));
    if(radius>p.scale.y+2e-5){atomicAdd(&cover[15],1u);return;}
    if(largest-smallest>1e-5){atomicAdd(&cover[15],1u);return;}
    cap_data[i*2u]=centre;cap_data[i*2u+1u]=vec4<f32>(radius,smallest,largest,0.0);atomicMax(&cover[8],bitcast<u32>(radius));atomicAdd(&cover[10],1u);
}
@compute @workgroup_size(128)
fn select_cap(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<p.seeds&&bitcast<u32>(cap_data[i*2u+1u].x)==atomicLoad(&cover[8])&&cap_data[i*2u].x>=0.0){atomicMin(&cover[9],i);}}
// Four positive-major-component cube charts cover RP3. The normalization map
// (1,x,y,z)/||(1,x,y,z)|| has differential norm <=1. A grid of m subdivisions
// therefore covers every orientation within delta=sqrt(3)/m on S3.
// If each folded chart vertex has a shell site within T-delta, the triangle
// inequality proves that the full quotient is covered within T. No guess about
// a small observed Delaunay radius is used as a shell-completeness certificate.
fn certificate_probe_near_zone(q:vec4<f32>)->bool{
    // A nearest chart vertex lies within delta of its domain point. If a
    // vertex is farther than delta from any FZ halfspace, it cannot be that
    // nearest vertex. Check both antipodes because the charts cover RP3.
    let tolerance=sin(p.scale.z)+4e-6;
    var positive=true;var negative=true;
    for(var k=0u;k<arrayLength(&zone_halfspaces);k++){
        let side=dot(q,zone_halfspaces[k]);
        positive=positive&&side>=-tolerance;negative=negative&&side<=tolerance;
        if(!positive&&!negative){return false;}
    }
    return true;
}
fn certificate_chart_probe(index:u32)->vec4<f32>{
    let side=p.cap+1u;let size=side*side*side;
    let chart=index/size;let local=index%size;let xyz=vec3<u32>(local%side,(local/side)%side,local/(side*side));
    var q:vec4<f32>;q[chart]=1.0;var at=0u;for(var k=0u;k<4u;k++){if(k!=chart){q[k]=2.0*f32(xyz[at])/f32(p.cap)-1.0;at++;}}
    return normalize(q);
}
@compute @workgroup_size(128)
fn certify_shell(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>){
    let index=id.x+id.y*groups.x*128u;let m=p.cap;let side=m+1u;let size=side*side*side;if(index>=4u*size){return;}
    var q=certificate_chart_probe(index);if(!certificate_probe_near_zone(q)){return;}
    var best=q;var scalar=abs(q.x);for(var k=0u;k<p.first;k++){var candidate=qmul(q,normalize(group_ops[k]));if(candidate.x<0.0){candidate=-candidate;}if(candidate.x>scalar){best=candidate;scalar=candidate.x;}}q=normalize(best);if(q.x<0.0){q=-q;}
    let radius=p.scale.y-p.scale.z-2e-5;if(radius<=0.0){atomicAdd(&cover[11],1u);return;}
    let denominator=q.x+cos(radius);let centre=q.yzw/denominator;let r=sin(radius)/denominator+4e-6;
    let lo=vec3<i32>(clamp(floor((centre-vec3<f32>(r)-p.lower.xyz)*p.scale.x)-vec3<f32>(1),vec3<f32>(0),vec3<f32>(f32(p.dim-1u))));
    let hi=vec3<i32>(clamp(floor((centre+vec3<f32>(r)-p.lower.xyz)*p.scale.x)+vec3<f32>(1),vec3<f32>(0),vec3<f32>(f32(p.dim-1u))));
    let chord=2.0*sin(radius*0.5)-4e-6;let threshold=chord*chord;var found=false;
    for(var z=lo.z;z<=hi.z&&!found;z++){for(var y=lo.y;y<=hi.y&&!found;y++){for(var x=lo.x;x<=hi.x&&!found;x++){
        let cell=u32(x)+p.dim*(u32(y)+p.dim*u32(z));var site=atomicLoad(&heads[cell]);
        while(site!=NONE){let d=q-stereo_quaternion(site);if(dot(d,d)<=threshold){found=true;break;}site=next_site[site];}
    }}}
    if(!found){atomicAdd(&cover[11],1u);}atomicAdd(&cover[12],1u);
}
@compute @workgroup_size(128)
fn check_witness(@builtin(global_invocation_id) id:vec3<u32>){
    let at=id.x;let n=p.budget;if(at>=n*p.first){return;}let cap=atomicLoad(&cover[9]);if(cap==NONE){return;}
    let q=normalize(qmul(normalize(original_sites[at%n]),normalize(group_ops[at/n])));let centre=cap_data[cap*2u];let d1=q-centre;let d2=q+centre;
    // The selected cap's centre rides back with the other results in words 16..19.
    if(at==0u){for(var k=0u;k<4u;k++){atomicStore(&cover[16u+k],bitcast<u32>(centre[k]));}}
    atomicMin(&cover[14],bitcast<u32>(min(dot(d1,d1),dot(d2,d2))));
}
@group(0) @binding(16) var<storage,read> site_origin:array<u32>;
@group(0) @binding(19) var<storage,read_write> local_cover:array<atomic<u32>>; // per source: largest incident cap (radius bits), then incident tetrahedra
// Per-source results for display: each counted cap (centre in the zone: one per orbit of
// caps, its vertices images of the same sources) raises its four sources' largest cap,
// and each tetrahedron counts once at each source whose identity image it holds. A
// vertex's link is a triangulated sphere, so its Delaunay neighbours, the faces of its
// Voronoi cell, number 2 + tetrahedra / 2.
@compute @workgroup_size(128)
fn local_caps(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if(i>=p.seeds){return;}
    let t=tetrahedra[i];let rho=cap_data[i*2u+1u].x;
    for(var j=0u;j<4u;j++){
        let o=site_origin[t[j]];
        if(o<p.budget){atomicAdd(&local_cover[p.budget+o],1u);}
        if(rho>0.0){atomicMax(&local_cover[o%p.budget],bitcast<u32>(rho));}
    }
}
@group(0) @binding(17) var<storage,read_write> cap_rows:array<vec4<f32>>;
@group(0) @binding(18) var<storage,read_write> row_count:array<atomic<u32>>;
// Linearized radius of each near-maximal cap under left moves exp(d) q of its sources.
// With c = sum_j lambda_j v_j and c.v_j = cos(rho): drho = -cot(rho) sum_j lambda_j c.(d v_j).
// Left moves commute with the right action, so every image follows its source's d.
@compute @workgroup_size(128)
fn cap_rows_kernel(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if(i>=p.seeds){return;}
    let rho=cap_data[i*2u+1u].x;let top=bitcast<f32>(atomicLoad(&cover[8]));
    if(!(rho>0.0)||rho<p.scale.w*top){return;}
    let c=cap_data[i*2u];let t=tetrahedra[i];var v:array<vec4<f32>,4>;var src:vec4<u32>;
    for(var j=0u;j<4u;j++){let o=site_origin[t[j]];let s=o%p.budget;let image=o/p.budget;var q=qmul(normalize(original_sites[s]),normalize(group_ops[image/2u]));if((image&1u)!=0u){q=-q;}v[j]=q;src[j]=s;}
    let m=mat4x4<f32>(v[0],v[1],v[2],v[3]);let d=determinant(m);if(!(abs(d)>1e-30)){return;}
    let k=atomicAdd(&row_count[0],1u);let cot=cos(rho)/sin(rho);
    cap_rows[k*5u]=bitcast<vec4<f32>>(src);
    for(var j=0u;j<4u;j++){var mj=m;mj[j]=c;let lambda=determinant(mj)/d;let w=v[j];
        let a=vec3<f32>(dot(c,qmul(vec4<f32>(0.0,1.0,0.0,0.0),w)),dot(c,qmul(vec4<f32>(0.0,0.0,1.0,0.0),w)),dot(c,qmul(vec4<f32>(0.0,0.0,0.0,1.0),w)));
        cap_rows[k*5u+1u+j]=vec4<f32>(-cot*lambda*a,rho);}
}
