// predict.h::insphereprepare and inspherefast, keeping CUDA coefficient order.
// A wider binary32 filter accommodates WebGPU arithmetic implementations.
struct GaoSphere{coefficient:vec4<f32>,bound:vec4<f32>}
fn gao_round(x:f32)->f32{return bitcast<f32>(bitcast<u32>(x)^p.zero);}
fn gao_prepare(ids:vec3<u32>)->GaoSphere{
    let b=gao_point(ids.x)-gao_origin;let c=gao_point(ids.y)-gao_origin;let d=gao_point(ids.z)-gao_origin;
    let bl=dot(b,b);let cl=dot(c,c);let dl=dot(d,d);
    let xy=gao_round(c.x*d.y-d.x*c.y);let xz=gao_round(c.x*d.z-d.x*c.z);let yz=gao_round(c.y*d.z-d.y*c.z);
    let xl=gao_round(c.x*dl-d.x*cl);let yl=gao_round(c.y*dl-d.y*cl);let zl=gao_round(c.z*dl-d.z*cl);
    let coefficients=vec4<f32>(gao_round(gao_round(b.x*yl-xl*b.y)+bl*xy),gao_round(gao_round(b.x*zl-xl*b.z)+bl*xz),gao_round(gao_round(b.y*zl-yl*b.z)+bl*yz),gao_round(gao_round(b.x*yz-b.y*xz)+b.z*xy));
    let xyplus=abs(c.x*d.y)+abs(d.x*c.y);let xzplus=abs(c.x*d.z)+abs(d.x*c.z);let yzplus=abs(c.y*d.z)+abs(d.y*c.z);
    let xlplus=abs(c.x*dl)+abs(d.x*cl);let ylplus=abs(c.y*dl)+abs(d.y*cl);let zlplus=abs(c.z*dl)+abs(d.z*cl);
    let bounds=vec4<f32>(xyplus*abs(b.z)+xzplus*abs(b.y)+yzplus*abs(b.x),xyplus*bl+xlplus*abs(b.y)+ylplus*abs(b.x),xzplus*bl+xlplus*abs(b.z)+zlplus*abs(b.x),yzplus*bl+ylplus*abs(b.z)+zlplus*abs(b.y));
    return GaoSphere(coefficients,bounds);
}
// (inside, trusted). Uncertainty never becomes an outside classification.
fn gao_sphere_test(s:GaoSphere,id:u32)->vec2<bool>{
    let a=gao_point(id)-gao_origin;let al=dot(a,a);
    let value=gao_round(gao_round(a.x*s.coefficient.z-a.y*s.coefficient.y)+gao_round(a.z*s.coefficient.x-al*s.coefficient.w));
    let error=max(1e-28,4e-6*(s.bound.x*al+s.bound.y*abs(a.z)+s.bound.z*abs(a.y)+s.bound.w*abs(a.x)));
    let trusted=abs(value)>error&&abs(value)<3.0e37&&error<3.0e37;
    return vec2<bool>(trusted&&value<0.0,trusted);
}
fn gao_conflict(encoded_id:u32,t:vec4<u32>)->vec2<bool>{
    let id=encoded_id&~GAO_GREATER;let ids=gao_global_triangle(t);
    return gao_sphere_test(gao_prepare(ids),id);
}
// Outward-rounded sphere/AABB enclosure. It replaces source cell_range's
// fp64 arithmetic, not its near-first search. An ambiguous denominator uses
// the whole data box. No heuristic neighborhood truncation is applied.
struct GaoInterval{lo:f32,hi:f32}
fn gao_down(x:f32)->f32{if(!(abs(x)<3.4e38)){return -bitcast<f32>(0x7f800000u^p.zero);}if(abs(x)<1.17549435e-38){return -1.17549435e-38;}let b=bitcast<u32>(x)^p.zero;return bitcast<f32>(select(b-1u,b+1u,x<0.0));}
fn gao_up(x:f32)->f32{if(!(abs(x)<3.4e38)){return bitcast<f32>(0x7f800000u^p.zero);}if(abs(x)<1.17549435e-38){return 1.17549435e-38;}let b=bitcast<u32>(x)^p.zero;return bitcast<f32>(select(b+1u,b-1u,x<0.0));}
fn gao_product(a:GaoInterval,b:GaoInterval)->GaoInterval{let v=vec4<f32>(a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi);return GaoInterval(gao_down(min(min(v.x,v.y),min(v.z,v.w))),gao_up(max(max(v.x,v.y),max(v.z,v.w))));}
struct GaoRange{low:vec3<i32>,high:vec3<i32>,start:vec3<i32>}
// Conservative cell rejection in the prepared sphere polynomial itself.
// This also handles a planar (infinite-radius) query without constructing an
// enormous centre and subtracting nearly equal squared distances.
fn gao_cell_outside(s:GaoSphere,cell:vec3<i32>)->bool{
    var low:vec3<f32>;var high:vec3<f32>;
    for(var k=0u;k<3u;k++){
        let a=gao_down(gao_down(gao_down(f32(cell[k]))/p.scale[k]));
        let b=gao_up(gao_up(gao_up(f32(cell[k]+1))/p.scale[k]));
        low[k]=gao_down(gao_down(p.lower[k]+a)-gao_origin[k]);
        high[k]=gao_up(gao_up(p.lower[k]+b)-gao_origin[k]);
    }
    let maximum=max(abs(low),abs(high));
    let minimum=max(vec3<f32>(0.0),max(low,-high));
    let low_lift=dot(minimum,minimum);let high_lift=dot(maximum,maximum);
    let co=vec3<f32>(s.coefficient.z,-s.coefficient.y,s.coefficient.x);
    let linear=min(co*low,co*high);
    let value=(linear.x+linear.y)+linear.z-s.coefficient.w*select(low_lift,high_lift,s.coefficient.w>=0.0);
    let error=max(1e-28,8e-6*(s.bound.x*high_lift+s.bound.y*maximum.z+s.bound.z*maximum.y+s.bound.w*maximum.x));
    return value>error&&abs(value)<3.0e37&&error<3.0e37;
}
fn gao_range(s:GaoSphere)->GaoRange{
    let full=GaoRange(vec3<i32>(0),vec3<i32>(i32(p.dim)-1),gao_cell(gao_origin));
    let co=vec4<f32>(s.coefficient.z,-s.coefficient.y,s.coefficient.x,s.coefficient.w);
    let error=max(vec4<f32>(1e-28),4e-6*vec4<f32>(s.bound.w,s.bound.z,s.bound.y,s.bound.x));
    let denominator=GaoInterval(gao_down(co.w-error.w),gao_up(co.w+error.w));
    if(denominator.lo<=0.0&&denominator.hi>=0.0){return full;}
    let reciprocal=GaoInterval(gao_down(0.5/denominator.hi),gao_up(0.5/denominator.lo));
    var low:vec3<f32>;var high:vec3<f32>;var radius2=0.0;var centre:vec3<f32>;
    for(var k=0u;k<3u;k++){
        let c=gao_product(GaoInterval(gao_down(co[k]-error[k]),gao_up(co[k]+error[k])),reciprocal);
        let size=max(abs(c.lo),abs(c.hi));if(!(size<1e15)){return full;}
        radius2=gao_up(radius2+gao_up(size*size));low[k]=gao_down(gao_origin[k]+c.lo);high[k]=gao_up(gao_origin[k]+c.hi);centre[k]=gao_origin[k]+co[k]/(2.0*co.w);
    }
    var distance:vec3<f32>;
    for(var k=0u;k<3u;k++){distance[k]=max(0.0,max(gao_down(p.lower[k]-high[k]),gao_down(low[k]-p.upper[k])));}
    var lo:vec3<i32>;var hi:vec3<i32>;
    for(var k=0u;k<3u;k++){
        let j=(k+1u)%3u;let l=(k+2u)%3u;
        let other=gao_down(gao_down(distance[j]*distance[j])+gao_down(distance[l]*distance[l]));
        let r=gao_up(sqrt(max(0.0,gao_up(radius2-other))));
        let a=gao_down(gao_down(gao_down(low[k]-r)-p.lower[k])*p.scale[k]);
        let b=gao_up(gao_up(gao_up(high[k]+r)-p.lower[k])*p.scale[k]);
        lo[k]=i32(clamp(floor(a),0.0,f32(p.dim-1u)));hi[k]=i32(clamp(floor(b),0.0,f32(p.dim-1u)));
    }
    let inside=all(centre>=p.lower.xyz)&&all(centre<=p.upper.xyz);
    return GaoRange(lo,hi,clamp(gao_cell(select(gao_origin,centre,inside)),lo,hi));
}
