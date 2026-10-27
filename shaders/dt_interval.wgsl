struct Interval{lo:f32,hi:f32}
fn down(x:f32)->f32{if(!(abs(x)<3.4e38)){return -bitcast<f32>(0x7f800000u^p.zero);}if(abs(x)<1.17549435e-38){return -1.17549435e-38;}let b=bitcast<u32>(x)^p.zero;return bitcast<f32>(select(b-1u,b+1u,x<0.0));}
fn up(x:f32)->f32{if(!(abs(x)<3.4e38)){return bitcast<f32>(0x7f800000u^p.zero);}if(abs(x)<1.17549435e-38){return 1.17549435e-38;}let b=bitcast<u32>(x)^p.zero;return bitcast<f32>(select(b+1u,b-1u,x<0.0));}
fn iv(x:f32)->Interval{return Interval(x,x);}
fn plus(a:Interval,b:Interval)->Interval{return Interval(down(a.lo+b.lo),up(a.hi+b.hi));}
fn minus(a:Interval,b:Interval)->Interval{return Interval(down(a.lo-b.hi),up(a.hi-b.lo));}
fn times(a:Interval,b:Interval)->Interval{let v=vec4<f32>(a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi);if(!all(abs(v)<vec4<f32>(3.4e38))){return Interval(down(3.4e38),up(3.4e38));}return Interval(down(min(min(v.x,v.y),min(v.z,v.w))),up(max(max(v.x,v.y),max(v.z,v.w))));}
fn quotient(a:Interval,b:Interval)->Interval{return times(a,Interval(down(1.0/b.hi),up(1.0/b.lo)));}
fn determinant_i(a:array<Interval,3>,b:array<Interval,3>,c:array<Interval,3>)->Interval {
    return plus(minus(times(a[0],minus(times(b[1],c[2]),times(b[2],c[1]))),times(a[1],minus(times(b[0],c[2]),times(b[2],c[0])))),times(a[2],minus(times(b[0],c[1]),times(b[1],c[0]))));
}
// Coordinate extrema of a sphere intersected with the data's bounding box.
// Minimum distances in the other two coordinates tighten each axis interval.
fn clipped_sphere(low:vec3<f32>,high:vec3<f32>,radius2:f32)->mat2x3<f32>{
    var distance:vec3<f32>;
    for(var k=0u;k<3u;k++){distance[k]=max(0.0,max(down(p.lower[k]-high[k]),down(low[k]-p.scale[k+1u])));}
    var a:vec3<f32>;var b:vec3<f32>;
    for(var k=0u;k<3u;k++){
        let j=(k+1u)%3u;let l=(k+2u)%3u;
        let other=down(down(distance[j]*distance[j])+down(distance[l]*distance[l]));
        let r=up(sqrt(max(0.0,up(radius2-other))));a[k]=down(low[k]-r);b[k]=up(high[k]+r);
    }
    return mat2x3<f32>(a,b);
}
