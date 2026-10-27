// Shared numerical filters; no constructor state.
fn point(i:u32)->vec3<f32>{return bitcast<vec4<f32>>(coordinates[i]).xyz;}
fn rounded(x:f32)->f32{return bitcast<f32>(bitcast<u32>(x)^p.zero);}
fn det(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>)->f32 {
    return rounded(rounded(rounded(a.x*rounded(b.y*c.z-b.z*c.y))-rounded(a.y*rounded(b.x*c.z-b.z*c.x)))+rounded(a.z*rounded(b.x*c.y-b.y*c.x)));
}
fn permanent(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>)->f32 {
    return dot(abs(a),abs(b.yzx*c.zxy)+abs(b.zxy*c.yzx));
}
fn signed_bits(v:vec3<u32>)->vec3<u32>{return select(v,vec3<u32>(0), (v&vec3<u32>(0x7fffffffu))==vec3<u32>(0));}
// Exact power-of-two input normalization through integer exponent bits. This
// also lifts subnormals before float arithmetic, rather than flushing them.
struct NormalPoint{value:vec3<f32>,valid:bool}
struct PredicatePoints{origin:vec3<f32>,positions:array<vec3<f32>,4>,valid:bool,shift:i32}
fn scaled_point(id:u32,shift:i32)->NormalPoint{
    let bits=coordinates[id].xyz;var result:vec3<f32>;
    let magnitude=bits&vec3<u32>(0x7fffffffu);let exponents=(bits>>vec3<u32>(23u))&vec3<u32>(255u);
    let zero=magnitude==vec3<u32>(0);let shifted=vec3<i32>(exponents)+vec3<i32>(shift);
    if(shift>=-126&&shift<=127&&all(zero|(exponents>vec3<u32>(0)))&&all(zero|((shifted>vec3<i32>(0))&(shifted<vec3<i32>(255))))){
        return NormalPoint(bitcast<vec3<f32>>(bits)*bitcast<f32>(u32(127+shift)<<23u),true);
    }
    for(var k=0u;k<3u;k++){let magnitude=bits[k]&0x7fffffffu;if(magnitude==0u){result[k]=0.0;continue;}
        let exponent=(bits[k]>>23u)&255u;var fraction=bits[k]&0x7fffffu;var power=i32(exponent)-127;
        if(exponent==0u){let leading=firstLeadingBit(fraction);power=i32(leading)-149;fraction=(fraction<<(23u-leading))&0x7fffffu;}
        let normalized=power+shift+127;if(normalized<=0||normalized>=255){return NormalPoint(vec3<f32>(0.0),false);}
        result[k]=bitcast<f32>((bits[k]&0x80000000u)|(u32(normalized)<<23u)|fraction);
    }return NormalPoint(result,true);
}
fn predicate_points(v:vec4<u32>,e:u32,kind:u32)->PredicatePoints{
    var maximum=0u;for(var i=0u;i<4u;i++){let b=coordinates[v[i]].xyz&vec3<u32>(0x7fffffffu);maximum=max(maximum,max(max(b.x,b.y),b.z));}
    if(kind==1u){let b=coordinates[e].xyz&vec3<u32>(0x7fffffffu);maximum=max(maximum,max(max(b.x,b.y),b.z));}
    var result:PredicatePoints;result.valid=true;if(maximum==0u){result.origin=vec3<f32>(0.0);for(var i=0u;i<4u;i++){result.positions[i]=vec3<f32>(0.0);}return result;}
    let exponent=(maximum>>23u)&255u;var power=i32(exponent)-127;if(exponent==0u){power=i32(firstLeadingBit(maximum&0x7fffffu))-149;}
    result.shift=-power;let origin=scaled_point(select(v.x,e,kind==1u),-power);result.origin=origin.value;if(!origin.valid){result.valid=false;return result;}
    let count=select(3u,4u,kind==1u);for(var i=0u;i<count;i++){let q=scaled_point(v[(i+select(1u,0u,kind==1u))%4u],-power);result.positions[i]=q.value;if(!q.valid){result.valid=false;return result;}}
    return result;
}
// 2 means uncertain. Floating filters never resolve a numerical tie.
fn orient_filter(v:vec4<u32>)->i32 {
    let x=signed_bits(coordinates[v.x].xyz);let y=signed_bits(coordinates[v.y].xyz);let z=signed_bits(coordinates[v.z].xyz);let w=signed_bits(coordinates[v.w].xyz);
    if(any((x==y)&(x==z)&(x==w))){return 0;}
    let input=predicate_points(v,NONE,0u);if(!input.valid){return 2;}
    let a=input.positions[0]-input.origin;let b=input.positions[1]-input.origin;let c=input.positions[2]-input.origin;
    let value=det(a,b,c);let bound=max(1e-28,2e-6*permanent(a,b,c));
    if(abs(value)>bound&&abs(value)<3e37){return select(-1,1,value>0.0);}return 2;
}
fn sphere_filter(v:vec4<u32>,e:u32)->i32 {
    let input=predicate_points(v,e,1u);if(!input.valid){return 2;}
    let a=input.positions[0]-input.origin;let b=input.positions[1]-input.origin;let c=input.positions[2]-input.origin;let d=input.positions[3]-input.origin;
    let lift=vec4<f32>(dot(a,a),dot(b,b),dot(c,c),dot(d,d));
    let minors=vec4<f32>(det(b,c,d),det(a,c,d),det(a,b,d),det(a,b,c));
    let value=rounded(rounded(-rounded(lift.x*minors.x)+rounded(lift.y*minors.y))+rounded(-rounded(lift.z*minors.z)+rounded(lift.w*minors.w)));
    let bound=max(1e-28,4e-6*dot(lift,vec4<f32>(permanent(b,c,d),permanent(a,c,d),permanent(a,b,d),permanent(a,b,c))));
    if(abs(value)>bound&&abs(value)<3e37){return select(-1,1,value>0.0);}return 2;
}
// Ordinary bounded coordinates need no per-coordinate exponent normalization.
// Restrict the exponent range so that every product in these filters is normal
// or its error is below the absolute uncertainty floor. Wider data use the
// power-of-two normalized filter and ultimately the exact integer predicate.
fn bounded_point(i:u32)->bool{
    let b=coordinates[i].xyz&vec3<u32>(0x7fffffffu);
    return all(b<=vec3<u32>(0x4a800000u));
}
fn orient_quick(v:vec4<u32>)->i32{
    if(!(bounded_point(v.x)&&bounded_point(v.y)&&bounded_point(v.z)&&bounded_point(v.w))){return 2;}
    let o=point(v.x);let a=point(v.y)-o;let b=point(v.z)-o;let c=point(v.w)-o;
    let magnitude=max(vec3<f32>(1.0),max(max(abs(a),abs(b)),abs(c)));let m=max(max(magnitude.x,magnitude.y),magnitude.z);
    let value=det(a,b,c);let bound=max(max(1e-28,1e-35*m*m),2e-6*permanent(a,b,c));
    if(abs(value)>bound&&abs(value)<3e37){return select(-1,1,value>0.0);}return 2;
}
fn sphere_quick(v:vec4<u32>,e:u32)->i32{
    if(!(bounded_point(v.x)&&bounded_point(v.y)&&bounded_point(v.z)&&bounded_point(v.w)&&bounded_point(e))){return 2;}
    let o=point(e);let a=point(v.x)-o;let b=point(v.y)-o;let c=point(v.z)-o;let d=point(v.w)-o;
    let lift=vec4<f32>(dot(a,a),dot(b,b),dot(c,c),dot(d,d));
    let minors=vec4<f32>(det(b,c,d),det(a,c,d),det(a,b,d),det(a,b,c));
    let value=rounded(rounded(-rounded(lift.x*minors.x)+rounded(lift.y*minors.y))+rounded(-rounded(lift.z*minors.z)+rounded(lift.w*minors.w)));
    let magnitude=max(vec3<f32>(1.0),max(max(abs(a),abs(b)),max(abs(c),abs(d))));let m=max(max(magnitude.x,magnitude.y),magnitude.z);let m2=m*m;
    let bound=max(max(1e-28,1e-35*m2*m2),4e-6*dot(lift,vec4<f32>(permanent(b,c,d),permanent(a,c,d),permanent(a,b,d),permanent(a,b,c))));
    if(abs(value)>bound&&abs(value)<3e37){return select(-1,1,value>0.0);}return 2;
}
