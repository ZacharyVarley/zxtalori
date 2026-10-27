// Gao post-processing prepares its higher-precision sphere once per face.
// Double-single arithmetic replaces CUDA REAL=double without repeating the
// translated determinant and cross products for every point in every cell.
struct GaoDoubleSphere{linear:DS3,determinant:vec2<f32>,bound:vec4<f32>}
fn gao_prepare_double(ids:vec3<u32>,sphere:GaoSphere)->GaoDoubleSphere{
    let b=ds3_delta(gao_point(ids.x),gao_origin);let c=ds3_delta(gao_point(ids.y),gao_origin);let d=ds3_delta(gao_point(ids.z),gao_origin);
    let bc=ds3_cross(b,c);let cd=ds3_cross(c,d);let db=ds3_cross(d,b);
    let linear=ds3_add(ds3_add(ds3_scale(cd,ds3_dot(b,b)),ds3_scale(db,ds3_dot(c,c))),ds3_scale(bc,ds3_dot(d,d)));
    return GaoDoubleSphere(linear,ds3_dot(b,cd),sphere.bound);
}
fn gao_prepared_double_sign(s:GaoDoubleSphere,id:u32)->i32{
    let delta=ds3_delta(gao_point(id),gao_origin);let lift=ds3_dot(delta,delta);
    let value=ds_sub(ds3_dot(s.linear,delta),ds_mul(s.determinant,lift));
    let error=max(1e-28,1e-12*(s.bound.x*lift.x+s.bound.y*abs(delta.hi.z)+s.bound.z*abs(delta.hi.y)+s.bound.w*abs(delta.hi.x)));
    let result=rounded(value.x+value.y);
    if(abs(result)>error&&abs(result)<3e37&&error<3e37){return select(-1,1,result>0.0);}
    return 2;
}
// Reuse the accurately prepared coefficients in the inexpensive binary32
// search and cell bounds too. Their errors depend on coefficient magnitudes
// plus the preparation error, rather than the much larger original minors.
fn gao_refined_sphere(s:GaoDoubleSphere)->GaoSphere{
    let linear=rounded3(s.linear.hi+s.linear.lo);let determinant=rounded(s.determinant.x+s.determinant.y);
    var coefficient=vec4<f32>(linear.z,-linear.y,linear.x,determinant);
    var bounds=vec4<f32>(abs(determinant),abs(linear.z),abs(linear.y),abs(linear.x))+1e-6*s.bound;
    let largest=max(max(abs(coefficient.x),abs(coefficient.y)),max(abs(coefficient.z),abs(coefficient.w)));
    let exponent=i32((bitcast<u32>(largest)>>23u)&255u)-127;
    if(largest>0.0&&exponent>=-126&&exponent<=126){let factor=bitcast<f32>(u32(127-exponent)<<23u);coefficient*=factor;bounds*=factor;}
    return GaoSphere(coefficient,bounds);
}
// Source do_predict also translates each sphere around the target site.
// Keeping that origin avoids the much larger error permanent produced by
// translating all four nearby vertices around a relatively distant candidate.
fn gao_inside_prepared(ids:vec3<u32>,id:u32)->bool{
    let sphere=gao_prepare(ids);let fast=gao_sphere_test(sphere,id);if(fast.y){return fast.x;}
    let sign=gao_prepared_double_sign(gao_prepare_double(ids,sphere),id);
    if(sign!=2){return sign<0;}
    return gao_inside_exact(ids,id);
}
