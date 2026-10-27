// Gao-style prepared sphere equation, translated around the defining seed.
// Constants are computed once per face, rather than once per candidate point.
// High/low words and permanent coefficients remain in the star's scratch space.
fn prepare_query_fast(s:u32,v:vec4<u32>){
    let mem=32u;set_word(s,mem+7u,0u);set_word(s,mem+23u,0u);
    let input=predicate_points(v,NONE,0u);if(!input.valid){return;}
    let a=input.positions[0]-input.origin;let b=input.positions[1]-input.origin;let c=input.positions[2]-input.origin;
    let lift=vec3<f32>(dot(a,a),dot(b,b),dot(c,c));
    let bc=cross(b,c);let ca=cross(c,a);let ab=cross(a,b);
    let coeff=vec4<f32>(lift.x*bc+lift.y*ca+lift.z*ab,dot(a,bc));
    let bcp=abs(b.yzx*c.zxy)+abs(b.zxy*c.yzx);let cap=abs(c.yzx*a.zxy)+abs(c.zxy*a.yzx);let abp=abs(a.yzx*b.zxy)+abs(a.zxy*b.yzx);
    let bound=vec4<f32>(lift.x*bcp+lift.y*cap+lift.z*abp,permanent(a,b,c));
    for(var k=0u;k<4u;k++){set_word(s,mem+8u+2u*k,bitcast<u32>(coeff[k]));set_word(s,mem+9u+2u*k,0u);set_word(s,mem+16u+k,bitcast<u32>(bound[k]));}
    for(var k=0u;k<3u;k++){set_word(s,mem+20u+k,bitcast<u32>(input.origin[k]));}
    set_word(s,mem+7u,u32(input.shift+256));
}
fn prepared_fast(s:u32,e:u32)->i32{
    let mem=32u;let flag=word(s,mem+7u);if(flag==0u){return 2;}
    let query=scaled_point(e,i32(flag)-256);if(!query.valid||any(abs(query.value)>vec3<f32>(1e6))){return 2;}
    let origin=bitcast<vec3<f32>>(vec3<u32>(word(s,mem+20u),word(s,mem+21u),word(s,mem+22u)));let d=query.value-origin;
    let coefficients=bitcast<vec4<f32>>(vec4<u32>(word(s,mem+8u),word(s,mem+10u),word(s,mem+12u),word(s,mem+14u)));
    let bounds=bitcast<vec4<f32>>(vec4<u32>(word(s,mem+16u),word(s,mem+17u),word(s,mem+18u),word(s,mem+19u)));
    let lift=dot(d,d);let value=rounded(coefficients.w*lift-dot(coefficients.xyz,d));let bound=bounds.w*lift+dot(bounds.xyz,abs(d));
    if(abs(value)>max(1e-28,4e-6*bound)&&abs(value)<3e37){return select(-1,1,value>0.0);}
    if(!POST_INLINE||word(s,mem+23u)!=1u){return 2;}
    // A refined face needs only four scalar high/low products per candidate.
    // Keep the matrix/cofactor construction out of this entry point.
    let dx=ds_sub(vec2<f32>(query.value.x,0.0),vec2<f32>(origin.x,0.0));
    let dy=ds_sub(vec2<f32>(query.value.y,0.0),vec2<f32>(origin.y,0.0));
    let dz=ds_sub(vec2<f32>(query.value.z,0.0),vec2<f32>(origin.z,0.0));
    let l=ds_add(ds_add(ds_mul(dx,dx),ds_mul(dy,dy)),ds_mul(dz,dz));
    let cx=bitcast<vec2<f32>>(vec2<u32>(word(s,mem+8u),word(s,mem+9u)));
    let cy=bitcast<vec2<f32>>(vec2<u32>(word(s,mem+10u),word(s,mem+11u)));
    let cz=bitcast<vec2<f32>>(vec2<u32>(word(s,mem+12u),word(s,mem+13u)));
    let ca=bitcast<vec2<f32>>(vec2<u32>(word(s,mem+14u),word(s,mem+15u)));
    let refined=ds_sub(ds_mul(ca,l),ds_add(ds_add(ds_mul(cx,dx),ds_mul(cy,dy)),ds_mul(cz,dz)));
    let result=rounded(refined.x+refined.y);if(abs(result)>max(1e-28,1e-12*bound)&&abs(result)<3e37){return select(-1,1,result>0.0);}return 2;
}
fn prepare_query(s:u32,v:vec4<u32>){
    let mem=32u;set_word(s,mem+23u,2u);
    let input=predicate_points(v,NONE,0u);if(!input.valid){return;}
    let a=ds3_delta(input.positions[0],input.origin);let b=ds3_delta(input.positions[1],input.origin);let c=ds3_delta(input.positions[2],input.origin);
    let la=ds3_dot(a,a);let lb=ds3_dot(b,b);let lc=ds3_dot(c,c);let bc=ds3_cross(b,c);let ca=ds3_cross(c,a);let ab=ds3_cross(a,b);
    let coeff=ds3_add(ds3_add(ds3_scale(bc,la),ds3_scale(ca,lb)),ds3_scale(ab,lc));let determinant=ds3_dot(a,bc);
    let high=vec4<f32>(coeff.hi,determinant.x);let low=vec4<f32>(coeff.lo,determinant.y);
    let bcp=abs(b.hi.yzx*c.hi.zxy)+abs(b.hi.zxy*c.hi.yzx);let cap=abs(c.hi.yzx*a.hi.zxy)+abs(c.hi.zxy*a.hi.yzx);let abp=abs(a.hi.yzx*b.hi.zxy)+abs(a.hi.zxy*b.hi.yzx);
    let bound=vec4<f32>(la.x*bcp+lb.x*cap+lc.x*abp,permanent(a.hi,b.hi,c.hi));
    for(var k=0u;k<4u;k++){set_word(s,mem+8u+2u*k,bitcast<u32>(high[k]));set_word(s,mem+9u+2u*k,bitcast<u32>(low[k]));set_word(s,mem+16u+k,bitcast<u32>(bound[k]));}
    for(var k=0u;k<3u;k++){set_word(s,mem+20u+k,bitcast<u32>(input.origin[k]));}
    set_word(s,mem+7u,u32(input.shift+256));set_word(s,mem+23u,1u);
}
fn prepared_sign(s:u32,e:u32)->i32{
    let mem=32u;let flag=word(s,mem+7u);if(flag==0u){return 2;}
    let query=scaled_point(e,i32(flag)-256);if(!query.valid||any(abs(query.value)>vec3<f32>(1e6))){return 2;}
    var delta:array<vec2<f32>,3>;var high:vec3<f32>;var coefficients:array<vec2<f32>,4>;var bounds:vec4<f32>;
    for(var k=0u;k<3u;k++){delta[k]=ds_sub(vec2<f32>(query.value[k],0.0),vec2<f32>(bitcast<f32>(word(s,mem+20u+k)),0.0));high[k]=delta[k].x;}
    for(var k=0u;k<4u;k++){coefficients[k]=bitcast<vec2<f32>>(vec2<u32>(word(s,mem+8u+2u*k),word(s,mem+9u+2u*k)));bounds[k]=bitcast<f32>(word(s,mem+16u+k));}
    let lift_high=dot(high,high);let bound=bounds.w*lift_high+dot(bounds.xyz,abs(high));
    let fast=rounded(rounded(coefficients[3].x*lift_high)-dot(vec3<f32>(coefficients[0].x,coefficients[1].x,coefficients[2].x),high));
    if(abs(fast)>max(1e-28,4e-6*bound)&&abs(fast)<3e37){return select(-1,1,fast>0.0);}
    var lift=vec2<f32>(0.0);for(var k=0u;k<3u;k++){lift=ds_add(lift,ds_mul(delta[k],delta[k]));}
    var value=ds_mul(coefficients[3],lift);for(var k=0u;k<3u;k++){value=ds_sub(value,ds_mul(coefficients[k],delta[k]));}
    let result=rounded(value.x+value.y);if(abs(result)>max(1e-28,1e-12*bound)&&abs(result)<3e37){return select(-1,1,result>0.0);}return 2;
}
fn prepared_box(s:u32,v:vec4<u32>,relative_error:f32)->QueryBox{
    let full=QueryBox(vec3<u32>(0),vec3<u32>(p.dim-1u));
    let mem=32u;set_word(s,mem+6u,0u);if(word(s,mem+7u)==0u){return full;}
    let shift=i32(word(s,mem+7u))-256;if(shift < -100||shift>100){return full;}
    let factor=bitcast<f32>(u32(127-shift)<<23u);
    var coefficient:array<Interval,4>;
    for(var k=0u;k<4u;k++){
        let h=bitcast<f32>(word(s,mem+8u+2u*k));let l=bitcast<f32>(word(s,mem+9u+2u*k));
        let error=max(1e-28,relative_error*bitcast<f32>(word(s,mem+16u+k)));
        coefficient[k]=Interval(down(down(h+l)-error),up(up(h+l)+error));
    }
    let determinant=coefficient[3];if(determinant.lo<=0.0&&determinant.hi>=0.0){return full;}
    let origin=point(v.x);var low:vec3<f32>;var high:vec3<f32>;var radius=iv(0.0);
    for(var k=0u;k<3u;k++){
        let centre=times(iv(0.5*factor),quotient(coefficient[k],determinant));
        if(!(max(abs(centre.lo),abs(centre.hi))<1e15)){return full;}
        set_word(s,mem+2u*k,bitcast<u32>(centre.lo));set_word(s,mem+2u*k+1u,bitcast<u32>(centre.hi));
        low[k]=down(origin[k]+centre.lo);high[k]=up(origin[k]+centre.hi);
        let bound=max(abs(centre.lo),abs(centre.hi));radius=plus(radius,times(iv(bound),iv(bound)));
    }
    let clipped=clipped_sphere(low,high,radius.hi);
    // The bucket map is monotone. Outward rounding its subtraction and product
    // encloses every mapped site without adding an entire cell on either side.
    var mapped_low:vec3<f32>;var mapped_high:vec3<f32>;
    for(var k=0u;k<3u;k++){mapped_low[k]=down(down(clipped[0][k]-p.lower[k])*p.scale.x);mapped_high[k]=up(up(clipped[1][k]-p.lower[k])*p.scale.x);}
    let lo=vec3<u32>(clamp(floor(mapped_low),vec3<f32>(0),vec3<f32>(f32(p.dim-1u))));
    let hi=vec3<u32>(clamp(floor(mapped_high),vec3<f32>(0),vec3<f32>(f32(p.dim-1u))));
    set_word(s,mem+6u,1u);return QueryBox(lo,hi);
}
fn refine_query_box(s:u32,v:vec4<u32>){
    let mem=32u;if(word(s,mem+23u)!=1u){return;}
    let box=prepared_box(s,v,1e-12);let origin=point(v.x);
    let cell=clamp(vec3<u32>(clamp(floor((origin-p.lower.xyz)*p.scale.x),vec3<f32>(0),vec3<f32>(f32(p.dim-1u)))),box.lo,box.hi);
    stars[s].low=vec4<u32>(box.lo,0u);stars[s].high=vec4<u32>(box.hi,0u);stars[s].cell=vec4<u32>(cell,NONE);stars[s].more.x=0u;stars[s].info.z=cell.x+p.dim*(cell.y+p.dim*cell.z);
}
