// Filter only: conservative double-single arithmetic. Unresolved signs still
// go to the integer-only CRT kernel. Runtime bit barriers prevent reassociation.
fn ds_add(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
    let s=rounded(a.x+b.x);let v=rounded(s-a.x);let e=rounded(rounded(rounded(a.x-rounded(s-v))+rounded(b.x-v))+rounded(a.y+b.y));let h=rounded(s+e);return vec2<f32>(h,rounded(e-rounded(h-s)));
}
fn ds_neg(a:vec2<f32>)->vec2<f32>{return -a;}
fn ds_sub(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{return ds_add(a,-b);}
fn ds_mul(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
    let h=rounded(a.x*b.x);let ah=bitcast<f32>(bitcast<u32>(a.x)&0xfffff000u);let bh=bitcast<f32>(bitcast<u32>(b.x)&0xfffff000u);let al=rounded(a.x-ah);let bl=rounded(b.x-bh);
    let err=rounded(rounded(rounded(rounded(ah*bh-h)+rounded(ah*bl))+rounded(al*bh))+rounded(al*bl));
    let low=rounded(err+rounded(rounded(a.x*b.y)+rounded(a.y*b.x)));let high=rounded(h+low);return vec2<f32>(high,rounded(low-rounded(high-h)));
}
fn ds_det(a:array<vec2<f32>,3>,b:array<vec2<f32>,3>,c:array<vec2<f32>,3>)->vec2<f32>{
    return ds_add(ds_sub(ds_mul(a[0],ds_sub(ds_mul(b[1],c[2]),ds_mul(b[2],c[1]))),ds_mul(a[1],ds_sub(ds_mul(b[0],c[2]),ds_mul(b[2],c[0])))),ds_mul(a[2],ds_sub(ds_mul(b[0],c[1]),ds_mul(b[1],c[0]))));
}
// Static vector forms avoid dynamically indexed private matrices in the GPU
// compiler. Each component retains the same high/low rounding barriers.
struct DS3{hi:vec3<f32>,lo:vec3<f32>}
fn rounded3(x:vec3<f32>)->vec3<f32>{return bitcast<vec3<f32>>(bitcast<vec3<u32>>(x)^vec3<u32>(p.zero));}
fn ds3_add(a:DS3,b:DS3)->DS3{
    let s=rounded3(a.hi+b.hi);let v=rounded3(s-a.hi);let e=rounded3(rounded3(rounded3(a.hi-rounded3(s-v))+rounded3(b.hi-v))+rounded3(a.lo+b.lo));let h=rounded3(s+e);return DS3(h,rounded3(e-rounded3(h-s)));
}
fn ds3_sub(a:DS3,b:DS3)->DS3{return ds3_add(a,DS3(-b.hi,-b.lo));}
fn ds3_mul(a:DS3,b:DS3)->DS3{
    let h=rounded3(a.hi*b.hi);let ah=bitcast<vec3<f32>>(bitcast<vec3<u32>>(a.hi)&vec3<u32>(0xfffff000u));let bh=bitcast<vec3<f32>>(bitcast<vec3<u32>>(b.hi)&vec3<u32>(0xfffff000u));let al=rounded3(a.hi-ah);let bl=rounded3(b.hi-bh);
    let err=rounded3(rounded3(rounded3(rounded3(ah*bh-h)+rounded3(ah*bl))+rounded3(al*bh))+rounded3(al*bl));
    let low=rounded3(err+rounded3(rounded3(a.hi*b.lo)+rounded3(a.lo*b.hi)));let high=rounded3(h+low);return DS3(high,rounded3(low-rounded3(high-h)));
}
fn ds3_scale(a:DS3,b:vec2<f32>)->DS3{return ds3_mul(a,DS3(vec3<f32>(b.x),vec3<f32>(b.y)));}
fn ds3_cross(a:DS3,b:DS3)->DS3{return ds3_sub(ds3_mul(DS3(a.hi.yzx,a.lo.yzx),DS3(b.hi.zxy,b.lo.zxy)),ds3_mul(DS3(a.hi.zxy,a.lo.zxy),DS3(b.hi.yzx,b.lo.yzx)));}
fn ds3_dot(a:DS3,b:DS3)->vec2<f32>{let v=ds3_mul(a,b);return ds_add(ds_add(vec2<f32>(v.hi.x,v.lo.x),vec2<f32>(v.hi.y,v.lo.y)),vec2<f32>(v.hi.z,v.lo.z));}
fn ds3_delta(a:vec3<f32>,b:vec3<f32>)->DS3{return ds3_sub(DS3(a,vec3<f32>(0)),DS3(b,vec3<f32>(0)));}
fn double_filter(v:vec4<u32>,e:u32,kind:u32)->i32{
    let input=predicate_points(v,e,kind);if(!input.valid){return 2;}
    let a=ds3_delta(input.positions[0],input.origin);let b=ds3_delta(input.positions[1],input.origin);let c=ds3_delta(input.positions[2],input.origin);
    var result=ds3_dot(a,ds3_cross(b,c));var bound=permanent(a.hi,b.hi,c.hi);
    if(kind==1u){
        let d=ds3_delta(input.positions[3],input.origin);let la=ds3_dot(a,a);let lb=ds3_dot(b,b);let lc=ds3_dot(c,c);let ld=ds3_dot(d,d);
        result=ds_add(ds_sub(ds_mul(lb,ds3_dot(a,ds3_cross(c,d))),ds_mul(la,ds3_dot(b,ds3_cross(c,d)))),ds_sub(ds_mul(ld,ds3_dot(a,ds3_cross(b,c))),ds_mul(lc,ds3_dot(a,ds3_cross(b,d)))));
        bound=la.x*permanent(b.hi,c.hi,d.hi)+lb.x*permanent(a.hi,c.hi,d.hi)+lc.x*permanent(a.hi,b.hi,d.hi)+ld.x*permanent(a.hi,b.hi,c.hi);
    }
    let value=rounded(result.x+result.y);if(abs(value)>max(1e-28,1e-12*bound)){return select(-1,1,value>0.0);}return 2;
}
