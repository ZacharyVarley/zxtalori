@group(0) @binding(0) var<storage, read_write> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> ops: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;
fn mul(a:vec4<f32>,b:vec4<f32>)->vec4<f32>{return vec4<f32>(a.x*b.x-dot(a.yzw,b.yzw),a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));}
// Port of laue_ops.py::ori_in_fz_laue. IDs use this project's group table.
// Homogeneous quaternion inequalities avoid Rodrigues division at pi.
fn in_zone(p:vec4<f32>)->bool {
    let a=abs(p.yzw); let w=p.x;
    switch u.z {
        case 0u:{return a.z<=w;}
        case 1u:{return a.z<=0.5773502691896258*w;}
        case 2u:{return a.z<=0.4142135623730951*w;}
        case 3u:{return a.z<=0.2679491924311227*w;}
        case 4u:{return max(max(a.x,a.y),a.z)<=w;}
        case 5u:{return a.z<=0.5773502691896258*w && max(a.x,0.5*a.x+0.8660254037844386*a.y)<=w;}
        case 6u:{return a.z<=0.4142135623730951*w && max(max(a.x,a.y),0.7071067811865476*(a.x+a.y))<=w;}
        case 7u:{let m=max(a.x,a.y);let n=min(a.x,a.y);return a.z<=0.2679491924311227*w && max(m,0.8660254037844386*m+0.5*n)<=w;}
        case 8u:{return a.x+a.y+a.z<=w;}
        case 9u:{return max(max(a.x,a.y),a.z)<=0.4142135623730951*w && a.x+a.y+a.z<=w;}
        default:{
            if(w>0.9510566){return true;}
            if(w<0.9256147){return false;}
            // Only the twelve face-defining five-fold rotations are needed.
            for(var k=1u;k<u.y;k++){
                if(abs(abs(ops[k].x)-0.8090169943749475)<1e-5 && abs(dot(p,ops[k]))>w){return false;}
            }
            return true;
        }
    }
}
@compute @workgroup_size(256)
fn fold(@builtin(global_invocation_id) id:vec3<u32>){
    if(id.x>=u.x){return;}
    // Scale by a power of two through exponent bits. Division by a very large
    // maximum can flush its reciprocal to zero on GPU, even with finite input.
    // The host rejects near-zero norms, so the largest component is normal.
    let a=q[id.x];let scale=max(max(abs(a.x),abs(a.y)),max(abs(a.z),abs(a.w)));
    let bits=bitcast<vec4<u32>>(a);
    let exponent=vec4<i32>((bits>>vec4<u32>(23u))&vec4<u32>(255u));
    let maximum=i32((bitcast<u32>(scale)>>23u)&255u);
    let scaled_exponent=exponent+vec4<i32>(127-maximum);
    // Values that become subnormal after scaling are far below f32 accuracy.
    let shifted=(bits&vec4<u32>(0x807fffffu))|(vec4<u32>(max(scaled_exponent,vec4<i32>(0)))<<vec4<u32>(23u));
    let keep=(exponent>vec4<i32>(0))&(scaled_exponent>vec4<i32>(0));
    let scaled=bitcast<vec4<f32>>(select(bits&vec4<u32>(0x80000000u),shifted,keep));
    var p=normalize(scaled); if(p.x<0.0){p=-p;}
    if(in_zone(p)){q[id.x]=p;return;}
    var best=p; var w=p.x;
    for(var k=0u;k<u.y;k++){let c=mul(p,ops[k]);if(abs(c.x)>w){best=select(-c,c,c.x>=0.0);w=abs(c.x);}}
    q[id.x]=normalize(best);
}
