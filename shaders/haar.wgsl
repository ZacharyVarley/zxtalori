@group(0) @binding(0) var<storage,read_write> q:array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params:vec4<u32>;
fn hash(v:u32)->u32{var x=v;x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;return x^(x>>16u);}
fn random_unit(v:u32)->f32{return (f32(hash(v)>>8u)+0.5)/16777216.0;}
@compute @workgroup_size(256)
fn haar(@builtin(global_invocation_id) id:vec3<u32>){
 if(id.x>=params.x){return;}let base=hash(id.x^hash(params.y));
 let u=random_unit(base);let a=6.28318530718*random_unit(base+1u);let b=6.28318530718*random_unit(base+2u);
 q[id.x]=vec4<f32>(sqrt(u)*cos(b),sqrt(1.0-u)*sin(a),sqrt(1.0-u)*cos(a),sqrt(u)*sin(b));
}
