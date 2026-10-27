// SPDX-License-Identifier: MIT
// ZYZ conversion and right-K fold for the archived MTEX equispacedSO3Grid port.
@group(0) @binding(0) var<storage, read> eulers: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> ops: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> quats: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
fn mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.x*b.x-dot(a.yzw,b.yzw), a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.x) { return; }
    let e=eulers[id.x];
    let cb=cos(e.y*0.5); let sb=sin(e.y*0.5);
    let sp=(e.x+e.z)*0.5; let sm=(e.x-e.z)*0.5;
    let q=normalize(vec4<f32>(cb*cos(sp), -sb*sin(sm), sb*cos(sm), cb*sin(sp)));
    var best=q; var w=-1.0;
    for (var k=0u; k<params.y; k++) {
        let candidate=mul(q,ops[k]);
        if (abs(candidate.x)>w) { best=select(-candidate,candidate,candidate.x>=0.0); w=abs(candidate.x); }
    }
    quats[id.x]=normalize(best);
}
