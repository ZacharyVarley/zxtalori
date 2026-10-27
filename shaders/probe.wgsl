// Setup diagnostic: dispatch and readback through the reference apps' GPU layer.
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&output)) {
        output[id.x] = f32(id.x) * 2.0 + 1.0;
    }
}

