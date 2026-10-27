// Barzilai-Borwein relaxation step, resident on the GPU: no readback between steps.
// Each step: relax_stats (per site, workgroup partial sums) -> relax_sums (one
// workgroup) -> four median_count/median_pick radix passes (exact median of the
// curvatures, both middle ranks) -> relax_move (per site move, BB history).
// The last pick also sets the step length alpha.
struct Row { energy: vec4<f32>, extra: vec4<f32>, force: vec4<f32> }
// fixed: index+1 of the held identity, or 0. round: radix pass 0..3, most significant first.
struct RelaxParams { n: u32, fixed: u32, round: u32, partials: u32, radius: f32, alpha0: f32, step_scale: f32, pad: f32 }
@group(0) @binding(0) var<uniform> u: RelaxParams;
@group(0) @binding(1) var<storage, read_write> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rows: array<Row>;
// Per site: force (xyz) at 2i and move (xyz) at 2i+1 of the previous step.
@group(0) @binding(3) var<storage, read_write> previous: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> state: array<atomic<u32>, 16>;
@group(0) @binding(5) var<storage, read_write> partials: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> histogram: array<atomic<u32>, 512>;

// state: iteration, flags (1 non-finite force, 2 duplicate orientation), largest move
// (f32 bits), s.y, y.y, s.s, f.f, energy (f32 bits), then the radix select: remaining
// rank and bit prefix for the lower and upper middle ranks, median, alpha.
const ITERATION=0u; const FLAGS=1u; const STEP=2u; const SUMS=3u;
const RANK0=8u; const PREFIX0=9u; const RANK1=10u; const PREFIX1=11u; const MEDIAN=12u; const ALPHA=13u;

var<workgroup> reduce_a: array<vec4<f32>, 64>;
var<workgroup> reduce_b: array<f32, 64>;

fn finite3(v: vec3<f32>) -> bool { return all(v == v) && all(abs(v) <= vec3<f32>(3.4e38)); }

@compute @workgroup_size(64)
fn relax_stats(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let i = gid.x;
    var dots = vec4<f32>(0.0); var energy = 0.0;
    let first = atomicLoad(&state[ITERATION]) == 0u;
    if (i < u.n) {
        let row = rows[i]; let f = row.force.xyz;
        if (!finite3(f)) { atomicOr(&state[FLAGS], 1u); }
        if (!(row.extra.z > 0.0)) { atomicOr(&state[FLAGS], 2u); }
        dots.w = dot(f, f); energy = row.extra.y;
        if (!first && i + 1u != u.fixed) {
            let s = previous[2u * i + 1u].xyz; let y = f - previous[2u * i].xyz;
            dots = vec4<f32>(dot(s, y), dot(y, y), dot(s, s), dots.w);
        }
    }
    reduce_a[lane] = dots; reduce_b[lane] = energy;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (lane < stride) { reduce_a[lane] += reduce_a[lane + stride]; reduce_b[lane] += reduce_b[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) { partials[2u * wid.x] = reduce_a[0]; partials[2u * wid.x + 1u] = vec4<f32>(reduce_b[0], 0.0, 0.0, 0.0); }
}

@compute @workgroup_size(64)
fn relax_sums(@builtin(local_invocation_index) lane: u32) {
    var dots = vec4<f32>(0.0); var energy = 0.0;
    for (var k = lane; k < u.partials; k += 64u) { dots += partials[2u * k]; energy += partials[2u * k + 1u].x; }
    reduce_a[lane] = dots; reduce_b[lane] = energy;
    for (var b = lane; b < 512u; b += 64u) { atomicStore(&histogram[b], 0u); }
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (lane < stride) { reduce_a[lane] += reduce_a[lane + stride]; reduce_b[lane] += reduce_b[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) {
        for (var k = 0u; k < 4u; k++) { atomicStore(&state[SUMS + k], bitcast<u32>(reduce_a[0][k])); }
        atomicStore(&state[SUMS + 4u], bitcast<u32>(reduce_b[0]));
        // Middle ranks: equal for odd n, adjacent for even n.
        atomicStore(&state[RANK0], (u.n - 1u) / 2u); atomicStore(&state[RANK1], u.n / 2u);
        atomicStore(&state[PREFIX0], 0u); atomicStore(&state[PREFIX1], 0u);
        atomicStore(&state[STEP], 0u);
    }
}

// Non-negative f32 bits order like the values.
fn curvature_bits(i: u32) -> u32 { let c = rows[i].force.w; return bitcast<u32>(select(0.0, c, c > 0.0)); }

var<workgroup> local_histogram: array<atomic<u32>, 512>;

@compute @workgroup_size(256)
fn median_count(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(num_workgroups) groups: vec3<u32>) {
    atomicStore(&local_histogram[lane], 0u); atomicStore(&local_histogram[lane + 256u], 0u);
    workgroupBarrier();
    let shift = 24u - 8u * u.round;
    let high = select(0u, 0xffffffffu << (32u - 8u * u.round), u.round > 0u);
    let prefix0 = atomicLoad(&state[PREFIX0]); let prefix1 = atomicLoad(&state[PREFIX1]);
    for (var i = gid.x; i < u.n; i += groups.x * 256u) {
        let bits = curvature_bits(i); let digit = (bits >> shift) & 255u;
        if ((bits & high) == prefix0) { atomicAdd(&local_histogram[digit], 1u); }
        if ((bits & high) == prefix1) { atomicAdd(&local_histogram[256u + digit], 1u); }
    }
    workgroupBarrier();
    for (var b = lane; b < 512u; b += 256u) {
        let count = atomicLoad(&local_histogram[b]);
        if (count > 0u) { atomicAdd(&histogram[b], count); }
    }
}

@compute @workgroup_size(256)
fn median_pick(@builtin(local_invocation_index) lane: u32) {
    if (lane == 0u) {
        let shift = 24u - 8u * u.round;
        for (var r = 0u; r < 2u; r++) {
            var remaining = atomicLoad(&state[RANK0 + 2u * r]);
            var digit = 255u;
            for (var d = 0u; d < 256u; d++) {
                let count = atomicLoad(&histogram[256u * r + d]);
                if (remaining < count) { digit = d; break; }
                remaining -= count;
            }
            atomicStore(&state[RANK0 + 2u * r], remaining);
            atomicStore(&state[PREFIX0 + 2u * r], atomicLoad(&state[PREFIX0 + 2u * r]) | (digit << shift));
        }
        if (u.round == 3u) {
            // Median curvature, then the BB step length (alternating BB1/BB2).
            let median = max(1e-12, 0.5 * (bitcast<f32>(atomicLoad(&state[PREFIX0])) + bitcast<f32>(atomicLoad(&state[PREFIX1]))));
            atomicStore(&state[MEDIAN], bitcast<u32>(median));
            let iteration = atomicLoad(&state[ITERATION]);
            var alpha = u.alpha0;
            if (iteration > 0u) {
                let sty = bitcast<f32>(atomicLoad(&state[SUMS])); let yty = bitcast<f32>(atomicLoad(&state[SUMS + 1u])); let sts = bitcast<f32>(atomicLoad(&state[SUMS + 2u]));
                let a1 = select(u.alpha0, abs(sty / yty), abs(yty) > 1e-20);
                let a2 = select(u.alpha0, abs(sts / sty), abs(sty) > 1e-20);
                alpha = clamp(select(a2, a1, iteration % 2u == 1u), 1e-12, max(1.0, 10.0 * u.radius));
            }
            atomicStore(&state[ALPHA], bitcast<u32>(alpha));
            atomicStore(&state[ITERATION], iteration + 1u);
        }
    }
    workgroupBarrier();
    // This pass's counts are consumed: clear them for the next pass.
    atomicStore(&histogram[lane], 0u); atomicStore(&histogram[lane + 256u], 0u);
}

fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.x*b.x-dot(a.yzw,b.yzw), a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}

// Curvature-scaled BB move, clipped to half the mean neighbour radius, applied as a left
// rotation exp(d) q. A held identity stays exactly (1,0,0,0). After a flagged step nothing moves.
@compute @workgroup_size(64)
fn relax_move(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n || atomicLoad(&state[FLAGS]) != 0u) { return; }
    if (i + 1u == u.fixed) {
        previous[2u * i] = vec4<f32>(0.0); previous[2u * i + 1u] = vec4<f32>(0.0);
        q[i] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
        return;
    }
    let alpha = bitcast<f32>(atomicLoad(&state[ALPHA])); let median = bitcast<f32>(atomicLoad(&state[MEDIAN]));
    let f = rows[i].force;
    let scale = alpha * u.step_scale / max(1e-6, f.w / median + 1e-2);
    let size = length(f.xyz) * scale;
    let clip = min(1.0, 0.5 * u.radius / max(size, 1e-12));
    let d = f.xyz * (scale * clip);
    previous[2u * i] = vec4<f32>(f.xyz, 0.0); previous[2u * i + 1u] = vec4<f32>(d, 0.0);
    atomicMax(&state[STEP], bitcast<u32>(size * clip));
    let angle = length(d);
    var sinc = 1.0 - angle * angle / 6.0;
    if (angle > 1e-4) { sinc = sin(angle) / angle; }
    var moved = normalize(qmul(vec4<f32>(cos(angle), d * sinc), q[i]));
    if (moved.x < 0.0) { moved = -moved; }
    q[i] = moved;
}
