// GPU drawing of large samples on the engine's WebGPU device: every orbit image is projected in
// a shader, either as depth-tested discs shaded as the CPU splat shades them, or accumulated per
// pixel into a density image. Picking runs the same projection to find the image nearest a click.
const WGSL = /* wgsl */`
struct View {
  rot0: vec4f, rot1: vec4f, rot2: vec4f,
  ndc: vec4f,      // NDC per data unit (x, y), NDC offset (x, y)
  px: vec4f,       // 2 / width, 2 / height (device pixels), disc radius (device pixels), extent
  canvas: vec4f,   // canvas colour, w device pixels per CSS pixel
  point: vec4f, cut: vec4f,   // cut.x: the cut plane's position on its axis
  value: vec4f,    // value at ramp 0, value at ramp 1, clip radius, colour kept at the far depth
  pixel: vec4f,    // width, height (device pixels), click x, y
  ints: vec4u,     // projection (0 homochoric, 1 Rodrigues, 2 axis-angle), order (1 or |K|), images, stride
  more: vec4u,     // colour (0 single, 1 zone, 2 value), cut axis (3 none), cut sign, density weight
  ramp: array<vec4f, 8>,
  zones: array<vec4f, 60>,
}
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> quats: array<vec4f>;
@group(0) @binding(2) var<storage, read> ops: array<vec4f>;
@group(0) @binding(3) var<storage, read> values: array<f32>;

fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w, a.x*b.y + b.x*a.y + a.z*b.w - a.w*b.z,
               a.x*b.z + b.x*a.z + a.w*b.y - a.y*b.w, a.x*b.w + b.x*a.w + a.y*b.z - a.z*b.y);
}
// Homochoric, Rodrigues or axis-angle coordinates; w = 0 when the point has none.
fn project(q0: vec4f) -> vec4f {
  var q = normalize(q0); if (q.x < 0.0) { q = -q; }
  let v = q.yzw; let s = length(v);
  if (s < 1e-12) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  if (V.ints.x == 1u) { if (q.x < 1e-9) { return vec4f(0.0); } return vec4f(v / q.x, 1.0); }
  let a = 2.0 * atan2(s, q.x);
  var r = a;
  if (V.ints.x == 0u) {
    // a - sin a by its series below 0.5: f32 cancellation would distort the centre.
    let a2 = a * a;
    let d = select(a - sin(a), a * a2 * (1.0/6.0 - a2 * (1.0/120.0 - a2 / 5040.0)), a < 0.5);
    r = pow(max(0.75 * d, 0.0), 1.0 / 3.0);
  }
  return vec4f(v * (r / s), 1.0);
}
fn culled(p: vec4f) -> bool {
  if (p.w == 0.0 || length(p.xyz) > V.value.z) { return true; }
  if (V.more.y < 3u) { let x = p[V.more.y] - V.cut.x; if ((V.more.z == 0u && x < 0.0) || (V.more.z == 1u && x > 0.0)) { return true; } }
  return false;
}
struct Placed { t: vec3f, near: f32, index: u32, op: u32, ok: bool }
fn placed(q: vec4f, index: u32, op: u32) -> Placed {
  var o: Placed; o.index = index; o.op = op;
  let p = project(q);
  o.ok = !culled(p);
  o.t = vec3f(dot(V.rot0.xyz, p.xyz), dot(V.rot1.xyz, p.xyz), dot(V.rot2.xyz, p.xyz));
  o.near = clamp((0.35 + o.t.z / (2.6 * V.px.w)) / 0.7, 0.0, 1.0);
  return o;
}
// Image k of those drawn: every stride-th of the index * order + operation images.
fn place(k: u32) -> Placed {
  let image = k * V.ints.w; let index = image / V.ints.y; let op = image % V.ints.y;
  var q = quats[index]; if (V.ints.y > 1u) { q = qmul(q, ops[op]); }
  return placed(q, index, op);
}
fn ramp(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0) * 7.0; let i = min(6u, u32(x));
  return mix(V.ramp[i].rgb, V.ramp[i + 1u].rgb, x - f32(i));
}
fn ramp_t(index: u32) -> f32 { return (values[index] - V.value.x) / (V.value.y - V.value.x); }
fn base(index: u32, op: u32) -> vec3f {
  if (V.more.x == 1u) { return V.zones[op % 60u].rgb; }
  if (V.more.x == 2u) { return ramp(ramp_t(index)); }
  return V.point.rgb;
}
fn centre(t: vec3f) -> vec2f { return vec2f(V.ndc.z + t.x * V.ndc.x, V.ndc.w + t.y * V.ndc.y); }
fn pixel_of(t: vec3f) -> vec2f { let c = centre(t); return vec2f((c.x + 1.0) * 0.5 * V.pixel.x, (1.0 - c.y) * 0.5 * V.pixel.y); }

struct Out { @builtin(position) pos: vec4f, @location(0) corner: vec2f, @location(1) @interpolate(flat) color: vec4f }
const CORNERS = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
fn disc(vi: u32, p: Placed, radius: f32, z: f32, color: vec3f) -> Out {
  var o: Out; o.corner = CORNERS[vi];
  if (!p.ok) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  o.pos = vec4f(centre(p.t) + o.corner * radius * V.px.xy, z, 1.0);
  o.color = vec4f(color, 1.0);
  return o;
}
// As the CPU splat: nearer discs are larger and keep more of their colour.
@vertex fn points_vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> Out {
  let p = place(k);
  return disc(vi, p, V.px.z * (0.85 + 0.3 * p.near), 0.5 - 0.49 * clamp(p.t.z / (1.5 * V.px.w), -1.0, 1.0),
              mix(V.canvas.rgb, base(p.index, p.op), V.value.w + (1.0 - V.value.w) * p.near));
}
@fragment fn points_fs(i: Out) -> @location(0) vec4f { if (dot(i.corner, i.corner) > 1.0) { discard; } return i.color; }

// Density: each image adds its weight to its pixel; with values, also its ramp position (0..255).
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> sums: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> peak: array<atomic<u32>>;
@compute @workgroup_size(256)
fn accumulate(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let k = g.x + g.y * n.x * 256u; if (k >= V.ints.z) { return; }
  let p = place(k); if (!p.ok) { return; }
  let xy = pixel_of(p.t);
  if (xy.x < 0.0 || xy.y < 0.0 || xy.x >= V.pixel.x || xy.y >= V.pixel.y) { return; }
  let at = u32(xy.y) * u32(V.pixel.x) + u32(xy.x);
  atomicAdd(&counts[at], V.more.w);
  if (V.more.x == 2u) { atomicAdd(&sums[at], V.more.w * u32(clamp(ramp_t(p.index), 0.0, 1.0) * 255.0 + 0.5)); }
}
// The busiest pixel, for the tone scale: one atomic per workgroup of pixels.
var<workgroup> most: array<u32, 256>;
@compute @workgroup_size(256)
fn peak_pixel(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u, @builtin(local_invocation_index) lane: u32) {
  let i = g.x + g.y * n.x * 256u;
  var c = 0u; if (i < u32(V.pixel.x) * u32(V.pixel.y)) { c = atomicLoad(&counts[i]); }
  most[lane] = c;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s /= 2u) { if (lane < s) { most[lane] = max(most[lane], most[lane + s]); } workgroupBarrier(); }
  if (lane == 0u) { atomicMax(&peak[0], most[0]); }
}
@group(0) @binding(7) var<storage, read> dense: array<u32>;
@group(0) @binding(9) var<storage, read> dense_sums: array<u32>;
@group(0) @binding(10) var<storage, read> dense_peak: array<u32>;
@vertex fn screen_vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)); return vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
}
// Brightness and colour rise with the log of the count (relative to the busiest pixel); with
// values, colour is their mean along the ramp.
@fragment fn density_fs(@builtin(position) at: vec4f) -> @location(0) vec4f {
  let i = u32(at.y) * u32(V.pixel.x) + u32(at.x);
  let c = dense[i]; if (c == 0u) { discard; }
  let level = min(1.0, log2(1.0 + f32(c)) / log2(1.0 + f32(max(dense_peak[0], 1u))));
  var color = ramp(level);
  if (V.more.x == 2u) { color = ramp(f32(dense_sums[i]) / (255.0 * f32(c))); }
  return vec4f(mix(V.canvas.rgb, color, 0.35 + 0.65 * level), 1.0);
}

// Picking: the nearest drawn image within 12 CSS pixels of the click, then the lowest image at that distance.
@group(0) @binding(11) var<storage, read_write> picked: array<atomic<u32>>;
fn pick_key(k: u32) -> u32 {
  if (k >= V.ints.z) { return 0xffffffffu; }
  let p = place(k); if (!p.ok) { return 0xffffffffu; }
  let d = distance(pixel_of(p.t), V.pixel.zw);
  return select(0xffffffffu, u32(d * 64.0), d < 12.0 * V.canvas.w);
}
@compute @workgroup_size(256)
fn pick_distance(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let key = pick_key(g.x + g.y * n.x * 256u); if (key != 0xffffffffu) { atomicMin(&picked[0], key); }
}
@compute @workgroup_size(256)
fn pick_image(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let k = g.x + g.y * n.x * 256u; let key = pick_key(k);
  if (key != 0xffffffffu && key == atomicLoad(&picked[0])) { atomicMin(&picked[1], k * V.ints.w); }
}
`;

const MAX_GROUPS = 65535;
const flat = groups => groups <= MAX_GROUPS ? [Math.max(1, groups), 1] : [MAX_GROUPS, Math.ceil(groups / MAX_GROUPS)];
// A stride through index * order + operation images that is coprime with the order, so a subset
// is never locked to one symmetry operation.
export function strideFor(total, budget, order) {
  if (total <= budget) return 1;
  const gcd = (a, b) => b ? gcd(b, a % b) : a;
  let stride = Math.ceil(total / budget);
  while (order > 1 && gcd(stride, order) !== 1) stride++;
  return stride;
}

export class GpuView {
  static create(device, canvas) {
    const context = canvas.getContext('webgpu');
    return context ? new GpuView(device, canvas, context) : null;
  }
  constructor(device, canvas, context) {
    this.device = device; this.canvas = canvas; this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: this.format, alphaMode: 'premultiplied' });
    const module = device.createShaderModule({ code: WGSL });
    const usage = GPUBufferUsage;
    this.uniform = device.createBuffer({ size: 80 * 16, usage: usage.UNIFORM | usage.COPY_DST });
    this.data = new ArrayBuffer(80 * 16);
    this.f32 = new Float32Array(this.data); this.u32 = new Uint32Array(this.data);
    this.buffers = {}; this.sources = {};
    this.picked = device.createBuffer({ size: 8, usage: usage.STORAGE | usage.COPY_SRC | usage.COPY_DST });
    this.pickRead = device.createBuffer({ size: 8, usage: usage.MAP_READ | usage.COPY_DST });
    this.peak = device.createBuffer({ size: 16, usage: usage.STORAGE | usage.COPY_DST });
    const depth = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };
    const targets = [{ format: this.format }];
    const render = (vertex, fragment, depthStencil) => device.createRenderPipelineAsync({ layout: 'auto', vertex: { module, entryPoint: vertex }, fragment: { module, entryPoint: fragment, targets }, primitive: { topology: 'triangle-list' }, ...(depthStencil ? { depthStencil } : {}) });
    const compute = entry => device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: entry } });
    // One small module: its pipelines compile in parallel as the page opens.
    this.ready = Promise.all([render('points_vs', 'points_fs', depth), compute('accumulate'), compute('peak_pixel'), render('screen_vs', 'density_fs'), compute('pick_distance'), compute('pick_image')])
      .then(([points, accumulate, peakPixel, density, pickDistance, pickImage]) => { Object.assign(this, { points, accumulate, peakPixel, density, pickDistance, pickImage }); return true; });
    this.inFlight = false;
    this.frameMs = 0;
  }
  // Each array is uploaded when it is a different object from the last one given.
  upload(name, array) {
    if (this.sources[name] === array && this.buffers[name]) return;
    this.sources[name] = array;
    const bytes = Math.max(16, array?.byteLength || 0), size = Math.ceil(bytes / 16) * 16, held = this.buffers[name];
    if (!held || held.size < size || held.size > 4 * size + 4096) {
      held?.destroy();
      this.buffers[name] = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    if (array?.byteLength) this.device.queue.writeBuffer(this.buffers[name], 0, array.buffer, array.byteOffset, array.byteLength);
  }
  setData({ quats, ops, values }) {
    this.upload('quats', quats); this.upload('ops', ops); this.upload('values', values);
  }
  ensureTargets(width, height) {
    if (this.width === width && this.height === height) return;
    this.width = width; this.height = height;
    for (const b of [this.depth, this.counts, this.sums]) b?.destroy();
    this.depth = this.device.createTexture({ size: [width, height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.counts = this.device.createBuffer({ size: width * height * 4, usage });
    this.sums = this.device.createBuffer({ size: width * height * 4, usage });
  }
  writeUniform(v, images, stride, click = [0, 0]) {
    const f = this.f32, u = this.u32;
    f.fill(0);
    for (let r = 0; r < 3; r++) f.set(v.rotation[r], r * 4);
    f.set(v.ndc, 12);
    f.set([2 / this.width, 2 / this.height, v.radius, v.extent], 16);
    f.set([...v.canvas, v.dpr], 20); f.set(v.point, 24); f.set([v.cut ?? 0, 0, 0, 0], 28);
    f.set([v.lo, v.hi, v.clip, v.depthMix], 32);
    f.set([this.width, this.height, click[0], click[1]], 36);
    u.set([v.projection, v.order, images, stride], 40);
    u.set([v.colour, v.cutAxis, v.cutSign, stride], 44);
    v.ramp.forEach((c, i) => f.set(c, 48 + 4 * i));
    v.zones.forEach((c, i) => f.set(c, 80 + 4 * i));
    this.device.queue.writeBuffer(this.uniform, 0, this.data);
  }
  bind(pipeline, entries) {
    return this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
  }
  // One frame. v describes the view (see writeUniform); mode 'points' or 'density'; budget caps
  // the images drawn (a coprime stride through them, each weighted by the stride in a density).
  // Returns null while the previous frame is still on the GPU.
  draw(v, mode, { budget = Infinity } = {}) {
    if (!this.points || this.inFlight) return null;
    const width = this.canvas.width, height = this.canvas.height;
    this.ensureTargets(width, height);
    const total = v.count * v.order, stride = strideFor(total, budget, v.order), images = Math.ceil(total / stride);
    this.writeUniform(v, images, stride);
    const encoder = this.device.createCommandEncoder();
    const view = this.context.getCurrentTexture().createView();
    const B = this.buffers, clear = { view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] };
    if (mode === 'density') {
      encoder.clearBuffer(this.counts); encoder.clearBuffer(this.peak);
      if (v.colour === 2) encoder.clearBuffer(this.sums);
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.accumulate);
      pass.setBindGroup(0, this.bind(this.accumulate, [[0, this.uniform], [1, B.quats], [2, B.ops], [3, B.values], [4, this.counts], [5, this.sums]]));
      pass.dispatchWorkgroups(...flat(Math.ceil(images / 256)));
      pass.setPipeline(this.peakPixel);
      pass.setBindGroup(0, this.bind(this.peakPixel, [[0, this.uniform], [4, this.counts], [6, this.peak]]));
      pass.dispatchWorkgroups(...flat(Math.ceil(width * height / 256)));
      pass.end();
      const pixels = encoder.beginRenderPass({ colorAttachments: [clear] });
      pixels.setPipeline(this.density);
      pixels.setBindGroup(0, this.bind(this.density, [[0, this.uniform], [7, this.counts], [9, this.sums], [10, this.peak]]));
      pixels.draw(3); pixels.end();
    } else {
      const discs = encoder.beginRenderPass({ colorAttachments: [clear], depthStencilAttachment: { view: this.depth.createView(), depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'discard' } });
      discs.setPipeline(this.points);
      discs.setBindGroup(0, this.bind(this.points, [[0, this.uniform], [1, B.quats], [2, B.ops], [3, B.values]]));
      discs.draw(6, images); discs.end();
    }
    const started = performance.now();
    this.inFlight = true;
    this.device.queue.submit([encoder.finish()]);
    this.device.queue.onSubmittedWorkDone().then(() => { this.frameMs = performance.now() - started; this.inFlight = false; this.onIdle?.(); }, () => { this.inFlight = false; });
    return { images, stride };
  }
  clear() {
    const encoder = this.device.createCommandEncoder();
    encoder.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] }).end();
    this.device.queue.submit([encoder.finish()]);
  }
  // The image (index * order + operation) nearest a click (device pixels) among those drawn, or -1.
  async pick(v, click, { budget = Infinity } = {}) {
    await this.ready;
    this.ensureTargets(this.canvas.width, this.canvas.height);
    const total = v.count * v.order, stride = strideFor(total, budget, v.order), images = Math.ceil(total / stride);
    this.writeUniform(v, images, stride, click);
    this.device.queue.writeBuffer(this.picked, 0, new Uint32Array([0xffffffff, 0xffffffff]));
    const encoder = this.device.createCommandEncoder(), B = this.buffers;
    for (const pipeline of [this.pickDistance, this.pickImage]) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bind(pipeline, [[0, this.uniform], [1, B.quats], [2, B.ops], [11, this.picked]]));
      pass.dispatchWorkgroups(...flat(Math.ceil(images / 256)));
      pass.end();
    }
    encoder.copyBufferToBuffer(this.picked, 0, this.pickRead, 0, 8);
    this.device.queue.submit([encoder.finish()]);
    await this.pickRead.mapAsync(GPUMapMode.READ);
    const image = new Uint32Array(this.pickRead.getMappedRange().slice(0))[1];
    this.pickRead.unmap();
    return image === 0xffffffff ? -1 : image;
  }
}
