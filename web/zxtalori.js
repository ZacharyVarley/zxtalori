/** WebGPU bridge adapted from zcmir (MIT, copyright 2026 Zachary Varley).
 * Zig owns all scientific computation; this adapter manages browser GPU resources.
 */
const utf8 = new TextDecoder();
const utf8enc = new TextEncoder();
const newStats = () => ({dispatches:0,submits:0,reads:0,waits:0,waitMs:0,compileWaitMs:0,pipelines:0,pipelineMs:0});
// The engine build's identity: FNV-1a over the module's bytes, sampled with a stride.
function buildId(binary) { const b=new Uint8Array(binary); let h=0x811c9dc5^b.length; for(let i=0;i<b.length;i+=7) h=Math.imul(h^b[i],0x01000193); return (h>>>0).toString(36); }
function hashCode(s) { let h=0x811c9dc5; for(let i=0;i<s.length;i++) h=Math.imul(h^s.charCodeAt(i),0x01000193); return (h>>>0).toString(36); }
// Warm start: the kernels a session compiles, in first-use order, are kept in localStorage for
// this exact engine build (about 0.5 MB of WGSL). The next session starts compiling them as
// soon as the device exists, on the browser's background compile threads, so features used
// before (covering optimization's largest kernels take seconds on Direct3D) are ready when
// asked for. A different build starts a new list. A first visit starts from the list shipped
// with the site (`pipelines.json` beside the wasm, recorded by tests/test_web.mjs
// --save-pipelines) when it belongs to the same build.
const WARM_KEY = 'zxtalori-warm-pipelines';
const usable = (s, build) => s?.build === build && Array.isArray(s.list) && s.codes ? s : null;
function ownWarm() { try { return JSON.parse(localStorage.getItem(WARM_KEY) || 'null'); } catch { return null; } }
async function shippedWarm(url) { try { const r = await fetch(url); return r.ok ? await r.json() : null; } catch { return null; } }
function warmRecord(w) {
  const codes = {};
  for (const [h] of w.list) codes[h] = w.codes.get(h);
  return { build: w.build, list: w.list, codes };
}
function saveWarm(state) {
  const w = state.warm;
  if (!w || w.timer) return;
  w.timer = setTimeout(() => {
    w.timer = 0;
    try { localStorage.setItem(WARM_KEY, JSON.stringify(warmRecord(w))); } catch {}
  }, 2000);
}
function gpuImports(state) {
  const { device } = state;
  const bufs = [null];
  const freeIds = [];
  // pipelines: { pipeline, layout } once compiled (pipeline null while compiling or failed)
  const pipes = [null];
  const modules = new Map();
  // (code hash, entry) → compile promise, shared by the warm-up and createPipeline
  const compiled = new Map();
  let enc = null;
  let pass = null;
  let recorded = [];
  // Pipelines compile asynchronously and in parallel (createComputePipelineAsync). While a
  // pipeline that recorded work uses is still compiling, queue operations are held back in
  // order (`backlog`) and replayed once it is ready, so the module keeps recording ahead and
  // only a wait (a readback) blocks on the compiler.
  let backlog = null;
  let blocking = new Set();
  let drained = null;
  const mem = () => state.memory.buffer;
  const bytes = (ptr, len) => new Uint8Array(mem(), ptr, len);
  const endPass = () => { if (pass) { pass.end(); pass = null; } };
  const encoder = () => (enc ||= device.createCommandEncoder());
  // counters for zc.stats(): what an operation cost in GPU round trips
  const st = state.stats;
  device.addEventListener?.("uncapturederror", (e) => state.errors.push(String(e.error?.message || e.error)));

  const defer = (op) => {
    if (!backlog) {
      backlog = [];
      let resolve;
      drained = { promise: new Promise((r) => (resolve = r)), resolve };
    }
    backlog.push(op);
  };
  const replay = () => {
    const ops = backlog;
    backlog = null;
    for (const op of ops) op();
    drained.resolve();
    drained = null;
  };

  // ── operations, issued directly (or replayed from the backlog) ──
  // indirect: [buffer, byte offset] of the workgroup counts (3 × u32) instead of x, y, z
  // Profiling: each dispatch gets its own pass with begin/end timestamps, resolved at submit.
  const prof = state.profile;
  const doDispatch = (p, x, y, z, entries, indirect) => {
    if (!p.pipeline) return; // failed to compile: reported through takeError
    if (prof.on && prof.used < prof.capacity) {
      endPass();
      pass = encoder().beginComputePass({ timestampWrites: { querySet: prof.queries, beginningOfPassWriteIndex: 2 * prof.used, endOfPassWriteIndex: 2 * prof.used + 1 } });
      prof.names.push(p.name);
      prof.used++;
    }
    if (!pass) pass = encoder().beginComputePass();
    pass.setPipeline(p.pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: p.layout, entries }));
    if (indirect) pass.dispatchWorkgroupsIndirect(indirect[0], indirect[1]);
    else pass.dispatchWorkgroups(Math.max(1, x), Math.max(1, y), Math.max(1, z));
    if (prof.on && prof.names.length === prof.used && pass) endPass();
  };
  const bindEntries = (bindsPtr, n) => {
    const v = new DataView(mem(), bindsPtr, n * 24);
    const entries = [];
    for (let i = 0; i < n; i++) {
      const o = i * 24;
      const size = Number(v.getBigUint64(o + 16, true));
      const resource = { buffer: bufs[v.getUint32(o + 4, true)], offset: Number(v.getBigUint64(o + 8, true)) };
      if (size) resource.size = size;
      entries.push({ binding: v.getUint32(o, true), resource });
    }
    return entries;
  };
  const doSubmit = () => {
    endPass();
    let timing = null;
    if (prof.used && enc) {
      const count = 2 * prof.used, names = prof.names;
      const resolved = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const staging = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.resolveQuerySet(prof.queries, 0, count, resolved, 0);
      enc.copyBufferToBuffer(resolved, 0, staging, 0, count * 8);
      timing = { resolved, staging, names };
      prof.used = 0; prof.names = [];
    }
    if (enc) { st.submits++; device.queue.submit([enc.finish()]); enc = null; }
    if (timing) prof.pending.push(timing.staging.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigUint64Array(timing.staging.getMappedRange());
      timing.names.forEach((name, i) => {
        const e = prof.totals.get(name) || { ms: 0, count: 0 };
        e.ms += Number(t[2 * i + 1] - t[2 * i]) / 1e6; e.count++; prof.totals.set(name, e);
      });
      timing.staging.unmap(); timing.staging.destroy(); timing.resolved.destroy();
    }));
    for (const b of recorded) b.destroy();
    recorded = [];
    for (const r of state.reads) {
      if (r.promise || !r.issued) continue;
      r.promise = r.staging.mapAsync(GPUMapMode.READ).then(() => {
        // wasm memory may have grown since the read was requested: take a fresh view.
        new Uint8Array(mem(), r.dst, r.size).set(new Uint8Array(r.staging.getMappedRange(), 0, r.size));
        r.staging.unmap();
        r.staging.destroy();
      });
    }
  };
  const doDestroy = (b) => { if (enc) recorded.push(b); else b.destroy(); }; // never under an open batch

  const moduleOf = (code) => {
    let module = modules.get(code);
    if (!module) { module = device.createShaderModule({ code }); modules.set(code, module); }
    return module;
  };
  // Compile progress for onCompile listeners: counts restart once everything started has finished.
  const cp = state.compiling;
  const told = () => { for (const fn of cp.listeners) try { fn({ done: cp.done, total: cp.total }); } catch (e) { console.error(e); } };
  // Warm-start compiles run in the background: they stay out of the progress counts.
  const compile = (code, h, entryPoint, background = false) => {
    const key = h + "/" + entryPoint;
    let c = compiled.get(key);
    if (!c) {
      c = device.createComputePipelineAsync({ label: entryPoint, layout: "auto", compute: { module: moduleOf(code), entryPoint } });
      if (!background) {
        if (cp.done === cp.total) cp.done = cp.total = 0;
        cp.total++;
        told();
        c.finally(() => { cp.done++; told(); }).catch(() => {});
      }
      compiled.set(key, c);
    }
    return c;
  };
  // Start compiling what this build used before (Zxtalori.create). An entry that fails to compile
  // keeps its error inside this error scope (not reported against a later call) and leaves
  // the list.
  const w = state.warm;
  if (w?.saved) {
    w.list = w.saved.list.slice();
    device.pushErrorScope("validation");
    for (const [h, entry] of w.saved.list) {
      const code = w.saved.codes[h];
      if (!code) continue;
      w.codes.set(h, code);
      compile(code, h, entry, true).catch(() => {
        const i = w.list.findIndex(([a, b]) => a === h && b === entry);
        if (i >= 0) w.list.splice(i, 1);
        saveWarm(state);
      });
    }
    device.popErrorScope().catch(() => {});
    w.saved = null;
  }

  return {
    limits(out) {
      const v = new DataView(mem(), out, 40);
      const L = device.limits;
      v.setBigUint64(0, BigInt(L.maxStorageBufferBindingSize), true);
      v.setBigUint64(8, BigInt(L.maxBufferSize), true);
      v.setUint32(16, L.maxComputeWorkgroupStorageSize, true);
      v.setUint32(20, L.maxStorageBuffersPerShaderStage, true);
      v.setUint32(24, L.minUniformBufferOffsetAlignment, true);
      v.setUint32(28, L.minStorageBufferOffsetAlignment, true);
      v.setUint32(32, device.features.has("shader-f16") ? 1 : 0, true);
      v.setUint32(36, navigator.gpu?.wgslLanguageFeatures?.has?.("packed_4x8_integer_dot_product") ? 1 : 0, true);
    },
    info(ptr, cap) {
      const b = utf8enc.encode(state.info).subarray(0, cap);
      bytes(ptr, b.length).set(b);
      return b.length;
    },
    createBuffer(size, usage) {
      const u = usage === 1
        ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT;
      const b = device.createBuffer({ size, usage: u });
      const id = freeIds.length ? freeIds.pop() : bufs.length;
      bufs[id] = b;
      return id;
    },
    destroyBuffer(id) {
      const b = bufs[id];
      if (!b) return;
      bufs[id] = null;
      freeIds.push(id);
      if (backlog) defer(() => doDestroy(b)); else doDestroy(b);
    },
    writeBuffer(id, offset, ptr, len) {
      const b = bufs[id];
      if (backlog) { const data = bytes(ptr, len).slice(); defer(() => device.queue.writeBuffer(b, offset, data)); }
      else device.queue.writeBuffer(b, offset, mem(), ptr, len);
    },
    // Starts compiling and returns at once; the first dispatch that needs it holds the queue
    // until it is ready.
    createPipeline(codePtr, codeLen, entryPtr, entryLen) {
      const t0 = performance.now();
      st.pipelines++;
      const code = utf8.decode(bytes(codePtr, codeLen));
      const entryPoint = utf8.decode(bytes(entryPtr, entryLen));
      const h = hashCode(code);
      if (w && !w.list.some(([a, b]) => a === h && b === entryPoint)) {
        w.codes.set(h, code);
        w.list.push([h, entryPoint]); // first use first
        saveWarm(state);
      }
      const p = { pipeline: null, layout: null, ready: false, name: entryPoint };
      p.promise = compile(code, h, entryPoint)
        .then((pl) => { p.pipeline = pl; p.layout = pl.getBindGroupLayout(0); })
        .catch((e) => state.errors.push(`pipeline ${entryPoint}: ${e.message || e}`))
        .finally(() => {
          p.ready = true;
          if (blocking.delete(p) && backlog && blocking.size === 0) replay();
        });
      pipes.push(p);
      st.pipelineMs += performance.now() - t0;
      return pipes.length - 1;
    },
    dispatch(pipe, x, y, z, bindsPtr, n) {
      st.dispatches++;
      const p = pipes[pipe];
      const entries = bindEntries(bindsPtr, n);
      if (!p.ready) blocking.add(p);
      if (backlog || !p.ready) defer(() => doDispatch(p, x, y, z, entries));
      else doDispatch(p, x, y, z, entries);
    },
    dispatchIndirect(pipe, buf, offset, bindsPtr, n) {
      st.dispatches++;
      const p = pipes[pipe];
      const entries = bindEntries(bindsPtr, n);
      const ind = [bufs[buf], offset];
      if (!p.ready) blocking.add(p);
      if (backlog || !p.ready) defer(() => doDispatch(p, 0, 0, 0, entries, ind));
      else doDispatch(p, 0, 0, 0, entries, ind);
    },
    clearBuffer(id, offset, size) {
      const b = bufs[id];
      const op = () => { endPass(); encoder().clearBuffer(b, offset, size); };
      if (backlog) defer(op); else op();
    },
    copyBuffer(src, soff, dst, doff, size) {
      const a = bufs[src];
      const b = bufs[dst];
      const op = () => { endPass(); encoder().copyBufferToBuffer(a, soff, b, doff, size); };
      if (backlog) defer(op); else op();
    },
    readBuffer(src, offset, size, dst) {
      st.reads++;
      const a = bufs[src];
      const padded = Math.ceil(size / 4) * 4;
      const staging = device.createBuffer({ size: padded, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const r = { staging, size, dst, promise: null, issued: false };
      state.reads.push(r);
      const op = () => { endPass(); encoder().copyBufferToBuffer(a, offset, staging, 0, padded); r.issued = true; };
      if (backlog) defer(op); else op();
    },
    submit() {
      if (backlog) defer(doSubmit); else doSubmit();
    },
    takeError(ptr, cap) {
      if (!state.errors.length) return 0;
      state.lastGpuError=state.errors.shift();
      const b = utf8enc.encode(state.lastGpuError).subarray(0, cap);
      bytes(ptr, b.length).set(b);
      return b.length;
    },
    // Suspending import (JS Promise Integration): the module's stack waits here while pipelines
    // still compiling and the submitted reads resolve, then resumes as if the call had blocked.
    wait: new WebAssembly.Suspending(async () => {
      const t0 = performance.now();
      st.waits++;
      try {
        if (drained) { await drained.promise; st.compileWaitMs += performance.now() - t0; }
        const reads = state.reads;
        state.reads = [];
        await Promise.all(reads.map((r) => r.promise));
        return 0;
      } catch (e) { state.errors.push(String(e)); return 1; } finally { st.waitMs += performance.now() - t0; }
    }),
  };
}

/** Browser interface to the same Zig/WGSL engine used by Python. */
export class Zxtalori {
  static async create({wasm = new URL('./zxtalori.wasm', import.meta.url), onCompile = null} = {}) {
    if (!isSecureContext) throw new Error('WebGPU needs HTTPS or localhost.');
    if (typeof WebAssembly.Suspending !== 'function' || typeof WebAssembly.promising !== 'function') {
      throw new Error('This browser needs WebAssembly Promise Integration. Use a current Chrome or Edge.');
    }
    const adapter = await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});
    if (!adapter) throw new Error('WebGPU is unavailable. Enable hardware acceleration or use a current Chrome or Edge.');
    const requiredLimits = {};
    for (const key of ['maxStorageBufferBindingSize','maxBufferSize','maxComputeWorkgroupStorageSize','maxStorageBuffersPerShaderStage']) requiredLimits[key] = adapter.limits[key];
    // timestamp-query, when offered, only serves development profiling (profile()).
    const requiredFeatures = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
    const device = await adapter.requestDevice({requiredLimits, requiredFeatures});
    const ai = adapter.info || {};
    const info = `${ai.vendor || 'GPU'} ${ai.architecture || ai.device || ''}`.trim();
    const profile = {on:false, used:0, names:[], totals:new Map(), pending:[], capacity:2048, queries:requiredFeatures.length?device.createQuerySet({type:'timestamp', count:4096}):null};
    const state = {device, memory:null, reads:[], errors:[], info, stats:newStats(), profile, compiling:{done:0,total:0,listeners:onCompile?[onCompile]:[]}};
    // This origin's own warm-start list, else (first visit) the one shipped beside the wasm.
    const own = ownWarm();
    const shippedUrl = new URL('pipelines.json', new URL(wasm, location.href));
    const shipped = own ? null : shippedWarm(shippedUrl);
    const response = await fetch(wasm);
    if (!response.ok) { device.destroy(); throw new Error('The engine build is missing. Build the WebAssembly target, then reload.'); }
    const log = (ptr,len) => console.warn('zxtalori:', utf8.decode(new Uint8Array(state.memory.buffer,ptr,len)));
    const env = {zo_log:log,zc_log:log,zo_event(){},zc_event(){},zo_now:()=>performance.now()};
    const binary = await response.arrayBuffer();
    const build = buildId(binary);
    const saved = usable(own, build) || usable(await (shipped || shippedWarm(shippedUrl)), build);
    state.warm = { build, saved, list: [], codes: new Map(), timer: 0 };
    const {instance} = await WebAssembly.instantiate(binary, {zcgpu:gpuImports(state),env});
    state.memory = instance.exports.memory;
    const sync = new Set(['zo_create','zo_destroy','zo_alloc','zo_free','zo_count','zo_quaternions','zo_stats','zo_error','zo_cancel','zo_operations','zo_order','zo_version','zo_mesh_vertices','zo_mesh_tets','zo_mesh_vertex_count','zo_mesh_tet_count','zo_point_values']);
    const x = {};
    for (const [name,value] of Object.entries(instance.exports)) x[name] = typeof value === 'function' && !sync.has(name) ? WebAssembly.promising(value) : value;
    const engine = new Zxtalori(x,state);
    device.lost.then(({message}) => {state.errors.push(`GPU connection lost: ${message || 'reload the page to reconnect'}`);});
    return engine;
  }
  constructor(x,state) {
    this.x=x; this.state=state; this.queue=Promise.resolve();
    this.handle=x.zo_create();
    if(!this.handle) throw new Error('The orientation engine could not initialize.');
  }
  get info() {return this.state.info;}
  /** The pipelines this page compiled, as the warm-start list (pipelines.json) records them. */
  warmList() {return this.state.warm ? warmRecord(this.state.warm) : null;}
  /** Development profiling: per-kernel GPU time from timestamp queries (when the adapter
   * offers them). profile(true) starts and clears, profile(false) stops; report() sums. */
  profile(on) {const p=this.state.profile;if(!p.queries)return false;p.on=on;if(on){p.totals.clear();p.pending=[];}return true;}
  async profileReport() {const p=this.state.profile;await Promise.all(p.pending);return [...p.totals].map(([name,{ms,count}])=>({name,ms:+ms.toFixed(3),count})).sort((a,b)=>b.ms-a.ms);}
  get memory() {return this.state.memory.buffer;}
  text(ptr) {
    if(!ptr) return '';
    const bytes=new Uint8Array(this.memory); let end=ptr;
    while(end<bytes.length && end<ptr+1048576 && bytes[end]!==0) end++;
    return utf8.decode(bytes.subarray(ptr,end));
  }
  error() {return this.text(this.x.zo_error(this.handle)) || this.state.errors.shift() || 'The engine could not complete this operation.';}
  stats() {
    const text=this.text(this.x.zo_stats(this.handle));
    return text ? JSON.parse(text) : {};
  }
  snapshot() {
    const count=this.x.zo_count(this.handle);
    const ptr=this.x.zo_quaternions(this.handle);
    const order=this.x.zo_order?.(this.handle) || 0;
    const opPtr=this.x.zo_operations?.(this.handle) || 0;
    // Per-point display values (null until the pass that makes them; stats give their iterations).
    const values=kind=>{const at=count&&this.x.zo_point_values?.(this.handle,kind);return at?new Float32Array(this.memory,at,count).slice():null;};
    return {quaternions:count&&ptr?new Float32Array(this.memory,ptr,count*4).slice():new Float32Array(),operations:order&&opPtr?new Float32Array(this.memory,opPtr,order*4).slice():new Float32Array(),points:{separation:values(0),covering:values(1),faces:values(2)},stats:this.stats()};
  }
  enqueue(operation) {
    const run=this.queue.then(operation);
    this.queue=run.catch(()=>{});
    return run;
  }
  async call(name,...args) {
    const result=await this.x[name](this.handle,...args);
    if(result!==0) throw new Error(this.error());
    if(this.state.errors.length) throw new Error(this.state.errors.shift());
    return this.snapshot();
  }
  generate(group,h,target=2048,seed=193) {return this.enqueue(()=>this.call('zo_generate',group,h,target,seed));}
  random(group,count,seed) {return this.enqueue(()=>this.call('zo_random',group,count,seed));}
  mesh() {return this.enqueue(async()=>{await this.call('zo_mesh');const n=this.x.zo_mesh_vertex_count(this.handle),t=this.x.zo_mesh_tet_count(this.handle);return {vertices:new Float32Array(this.memory,this.x.zo_mesh_vertices(this.handle),n*4).slice(),tets:new Uint32Array(this.memory,this.x.zo_mesh_tets(this.handle),t*4).slice()};});}
  generateExact(group,count,{identity=false}={}) {return this.enqueue(()=>this.call('zo_generate_exact',group,count,+identity));}
  measure() {return this.enqueue(()=>this.call('zo_measure'));}
  covering() {return this.enqueue(()=>this.call('zo_cover'));}
  relax(steps=1) {return this.enqueue(()=>this.call('zo_relax',steps));}
  // Forces from orbit points within delta mean orientation radii (Quey's near field), 0 for all pairs.
  relaxNeighbourhood(delta) {return this.enqueue(()=>this.call('zo_relax_neighbourhood',delta));}
  // mode 0 frees the offending tet, 1 also its face neighbours, 2 its one-ring;
  // finish leaves the best sample of the run in place; floorDeg is the minimum
  // separation local steps keep (0 for none).
  optimizeCovering(steps,mode=1,finish=false,floorDeg=0) {return this.enqueue(()=>this.call('zo_cover_optimize',steps,mode,+finish,floorDeg));}
  identity({add=false,fixed=false}={}) {return this.enqueue(()=>this.call('zo_identity',(+add)|(+fixed<<1)));}
  set(group,quaternions) {
    if(quaternions.length%4 || !quaternions.length) throw new Error('A sample needs rows of four quaternion values.');
    return this.enqueue(async()=>{
      const bytes=quaternions.length*4,ptr=this.x.zo_alloc(bytes);
      if(!ptr) throw new Error('The engine could not allocate the imported sample.');
      try {new Float32Array(this.memory,ptr,quaternions.length).set(quaternions);return await this.call('zo_set',group,ptr,quaternions.length/4);}
      finally {this.x.zo_free(ptr,bytes);}
    });
  }
  cancel() {this.x.zo_cancel?.(this.handle);}
  destroy() {this.x.zo_destroy(this.handle);this.state.device.destroy();}
}

