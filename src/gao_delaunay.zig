//! Source-guided Gao GPU constructor. No CPU triangulation or completion.
//! Default for certified regional covering; general hull construction is separate.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const primes = @import("dt_primes.zig");
pub const Result = struct { tets: gpu.Buf, count: u32, cold_calls: u64, rounds: u32, max_faces: u32 };
/// Same layout as GaoParams: sites whose original id is at least `seeds` get no star.
const Params = extern struct { n: u32, dim: u32, zero: u32 = 0, reserved: u32 = 0, lower: [4]f32, upper: [3]f32, seeds: u32, scale: [4]f32 };
const geometry = shaders.gao_storage ++ shaders.gao_topology ++ shaders.gao_geometry;
const main_code = "const GAO_POST=false;const GAO_MAX_P=32u;const GAO_MAX_T=64u;\n" ++ geometry ++ shaders.gao_main;
const numeric = shaders.dt_math ++ shaders.dt_double ++ shaders.dt_integer ++ shaders.gao_predicates ++ shaders.gao_double;
const numeric_header = "const POST_LANES=32u;const NONE=0xffffffffu;struct Cache{ids:vec4<u32>,more:vec4<u32>}\n";
const numeric_prefix = "const GAO_FILTERED_ONLY=false;const GAO_INTEGER_PRIMES=32u;" ++ numeric_header;
const filtered_prefix = "const GAO_FILTERED_ONLY=true;const GAO_INTEGER_PRIMES=32u;" ++ numeric_header;
const wide_prefix = "const GAO_FILTERED_ONLY=false;const GAO_INTEGER_PRIMES=96u;" ++ numeric_header;
const mark_code = "const GAO_POST=false;const GAO_MAX_P=32u;const GAO_MAX_T=64u;\n" ++ geometry ++ shaders.gao_mark;
const live_code = "const GAO_POST=true;const GAO_MAX_P=64u;const GAO_MAX_T=128u;\n" ++ numeric_prefix ++ geometry ++ numeric ++ shaders.gao_post ++ shaders.gao_live;
const filtered_post_code = "const GAO_POST=true;const GAO_MAX_P=64u;const GAO_MAX_T=128u;\n" ++ filtered_prefix ++ geometry ++ numeric ++ shaders.gao_post;
const post_code = "const GAO_POST=true;const GAO_MAX_P=64u;const GAO_MAX_T=128u;\n" ++ numeric_prefix ++ geometry ++ numeric ++ shaders.gao_post;
const output_code = "const GAO_POST=false;const GAO_MAX_P=64u;const GAO_MAX_T=128u;\n" ++ numeric_prefix ++ geometry ++ numeric ++ shaders.gao_output;
const wide_post_code = "const GAO_POST=true;const GAO_MAX_P=64u;const GAO_MAX_T=128u;\n" ++ wide_prefix ++ geometry ++ numeric ++ shaders.gao_post;
const wide_output_code = "const GAO_POST=false;const GAO_MAX_P=64u;const GAO_MAX_T=128u;\n" ++ wide_prefix ++ geometry ++ numeric ++ shaders.gao_output;
const grown_post_code = "const GAO_POST=true;const GAO_MAX_P=128u;const GAO_MAX_T=256u;\n" ++ wide_prefix ++ geometry ++ numeric ++ shaders.gao_post;
const grown_output_code = "const GAO_POST=false;const GAO_MAX_P=128u;const GAO_MAX_T=256u;\n" ++ wide_prefix ++ geometry ++ numeric ++ shaders.gao_output;
const maximum_prefix = "const GAO_FILTERED_ONLY=false;const GAO_INTEGER_PRIMES=96u;const POST_LANES=16u;const NONE=0xffffffffu;struct Cache{ids:vec4<u32>,more:vec4<u32>}\n";
const maximum_post_code = "const GAO_POST=true;const GAO_MAX_P=256u;const GAO_MAX_T=512u;\n" ++ maximum_prefix ++ geometry ++ numeric ++ shaders.gao_post;
const maximum_output_code = "const GAO_POST=false;const GAO_MAX_P=256u;const GAO_MAX_T=512u;\n" ++ maximum_prefix ++ geometry ++ numeric ++ shaders.gao_output;
fn check(g: *gpu.Gpu) !void {
    g.submit();
    try g.dev.wait();
    if (g.dev.takeError() != null) return error.GpuGaoValidation;
}
/// A Gao triangulation kept on the GPU: bucket-sorted coordinates, the grid, and every
/// site's star (its local mesh, owning the tetrahedra it outputs). `restar` recomputes
/// only the stars marked pending, with the same kernels and exact predicates, after the
/// caller has rewritten moved sites in `coordinates` (see covering.Live).
pub const Live = struct {
    n: u32 = 0,
    params: Params = undefined,
    cells: u32 = 0,
    coordinates: gpu.Buf = .{},
    offsets: gpu.Buf = .{},
    triangles: gpu.Buf = .{},
    indices: gpu.Buf = .{},
    states: gpu.Buf = .{},
    table: gpu.Buf = .{},
    counters: gpu.Buf = .{},
    records: gpu.Buf = .{},
    /// Slot in `coordinates` of each input site.
    slots: gpu.Buf = .{},
    /// Grid counting cursors (and, past the cells, the occupied-cell count).
    cursors: gpu.Buf = .{},
    /// Local mesh capacity per star (64, 128 or 256 vertices) and the 96-prime workspace.
    capacity: u32 = 64,
    wide: bool = false,
    pub fn release(self: *Live, g: *gpu.Gpu) void {
        for ([_]*gpu.Buf{ &self.coordinates, &self.offsets, &self.triangles, &self.indices, &self.states, &self.table, &self.counters, &self.records, &self.slots, &self.cursors }) |b| g.release(b);
        self.* = .{};
    }
    fn postPipe(self: *const Live, g: *gpu.Gpu) !u32 {
        return switch (self.capacity) {
            64 => if (self.wide) try g.pipeline("gao/post_wide", wide_post_code, "gao_post") else try g.pipeline("gao/post", post_code, "gao_post"),
            128 => try g.pipeline("gao/post_128", grown_post_code, "gao_post"),
            else => try g.pipeline("gao/post_256", maximum_post_code, "gao_post"),
        };
    }
    fn outputCode(self: *const Live) struct { key: []const u8, code: []const u8 } {
        return switch (self.capacity) {
            64 => if (self.wide) .{ .key = "wide", .code = wide_output_code } else .{ .key = "", .code = output_code },
            128 => .{ .key = "128", .code = grown_output_code },
            else => .{ .key = "256", .code = maximum_output_code },
        };
    }
    fn summaryPipe(self: *const Live, g: *gpu.Gpu) !u32 {
        const o = self.outputCode();
        return switch (self.capacity) {
            64 => if (self.wide) try g.pipeline("gao/summary_wide", o.code, "gao_summary") else try g.pipeline("gao/summary", o.code, "gao_summary"),
            128 => try g.pipeline("gao/summary_128", o.code, "gao_summary"),
            else => try g.pipeline("gao/summary_256", o.code, "gao_summary"),
        };
    }
    fn emitPipe(self: *const Live, g: *gpu.Gpu) !u32 {
        const o = self.outputCode();
        return switch (self.capacity) {
            64 => if (self.wide) try g.pipeline("gao/emit_wide", o.code, "gao_emit") else try g.pipeline("gao/emit", o.code, "gao_emit"),
            128 => try g.pipeline("gao/emit_128", o.code, "gao_emit"),
            else => try g.pipeline("gao/emit_256", o.code, "gao_emit"),
        };
    }
    fn postBinds(self: *const Live, g: *gpu.Gpu, params: *const Params) [9]gpu.Bind {
        return .{ g.uniform(0, std.mem.asBytes(params)), self.coordinates.at(1), self.offsets.at(2), self.triangles.at(3), self.indices.at(4), self.states.at(5), self.table.at(6), self.counters.at(7), self.records.at(8) };
    }
    fn summarize(self: *Live, g: *gpu.Gpu, params: *const Params) !void {
        g.dispatch(try self.summaryPipe(g), (self.n + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(params)), self.coordinates.at(1), self.triangles.at(3), self.indices.at(4), self.states.at(5), self.table.at(6), self.counters.at(7) });
    }
    /// Recomputes the stars marked GAO_PENDING in `states` (after the caller has
    /// rewritten the changed sites in `coordinates` and listed the grid again).
    pub fn restar(self: *Live, g: *gpu.Gpu, pending: ?[2]gpu.Buf, out: ?*[24]u32) !void {
        var params = self.params;
        params.scale[3] = 1;
        const main = try g.pipeline("gao/main", main_code, "gao_main");
        g.clear(self.counters);
        if (pending) |list| {
            // The list takes the records' binding (see shaders/gao_live.wgsl).
            var live_params = params;
            live_params.reserved &= ~@as(u32, 2);
            var binds = self.postBinds(g, &live_params);
            binds[8] = list[0].at(8);
            g.dispatchIndirect(try g.pipeline("gao/live_star", live_code, "gao_live_star"), list[1], 0, &binds);
        }
        g.dispatch(main, (self.n + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), self.coordinates.at(1), self.offsets.at(2), self.triangles.at(3), self.indices.at(4), self.states.at(5) });
        // Only stars the main pass left unfinished have a nonzero state: the post passes
        // touch nothing else.
        params.scale[3] = 0;
        if (self.capacity == 64 and !self.wide) g.dispatchFlat(try g.pipeline("gao/post_filtered", filtered_post_code, "gao_post"), self.n, 1, &self.postBinds(g, &params));
        g.dispatchFlat(try self.postPipe(g), self.n, 1, &self.postBinds(g, &params));
        if (out) |stats| {
            try self.summarize(g, &params);
            g.read(self.counters, 0, std.mem.asBytes(stats));
        }
    }
    /// Lists every live slot (w != NONE) under the cell of its current position, for a
    /// GAO_LISTED triangulation whose slots do not move.
    pub fn relist(self: *Live, g: *gpu.Gpu) !void {
        const u = g.uniform(0, std.mem.asBytes(&self.params));
        g.dispatch(try g.pipeline("gao/grid_clear", shaders.gao_grid, "gao_grid_clear"), (self.cells + 128) / 128, 1, 1, &.{ u, self.offsets.at(2) });
        g.dispatch(try g.pipeline("gao/list_count", shaders.gao_grid, "gao_list_count"), (self.n + 127) / 128, 1, 1, &.{ u, self.coordinates.at(1), self.offsets.at(2) });
        g.dispatch(try g.pipeline("gao/grid_prefix", shaders.gao_grid, "gao_grid_prefix"), 1, 1, 1, &.{ u, self.offsets.at(2), self.cursors.at(4) });
        g.dispatch(try g.pipeline("gao/list_scatter", shaders.gao_grid, "gao_list_scatter"), (self.n + 127) / 128, 1, 1, &.{ u, self.coordinates.at(1), self.offsets.at(2), self.cursors.at(4) });
    }
    /// Marks the stars a move's changes can alter (shaders/gao_mark.wgsl): `flags` holds
    /// the marks and reaches, `pending` receives the marked stars, `changes` is the move's
    /// scratch, and `launch` holds the indirect sizes (one workgroup per change, word 4).
    pub fn mark(self: *Live, g: *gpu.Gpu, flags: gpu.Buf, pending: gpu.Buf, changes: gpu.Buf, launch: gpu.Buf, reach_cap: f32) !void {
        var params = self.params;
        params.lower[3] = reach_cap;
        g.dispatchIndirect(try g.pipeline("gao/mark", mark_code, "gao_mark"), launch, 16, &.{ g.uniform(0, std.mem.asBytes(&params)), self.coordinates.at(1), self.offsets.at(2), self.triangles.at(3), self.indices.at(4), self.states.at(5), flags.at(6), pending.at(7), changes.at(8) });
    }
    /// Stores the reach of each star in `pending` (indirect sizes at word 0 of `launch`),
    /// or with `launch` null of every slot.
    pub fn reach(self: *Live, g: *gpu.Gpu, flags: gpu.Buf, pending: gpu.Buf, launch: ?gpu.Buf, reach_cap: f32) !void {
        var params = self.params;
        params.lower[3] = reach_cap;
        params.scale[3] = if (launch == null) 2 else 0;
        const pipe = try g.pipeline("gao/reach", mark_code, "gao_reach");
        const binds = [_]gpu.Bind{ g.uniform(0, std.mem.asBytes(&params)), self.coordinates.at(1), self.triangles.at(3), self.indices.at(4), self.states.at(5), flags.at(6), pending.at(7) };
        if (launch) |l| g.dispatchIndirect(pipe, l, 0, &binds) else g.dispatchFlat(pipe, self.n, 1, &binds);
    }
    /// The verdict on a restar's summary counters (once they have landed).
    pub fn restarResult(stats: *const [24]u32) !void {
        if (stats[2] & (1 << 8) != 0 or stats[2] == 1 << 1) return error.LiveRebuild;
        if (stats[1] != 0 or stats[2] != 0) return error.GpuGaoTopology;
    }
    /// Every seed's output tetrahedra (original site ids) into a new buffer of `count`.
    pub fn emit(self: *Live, g: *gpu.Gpu, count: u32) !gpu.Buf {
        if (count == 0) return error.LowerDimensionalDelaunay;
        const bytes = @as(u64, count) * 16;
        if (bytes > g.lim.max_storage_binding) return error.GpuDelaunayOutputTooLarge;
        var output = g.storage(bytes);
        errdefer g.release(&output);
        g.clear(self.counters);
        // No wait here: the caller's next readback surfaces any error in the emit.
        g.dispatch(try self.emitPipe(g), (self.n + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&self.params)), self.coordinates.at(1), self.triangles.at(3), self.indices.at(4), self.states.at(5), self.table.at(6), self.counters.at(7), output.at(8) });
        return output;
    }
};
pub const Built = struct { live: Live, stats: [24]u32, cold: u64, rounds: u32 };

/// Original ids [seeds, n) are non-seed sites: they take part in every other star but
/// get none of their own, so only tetrahedra owned by a seed are returned.
pub fn triangulate(g: *gpu.Gpu, input: gpu.Buf, n: u32, seeds: u32, lower: [3]f32, upper: [3]f32, regional_margin: ?f64) !Result {
    var built = try build(g, input, n, seeds, lower, upper, regional_margin, false);
    defer built.live.release(g);
    const output = try built.live.emit(g, built.stats[0]);
    return .{ .tets = output, .count = built.stats[0], .cold_calls = built.cold, .rounds = built.rounds, .max_faces = built.stats[3] };
}

/// Every star of a new triangulation, kept on the GPU (see Live); `emit` outputs it.
/// With `full`, every face of every star is searched, so whole stars are exact (live
/// updates need them); otherwise only the faces that decide this seed's own output are.
pub fn build(g: *gpu.Gpu, input: gpu.Buf, n: u32, seeds: u32, lower: [3]f32, upper: [3]f32, regional_margin: ?f64, full: bool) !Built {
    var live = try prepare(g, input, n, n, seeds, lower, upper, regional_margin, full);
    errdefer live.release(g);
    return solve(g, &live);
}

/// The sites of `input` bucket-sorted into slots [0, n) of a triangulation with room for
/// `capacity` slots (the rest left for the caller), guards after them; `solve` computes
/// the stars. Live covering builds adopt the sorted slots and list the grid in between.
pub fn prepare(g: *gpu.Gpu, input: gpu.Buf, n: u32, capacity: u32, seeds: u32, lower: [3]f32, upper: [3]f32, regional_margin: ?f64, full: bool) !Live {
    if (n < 4) return error.TooFewDelaunaySites;
    if (regional_margin) |margin| if (!(margin > 0 and margin < std.math.pi / 2.0 and std.math.isFinite(margin))) return error.InvalidCoveringMargin;
    if (@as(u64, capacity) * 128 * 4 > g.lim.max_storage_binding) return error.GpuGaoBatchTooLarge;
    const target = (g.delaunay_mode >> 16) & 15;
    const bucket_target: f64 = if (target == 1 or target == 2 or target == 4 or target == 8) @floatFromInt(target) else 4;
    // Sized from the occupied volume of the previous grid: a shell filling a sixth of
    // its box (the T zone) would otherwise crowd each occupied cell.
    const fill: f64 = std.math.clamp(g.gao_fill, 0.05, 1);
    const dim: u32 = @intFromFloat(@max(1.0, @min(@round(std.math.cbrt(@as(f64, @floatFromInt(n)) / (bucket_target * fill))), 1024)));
    const cells = dim * dim * dim;
    var width: f32 = 1e-5;
    var extent: f32 = 0;
    for (lower, upper) |x, y| {
        width = @max(width, y - x);
        extent = @max(extent, @max(@abs(x), @abs(y)));
    }
    const scale = @as(f32, @floatFromInt(dim)) / width;
    const params = Params{ .n = capacity, .dim = dim, .reserved = (g.delaunay_mode & 2) | @as(u32, if (full) 4 else 0), .lower = .{ lower[0], lower[1], lower[2], 0 }, .upper = upper, .seeds = seeds, .scale = .{ scale, scale, scale, 0 } };
    var live = Live{ .n = capacity, .params = params, .cells = cells };
    errdefer live.release(g);
    live.coordinates = g.storage((@as(u64, capacity) + 4) * 16);
    // Cell offsets, then (listed grids) each cell's slots.
    live.offsets = g.storage((@as(u64, cells) + 1 + capacity) * 4);
    live.cursors = g.storage((@as(u64, cells) + 1) * 4);
    live.triangles = g.storage(@as(u64, capacity) * 128 * 4);
    live.indices = g.storage(@as(u64, capacity) * 64 * 4);
    live.states = g.storage(@as(u64, capacity) * 16);
    live.table = g.storage(primes.table.len * 4);
    live.counters = g.storage(96);
    live.records = g.storage(64 * 32);
    live.slots = g.storage(@as(u64, n) * 4);
    g.writeSlice(live.table, 0, u32, &primes.table);
    // Source's large virtual tetrahedron is retained for general DT. Covering
    // only needs empty caps centred in the northern FZ with radius <= margin.
    // A guard at stereographic radius r has vector norm 2r/(1+r*r), so it is
    // outside every such cap when r > (1+sin(margin))/cos(margin). The source
    // tetrahedron's minimum guard radius is B; B>=8*extent also encloses the
    // entire input AABB (its tightest side inequality needs only B>=7*extent).
    const bound: f32 = if (regional_margin) |margin|
        @floatCast(@max(8 * @as(f64, extent), 1.25 * (1 + @sin(margin)) / @cos(margin)))
    else
        @max(1, extent) * 1e8;
    if (!std.math.isFinite(bound)) return error.DelaunayCoordinateRange;
    const guards = [4][4]f32{ .{ -bound, -bound, -bound, @bitCast(capacity) }, .{ bound, -bound, -bound, @bitCast(capacity + 1) }, .{ 0, bound, -bound, @bitCast(capacity + 2) }, .{ 0, 0, bound, @bitCast(capacity + 3) } };
    g.writeSlice(live.coordinates, @as(u64, capacity) * 16, [4]f32, &guards);
    const clear = try g.pipeline("gao/grid_clear", shaders.gao_grid, "gao_grid_clear");
    const count = try g.pipeline("gao/grid_count", shaders.gao_grid, "gao_grid_count");
    const prefix = try g.pipeline("gao/grid_prefix", shaders.gao_grid, "gao_grid_prefix");
    const scatter = try g.pipeline("gao/grid_scatter", shaders.gao_grid, "gao_grid_scatter");
    const order = try g.pipeline("gao/grid_order", shaders.gao_grid, "gao_grid_order");
    const slot_map = try g.pipeline("gao/grid_slots", shaders.gao_grid, "gao_grid_slots");
    // The sort itself covers the n input sites.
    var sort_params = params;
    sort_params.n = n;
    const u = g.uniform(0, std.mem.asBytes(&sort_params));
    g.dispatch(clear, (cells + 128) / 128, 1, 1, &.{ u, live.offsets.at(2) });
    g.dispatch(count, (n + 127) / 128, 1, 1, &.{ u, input.at(1), live.offsets.at(2) });
    g.dispatch(prefix, 1, 1, 1, &.{ u, live.offsets.at(2), live.cursors.at(4) });
    g.dispatch(scatter, (n + 127) / 128, 1, 1, &.{ u, input.at(1), live.coordinates.at(3), live.cursors.at(4) });
    g.dispatch(order, (cells + 127) / 128, 1, 1, &.{ u, live.offsets.at(2), live.coordinates.at(3) });
    g.dispatch(slot_map, (n + 127) / 128, 1, 1, &.{ u, live.coordinates.at(3), live.slots.at(5) });
    return live;
}

/// The stars of a prepared triangulation (every slot of `live`), its output summary.
pub fn solve(g: *gpu.Gpu, live: *Live) !Built {
    const n = live.n;
    const cells = live.cells;
    const params = live.params;
    const u = g.uniform(0, std.mem.asBytes(&params));
    const main = try g.pipeline("gao/main", main_code, "gao_main");
    g.dispatch(main, (n + 127) / 128, 1, 1, &.{ u, live.coordinates.at(1), live.offsets.at(2), live.triangles.at(3), live.indices.at(4), live.states.at(5) });
    if (@import("builtin").target.os.tag != .freestanding and g.delaunay_mode & 2 != 0) {
        const diagnostic = try std.heap.page_allocator.alloc([4]u32, n);
        defer std.heap.page_allocator.free(diagnostic);
        g.read(live.states, 0, std.mem.sliceAsBytes(diagnostic));
        try check(g);
        var counts: [5]u32 = @splat(0);
        var maximum: [3]u32 = @splat(0);
        for (diagnostic) |s| {
            if (s[0] < counts.len) counts[s[0]] += 1;
            for (0..3) |i| maximum[i] = @max(maximum[i], s[i + 1]);
        }
        std.debug.print("Gao main status={any}; max nv/nt/cycles={any}\n", .{ counts, maximum });
    }
    g.dispatchFlat(try g.pipeline("gao/post_filtered", filtered_post_code, "gao_post"), n, 1, &live.postBinds(g, &params));
    g.dispatchFlat(try live.postPipe(g), n, 1, &live.postBinds(g, &params));
    try live.summarize(g, &params);
    var stats: [24]u32 = undefined;
    var previous_cold: u64 = 0;
    var rounds: u32 = 3;
    var occupied: [4]u32 = undefined;
    g.read(live.counters, 0, std.mem.asBytes(&stats));
    g.read(live.cursors, @as(u64, cells) * 4, std.mem.asBytes(occupied[0..1]));
    try check(g);
    const occupied_fill = @as(f32, @floatFromInt(occupied[0])) / @as(f32, @floatFromInt(cells));
    // A compact CRT workspace covers ordinary orientation coordinates. Wider
    // dyadic ranges restart on the GPU with the full finite-binary32 modulus.
    // No mesh touched by an unresolved predicate is ever returned as output.
    if (stats[2] & (1 << 8) != 0) {
        previous_cold += stats[7];
        rounds += 2;
        live.wide = true;
        g.clear(live.counters);
        g.dispatch(main, (n + 127) / 128, 1, 1, &.{ u, live.coordinates.at(1), live.offsets.at(2), live.triangles.at(3), live.indices.at(4), live.states.at(5) });
        g.dispatchFlat(try live.postPipe(g), n, 1, &live.postBinds(g, &params));
        try live.summarize(g, &params);
        g.read(live.counters, 0, std.mem.asBytes(&stats));
        try check(g);
    }
    // Port the source's overflow handoff using larger packed local meshes.
    // Component-major storage makes capacity growth a contiguous GPU copy;
    // already-completed stars remain untouched and only failed stars resume.
    for ([_]u32{ 128, 256 }) |capacity| {
        if (stats[2] != 1 << 1 or stats[1] == 0) break;
        const triangle_bytes = @as(u64, n) * capacity * 2 * 4;
        if (triangle_bytes > g.lim.max_storage_binding) return error.GpuGaoCapacity;
        const bigger_triangles = g.storage(triangle_bytes);
        g.copy(live.triangles, 0, bigger_triangles, 0, live.triangles.size);
        g.release(&live.triangles);
        live.triangles = bigger_triangles;
        const bigger_indices = g.storage(@as(u64, n) * capacity * 4);
        g.copy(live.indices, 0, bigger_indices, 0, live.indices.size);
        g.release(&live.indices);
        live.indices = bigger_indices;
        live.capacity = capacity;
        previous_cold += stats[7];
        rounds += 1;
        g.clear(live.counters);
        g.dispatchFlat(try live.postPipe(g), n, 1, &live.postBinds(g, &params));
        try live.summarize(g, &params);
        g.read(live.counters, 0, std.mem.asBytes(&stats));
        try check(g);
    }
    if (@import("builtin").target.os.tag != .freestanding and g.delaunay_mode & 2 != 0) std.debug.print("Gao output stats={any}\n", .{stats});
    if (@import("builtin").target.os.tag != .freestanding and g.delaunay_mode & 2 != 0 and stats[20] != 0) {
        var samples: [4][8]u32 = undefined;
        const values = try std.heap.page_allocator.alloc([4]u32, n + 4);
        defer std.heap.page_allocator.free(values);
        g.read(live.records, 0, std.mem.asBytes(&samples));
        g.read(live.coordinates, 0, std.mem.sliceAsBytes(values));
        try check(g);
        for (samples[0..@min(4, stats[20])]) |sample| {
            var bits: [5][3]u32 = @splat(@splat(0));
            const rows: usize = if (sample[5] == 1) 5 else 4;
            for (0..rows) |i| bits[i] = values[sample[i]][0..3].*;
            std.debug.print("Gao zero kind={d} xyz_bits={any}\n", .{ sample[5], bits });
        }
    }
    if (stats[1] != 0 or stats[2] != 0) {
        if (@import("builtin").target.os.tag != .freestanding and g.delaunay_mode & 2 != 0) std.debug.print("Gao incomplete: {any}\n", .{stats});
        if (stats[2] == 1 << 1) return error.GpuGaoCapacity;
        return error.GpuGaoTopology;
    }
    if (stats[0] == 0) return error.LowerDimensionalDelaunay;
    // Degenerate (lattice-like, co-spherical) inputs need many exact predicates, and a
    // finer grid sends more of their sites to the exact post pass: keep box sizing
    // for them. Generic samples size the next grid from this one's occupied cells.
    g.gao_degenerate = previous_cold + stats[7] > n / 256;
    g.gao_fill = if (g.gao_degenerate) 1 else occupied_fill;
    return .{ .live = live.*, .stats = stats, .cold = previous_cold + stats[7], .rounds = rounds };
}
