//! GPU-private Delaunay stars, exact cold signs and checked completion.
//! CPU code only schedules kernels, grows capacity and transfers requested output.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const primes = @import("dt_primes.zig");
pub const Point = [3]f32;
pub const Tet = [4]u32;
pub const Result = struct { tets: gpu.Buf, count: u32, cold_calls: u64, rounds: u32, max_faces: u32 };
const Params = extern struct { n: u32, seeds: u32, first: u32, cap: u32, dim: u32, zero: u32 = 0, budget: u32 = 256, pad: u32 = 0, lower: [4]f32, scale: [4]f32 };
const base_code = shaders.dt_common ++ shaders.dt_math ++ shaders.dt_double ++ shaders.dt_interval ++ shaders.dt_scratch ++ shaders.dt_prepared ++ shaders.dt_star ++ shaders.dt_integer ++ shaders.dt_exact ++ shaders.dt_cooperative ++ shaders.dt_deferred;
const code = "const POST_LANES=128u; const FAST_PASS=false; const POST_INLINE=false; const SHARED_FACES=false; const FACE_CAP=128u; const CACHED_FACES=false;\n" ++ base_code;
const fast_code = "const POST_LANES=128u; const FAST_PASS=true; const POST_INLINE=false; const SHARED_FACES=false; const FACE_CAP=128u; const CACHED_FACES=false;\n" ++ base_code;
const post_code = "const POST_LANES=32u; const FAST_PASS=false; const POST_INLINE=false; const SHARED_FACES=true; const FACE_CAP=128u; const CACHED_FACES=false;\n" ++ base_code;
const deferred_code = "const POST_LANES=128u; const FAST_PASS=false; const POST_INLINE=true; const SHARED_FACES=true; const FACE_CAP=128u; const CACHED_FACES=false;\n" ++ base_code;
pub const Index = struct { heads: gpu.Buf, next: gpu.Buf, dim: u32, scale: f32 };
pub fn makeIndex(g: *gpu.Gpu, coordinates: gpu.Buf, n: u32, lower: [3]f32, upper: [3]f32) !Index {
    const dim: u32 = @intFromFloat(@max(1.0, @round(std.math.cbrt(@as(f64, @floatFromInt(n)) / 4))));
    var width: f32 = 1e-5;
    for (lower, upper) |x, y| width = @max(width, y - x);
    width *= 1.00001;
    var heads = g.storage(@as(u64, dim) * dim * dim * 8);
    errdefer g.release(&heads);
    var next = g.storage(@as(u64, n) * 4);
    errdefer g.release(&next);
    const params = Params{ .n = n, .seeds = 0, .first = 0, .cap = 0, .dim = dim, .lower = .{ lower[0], lower[1], lower[2], 0 }, .scale = .{ @as(f32, @floatFromInt(dim)) / width, 0, 0, 0 } };
    const clear = try g.pipeline("dt/grid_clear", code, "grid_clear");
    const build = try g.pipeline("dt/grid_build", code, "grid_build");
    g.dispatch(clear, (dim * dim * dim + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), heads.at(2) });
    g.dispatch(build, (n + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), coordinates.at(1), heads.at(2), next.at(3) });
    return .{ .heads = heads, .next = next, .dim = dim, .scale = @as(f32, @floatFromInt(dim)) / width };
}
fn check(g: *gpu.Gpu) !void {
    g.submit();
    try g.dev.wait();
    if (g.dev.takeError() != null) return error.GpuDelaunayValidation;
}
pub fn triangulate(a: std.mem.Allocator, g: *gpu.Gpu, coordinates: gpu.Buf, n: u32, lower: [3]f32, upper: [3]f32) !Result {
    if (g.delaunay_mode & 32 != 0) {
        const result = try @import("gao_delaunay.zig").triangulate(g, coordinates, n, n, lower, upper, null);
        return .{ .tets = result.tets, .count = result.count, .cold_calls = result.cold_calls, .rounds = result.rounds, .max_faces = result.max_faces };
    }
    if (n < 4) return error.TooFewDelaunaySites;
    var cap: u32 = 128;
    var extent: f32 = 1;
    for (lower, upper) |x, y| extent = @max(extent, @max(@abs(x), @abs(y)));
    var enclosing: f32 = if (g.delaunay_mode & 4 != 0) 4 else 0x1p20;
    for (0..7) |_| {
        const result = attempt(a, g, coordinates, n, lower, upper, extent * enclosing, cap) catch |err| switch (err) {
            error.GpuDelaunayCapacity => {
                if (cap >= 1024) return err;
                cap *= 2;
                continue;
            },
            error.GpuDelaunayHull => {
                if (enclosing >= 0x1p80) return err;
                enclosing *= 0x1p20;
                continue;
            },
            else => return err,
        };
        return result;
    }
    return error.GpuDelaunayIncomplete;
}
/// The certified FZ covering domain permits bounded virtual points without
/// changing any cap that can determine its covering radius.
/// Sites [seeds, n) are fence sites (see covering.zig): Gao skips their stars; the
/// fallback constructor triangulates them like any other site.
pub fn triangulateCovering(a: std.mem.Allocator, g: *gpu.Gpu, coordinates: gpu.Buf, n: u32, seeds: u32, lower: [3]f32, upper: [3]f32, margin: f64) !Result {
    if (g.delaunay_mode & (1 | 4 | 16 | 128) == 0 or g.delaunay_mode & 32 != 0) {
        const result = try @import("gao_delaunay.zig").triangulate(g, coordinates, n, seeds, lower, upper, margin);
        return .{ .tets = result.tets, .count = result.count, .cold_calls = result.cold_calls, .rounds = result.rounds, .max_faces = result.max_faces };
    }
    return triangulate(a, g, coordinates, n, lower, upper);
}
fn attempt(a: std.mem.Allocator, g: *gpu.Gpu, coordinates: gpu.Buf, n: u32, lower: [3]f32, upper: [3]f32, bound: f32, cap: u32) !Result {
    if (!std.math.isFinite(bound)) return error.DelaunayCoordinateRange;
    const dim: u32 = @intFromFloat(@max(1.0, @round(std.math.cbrt(@as(f64, @floatFromInt(n)) / 4))));
    var width: f32 = 1e-5;
    for (lower, upper) |x, y| width = @max(width, y - x);
    width *= 1.00001;
    const batch_limit: u32 = @intCast(@max(64, @min(16384, @min(128 * 1024 * 1024, g.lim.max_storage_binding) / (@as(u64, cap) * 48))));
    const batch: u32 = @min(n, batch_limit);
    var heads = g.storage(@as(u64, dim) * dim * dim * 8);
    defer g.release(&heads);
    var next = g.storage((@as(u64, n) + 2 * @as(u64, dim) * dim * dim) * 4);
    defer g.release(&next);
    var ordering = g.storage(@as(u64, n) * 4);
    defer g.release(&ordering);
    var faces = g.storage(@as(u64, batch) * cap * 48);
    defer g.release(&faces);
    var states = g.storage(@as(u64, batch) * 112);
    defer g.release(&states);
    var cache = g.storage(@as(u64, batch) * 8 * 32);
    defer g.release(&cache);
    var scratch = g.storage(@as(u64, batch) * (32 + cap * 2) * 4);
    defer g.release(&scratch);
    var table = g.storage(primes.table.len * 4);
    defer g.release(&table);
    g.writeSlice(table, 0, u32, &primes.table);
    var counters = g.storage(32);
    defer g.release(&counters);
    const supers = [4][4]f32{ .{ -bound, -bound, -bound, 0 }, .{ bound, bound, -bound, 0 }, .{ bound, -bound, bound, 0 }, .{ -bound, bound, bound, 0 } };
    g.writeSlice(coordinates, @as(u64, n) * 16, [4]f32, &supers);
    var params = Params{ .n = n, .seeds = batch, .first = 0, .cap = cap, .dim = dim, .pad = g.delaunay_mode & 1, .budget = if (g.delaunay_mode & 1 == 0) 256 else if (g.delaunay_mode >> 24 != 0) (g.delaunay_mode >> 24) * 64 else 4096, .lower = .{ lower[0], lower[1], lower[2], 0 }, .scale = .{ @as(f32, @floatFromInt(dim)) / width, upper[0], upper[1], upper[2] } };
    const clear = try g.pipeline("dt/grid_clear", code, "grid_clear");
    const build = try g.pipeline("dt/grid_build", code, "grid_build");
    const order = try g.pipeline("dt/grid_order", code, "grid_order");
    const init = try g.pipeline("dt/star_init", code, "star_init");
    const step = try g.pipeline(if (g.delaunay_mode & 1 != 0) "dt/star_fast" else "dt/star_step", if (g.delaunay_mode & 1 != 0) fast_code else code, "star_step");
    const wide = try g.pipeline("dt/double_sign", code, "double_sign");
    const query_wide = try g.pipeline("dt/query_double_sign", code, "query_double_sign");
    const exact = try g.pipeline("dt/exact_sign", code, "exact_sign");
    const post_key = try std.fmt.allocPrint(a, "dt/cooperative_step/{d}/{d}", .{ cap, g.delaunay_mode & 16 });
    defer a.free(post_key);
    const face_capacity = try std.fmt.allocPrint(a, "FACE_CAP={d}u", .{cap});
    defer a.free(face_capacity);
    const specialized_post = try std.mem.replaceOwned(u8, a, if (g.delaunay_mode & 16 != 0) deferred_code else post_code, "FACE_CAP=128u", face_capacity);
    defer a.free(specialized_post);
    const post = try g.pipeline(post_key, specialized_post, if (g.delaunay_mode & 16 != 0) "cooperative_deferred" else "cooperative_step");
    const summary = try g.pipeline("dt/star_summary", code, "star_summary");
    const emit = try g.pipeline("dt/star_emit", code, "star_emit");
    g.dispatch(clear, (dim * dim * dim + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), heads.at(2) });
    g.dispatch(build, (n + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), coordinates.at(1), heads.at(2), next.at(3) });
    g.dispatch(order, 1, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), heads.at(2), next.at(3), ordering.at(11) });
    var output = g.storage(16);
    errdefer g.release(&output);
    var total: u32 = 0;
    var cold_calls: u64 = 0;
    var rounds: u32 = 0;
    var max_faces: u32 = 0;
    while (params.first < n) : (params.first += params.seeds) {
        params.seeds = @min(batch, n - params.first);
        const binds = [_]gpu.Bind{ g.uniform(0, std.mem.asBytes(&params)), coordinates.at(1), next.at(3), faces.at(4), states.at(5), cache.at(6), scratch.at(7) };
        g.dispatch(init, (params.seeds + 63) / 64, 1, 1, &.{ binds[0], binds[1], binds[2], binds[3], binds[4], binds[5], binds[6], ordering.at(11) });
        var stats: [8]u32 = undefined;
        var complete = false;
        // A hull with many co-planar sites can require one exact sign per site.
        // Scale the work allowance with N instead of rejecting larger valid grids
        // at the small-input dispatch limit. Cancellation is checked each batch.
        for (0..@max(1024, @as(usize, n) / 2)) |_| {
            if (g.cancel_flag) |flag| if (flag.load(.acquire)) return error.Cancelled;
            const requested_rounds = (g.delaunay_mode >> 8) & 255;
            const first_rounds: u32 = if (g.delaunay_mode & 1 != 0 and requested_rounds != 0) requested_rounds else 32;
            if (g.delaunay_mode & 1 != 0 and rounds >= first_rounds) {
                const post_rounds: u32 = if (g.delaunay_mode & 16 != 0) 8 else 1;
                for (0..post_rounds) |_| {
                    if (g.delaunay_mode & 16 != 0) {
                        g.dispatch(post, params.seeds, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), coordinates.at(1), next.at(3), faces.at(4), states.at(5), cache.at(6), scratch.at(7) });
                    } else {
                        g.dispatch(post, params.seeds, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), coordinates.at(1), next.at(3), faces.at(4), states.at(5), cache.at(6), scratch.at(7), table.at(8) });
                    }
                    if (g.delaunay_mode & 16 != 0) {
                        g.dispatch(query_wide, params.seeds, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), coordinates.at(1), states.at(5), scratch.at(7) });
                        g.dispatch(wide, (params.seeds + 63) / 64, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), coordinates.at(1), states.at(5), cache.at(6), scratch.at(7) });
                        g.dispatch(exact, params.seeds, 1, 1, &.{ coordinates.at(1), states.at(5), cache.at(6), table.at(8) });
                    }
                }
                rounds += post_rounds;
            } else {
                for (0..first_rounds) |_| {
                    // Uniform slots are renewed across command-buffer arena flushes.
                    var b = binds;
                    b[0] = g.uniform(0, std.mem.asBytes(&params));
                    g.dispatch(step, (params.seeds + 63) / 64, 1, 1, &b);
                    g.dispatch(wide, (params.seeds + 63) / 64, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), coordinates.at(1), states.at(5), cache.at(6), scratch.at(7) });
                    g.dispatch(exact, params.seeds, 1, 1, &.{ coordinates.at(1), states.at(5), cache.at(6), table.at(8) });
                }
                rounds += first_rounds;
            }
            g.clear(counters);
            g.dispatch(summary, (params.seeds + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), coordinates.at(1), faces.at(4), states.at(5), counters.at(9) });
            g.read(counters, 0, std.mem.asBytes(&stats));
            try check(g);
            if (@import("builtin").target.os.tag != .freestanding and g.delaunay_mode & 2 != 0) {
                std.debug.print("DT round={d} done={d} active={d} failed={d} exact={d} max_faces={d}\n", .{ rounds, stats[0], stats[1], stats[2], stats[3], stats[4] });
            }
            if (stats[2] != 0) {
                if (stats[6] & ~@as(u32, 1 << 1) == 0) return error.GpuDelaunayCapacity;
                if (stats[6] & ~@as(u32, 1 << 8) == 0) return error.GpuDelaunayHull;
                if (stats[6] & @as(u32, 1 << 9) != 0) return error.DuplicateDelaunaySites;
                return error.GpuDelaunayTopology;
            }
            if (stats[1] == 0) {
                complete = true;
                break;
            }
        }
        if (!complete) {
            const diagnostic = try a.alloc(u32, @as(usize, params.seeds) * 28);
            defer a.free(diagnostic);
            g.read(states, 0, std.mem.sliceAsBytes(diagnostic));
            try check(g);
            var phase: u32 = 0;
            for (0..params.seeds) |s| if (diagnostic[s * 28] < 9) {
                phase = @max(phase, diagnostic[s * 28]);
            };
            if (phase == 8) return error.GpuDelaunayHullIterationLimit;
            if (phase == 4) return error.GpuDelaunaySearchIterationLimit;
            return error.GpuDelaunayIterationLimit;
        }
        cold_calls += stats[3];
        if (@import("builtin").target.os.tag != .freestanding) {
            if (g.delaunay_mode & 2 != 0) {
                const diagnostic = try a.alloc(u32, @as(usize, params.seeds) * 28);
                defer a.free(diagnostic);
                g.read(states, 0, std.mem.sliceAsBytes(diagnostic));
                try check(g);
                var worst: usize = 0;
                for (0..params.seeds) |s| if (diagnostic[s * 28 + 27] > diagnostic[worst * 28 + 27]) {
                    worst = s;
                };
                var c: [64]u32 = undefined;
                g.read(cache, worst * 8 * 32, std.mem.asBytes(&c));
                try check(g);
                std.debug.print("DT trace N={d} seed={d} cold={d} faces={d} cache={any}\n", .{ n, params.first + worst, diagnostic[worst * 28 + 27], diagnostic[worst * 28 + 1], c });
            }
        }
        max_faces = @max(max_faces, stats[4]);
        const needed = @as(u64, total + stats[5]) * 16;
        if (needed > g.lim.max_storage_binding) return error.GpuDelaunayOutputTooLarge;
        if (needed > output.size) {
            const bigger = g.storage(@max(needed, @min(g.lim.max_storage_binding, output.size * 2)));
            if (total != 0) g.copy(output, 0, bigger, 0, @as(u64, total) * 16);
            g.release(&output);
            output = bigger;
        }
        const position = [_]u32{total};
        g.writeSlice(counters, 7 * 4, u32, &position);
        g.dispatch(emit, (params.seeds + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), coordinates.at(1), faces.at(4), states.at(5), counters.at(9), output.at(10) });
        total += stats[5];
    }
    try check(g);
    if (total == 0) return error.LowerDimensionalDelaunay;
    return .{ .tets = output, .count = total, .cold_calls = cold_calls, .rounds = rounds, .max_faces = max_faces };
}
pub fn triangulateHost(a: std.mem.Allocator, g: *gpu.Gpu, points: []const Point) ![]Tet {
    const values = try a.alloc([4]f32, points.len + 4);
    defer a.free(values);
    var lower: [3]f32 = @splat(std.math.inf(f32));
    var upper: [3]f32 = @splat(-std.math.inf(f32));
    for (points, values[0..points.len]) |point_value, *v| {
        v.* = .{ point_value[0], point_value[1], point_value[2], 0 };
        for (point_value, 0..) |x, k| {
            if (!std.math.isFinite(x)) return error.NonFinite;
            lower[k] = @min(lower[k], x);
            upper[k] = @max(upper[k], x);
        }
    }
    var input = g.storage(values.len * 16);
    defer g.release(&input);
    g.writeSlice(input, 0, [4]f32, values[0..points.len]);
    var result = try triangulate(a, g, input, @intCast(points.len), lower, upper);
    defer g.release(&result.tets);
    const tets = try a.alloc(Tet, result.count);
    errdefer a.free(tets);
    g.read(result.tets, 0, std.mem.sliceAsBytes(tets));
    try check(g);
    return tets;
}

pub fn triangulateHostRegional(a: std.mem.Allocator, g: *gpu.Gpu, points: []const Point, margin: f64) ![]Tet {
    const values = try a.alloc([4]f32, points.len + 4);
    defer a.free(values);
    var lower: [3]f32 = @splat(std.math.inf(f32));
    var upper: [3]f32 = @splat(-std.math.inf(f32));
    for (points, values[0..points.len]) |point_value, *v| {
        v.* = .{ point_value[0], point_value[1], point_value[2], 0 };
        for (point_value, 0..) |x, k| {
            if (!std.math.isFinite(x)) return error.NonFinite;
            lower[k] = @min(lower[k], x);
            upper[k] = @max(upper[k], x);
        }
    }
    var input = g.storage(values.len * 16);
    defer g.release(&input);
    g.writeSlice(input, 0, [4]f32, values[0..points.len]);
    var result = try @import("gao_delaunay.zig").triangulate(g, input, @intCast(points.len), @intCast(points.len), lower, upper, margin);
    defer g.release(&result.tets);
    const tets = try a.alloc(Tet, result.count);
    errdefer a.free(tets);
    g.read(result.tets, 0, std.mem.sliceAsBytes(tets));
    try check(g);
    return tets;
}
