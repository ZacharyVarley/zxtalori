//! Fused orbit measures and Quey--Villani--Maurice relaxation.
//! Symmetry images are generated tile by tile in WGSL, never stored as a cloud.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const near_field = @import("near.zig");
const Group = @import("groups.zig").Group;
const Allocator = std.mem.Allocator;
const pi = std.math.pi;

pub const Row = extern struct {
    energy: [4]f32,
    extra: [4]f32,
    force: [4]f32,
};

pub const Measures = extern struct {
    count_extended: f64,
    riesz_1: f64,
    riesz_2: f64,
    riesz_3: f64,
    green_ratio: f64,
    log_ratio: f64,
    /// Mean of 1/d+1/(pi-d) over (source i, other source j, operator g).
    /// Constant own-orbit interactions are excluded, as in the Quey force.
    quey_energy: f64,
    /// Minimum SO(3) angle, not the S3 angle (degrees).
    separation_deg: f64,
    force_rms: f64,
};

pub const StepResult = extern struct {
    max_step_deg: f64,
    force_rms: f64,
    /// Energy of the input positions, before this update.
    energy: f64,
};

/// Pair evaluations one dispatch may do: about 0.1 s of work, so that no thread
/// outlives a GPU watchdog (Windows resets a device after 2 s; a browser loses it).
/// The full measures cost about 10x a force-only pass per pair.
const full_budget: f64 = 2.5e8;
const force_budget: f64 = 3e9;

/// Largest multiple of 64 that keeps `threads * per_thread` under `budget`, within [64, n].
fn chunkLength(budget: f64, threads: u64, n: u32) u32 {
    const fit = budget / @as(f64, @floatFromInt(@max(1, threads)));
    const length: u32 = @intFromFloat(@min(@as(f64, @floatFromInt(n)), @max(64, fit)));
    return @min(n, @max(64, length / 64 * 64));
}

pub fn forceRows(gpa: Allocator, g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32, force_only: bool) ![]Row {
    if (n == 0 or order == 0) return error.EmptyOrientations;
    if (force_only) return forceRowsParallel(gpa, g, q, n, ops, order);
    const pipe = try g.pipeline("pairs/fused", shaders.pairs, "pairs");
    var out = g.storage(@as(u64, n) * @sizeOf(Row));
    defer g.release(&out);
    const rows = try gpa.alloc(Row, n);
    errdefer gpa.free(rows);
    // Dispatches cover a few operators over a range of sites; the rows carry the
    // sums (double-single log high/low terms included) from one dispatch to the next.
    const full_limit = full_budget * g.pair_budget_scale;
    const squared = @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n));
    const span: u32 = @intFromFloat(@min(@as(f64, @floatFromInt(order)), @max(1, @floor(full_limit / squared))));
    const length = chunkLength(full_limit, @as(u64, n) * span, n);
    var first: u32 = 0;
    while (first < order) : (first += span) {
        const count = @min(span, order - first);
        var j: u32 = 0;
        while (j < n) : (j += length) {
            if (g.cancel_flag) |flag| if (flag.load(.acquire)) return error.Cancelled;
            const resumed: u32 = @intFromBool(first > 0 or j > 0);
            const params = [8]u32{ n, order, (first << 8) | (count << 16), 0, j, @min(n, j + length), resumed, 0 };
            g.dispatch(pipe, (n + 63) / 64, 1, 1, &.{
                g.uniform(0, std.mem.asBytes(&params)), q.at(1), ops.at(2), out.at(3),
            });
            g.submit();
        }
    }
    g.read(out, 0, std.mem.sliceAsBytes(rows));
    g.submit();
    try g.dev.wait();
    if (g.dev.takeError() != null) return error.GpuValidation;
    return rows;
}

fn forceRowsParallel(gpa: Allocator, g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32) ![]Row {
    var partial: gpu.Buf = .{};
    defer g.release(&partial);
    var out: gpu.Buf = .{};
    defer g.release(&out);
    try forcePass(g, q, n, ops, order, &partial, &out);
    const rows = try gpa.alloc(Row, n);
    errdefer gpa.free(rows);
    g.read(out, 0, std.mem.sliceAsBytes(rows));
    g.submit();
    try g.dev.wait();
    if (g.dev.takeError() != null) return error.GpuValidation;
    return rows;
}

/// Records the Quey force pass into `out` (one Row per site); `partial` is scratch.
fn forcePass(g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32, partial: *gpu.Buf, out: *gpu.Buf) !void {
    const pipe = try g.pipeline("pairs/quey-partial", shaders.pairs, "quey_partial");
    const reducer = try g.pipeline("pairs/quey-reduce", shaders.pairs, "quey_reduce");
    const row_bytes = @as(u64, n) * @sizeOf(Row);
    // Small sets get one operator per partial row for occupancy. Bound scratch
    // memory as N grows: increase the batch instead of extending orientations.
    const budget = @min(@as(u64, 32 * 1024 * 1024), @min(g.lim.max_storage_binding, g.lim.max_buffer));
    const max_batches: u32 = @intCast(@max(1, @min(@as(u64, order), budget / row_bytes)));
    const batch = (order + max_batches - 1) / max_batches;
    const batches = (order + batch - 1) / batch;
    g.ensure(partial, row_bytes * batches);
    g.ensure(out, row_bytes);
    // Every dispatch sums a range of sites for all operators; the partial rows carry on.
    const length = chunkLength(force_budget * g.pair_budget_scale, @as(u64, n) * order, n);
    var j: u32 = 0;
    while (j < n) : (j += length) {
        if (g.cancel_flag) |flag| if (flag.load(.acquire)) return error.Cancelled;
        const params = [8]u32{ n, order, batch, 0, j, @min(n, j + length), @intFromBool(j > 0), 0 };
        g.dispatch(pipe, (n + 63) / 64, batches, 1, &.{
            g.uniform(0, std.mem.asBytes(&params)), q.at(1), ops.at(2), partial.at(3),
        });
        g.submit();
    }
    const params = [8]u32{ n, order, batch, 0, 0, 0, 0, 0 };
    g.dispatch(reducer, (n + 63) / 64, 1, 1, &.{
        g.uniform(0, std.mem.asBytes(&params)), partial.at(6), out.at(7),
    });
}

fn summary(rows: []const Row, order: u32) struct { quey: f64, rms: f64 } {
    var energy: f64 = 0;
    var sq: f64 = 0;
    for (rows) |row| {
        energy += row.extra[1];
        for (row.force[0..3]) |value| sq += @as(f64, value) * value;
    }
    const n: f64 = @floatFromInt(rows.len);
    const denominator = n * @as(f64, @floatFromInt(order)) * @max(n - 1, 1);
    return .{ .quey = energy / denominator, .rms = @sqrt(sq / n) };
}

pub fn measure(gpa: Allocator, g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32) !Measures {
    const rows = try forceRows(gpa, g, q, n, ops, order, false);
    defer gpa.free(rows);
    var sum = [_]f64{0} ** 5;
    var near: f64 = 4;
    for (rows) |row| {
        for (0..3) |index| sum[index] += row.energy[index];
        sum[3] += @as(f64, row.energy[3]) + row.extra[3];
        sum[4] += row.extra[0];
        near = @min(near, row.extra[2]);
    }
    if (near <= 0) return error.DuplicateOrientations;
    const extension: f64 = @floatFromInt(2 * order);
    const total: f64 = @as(f64, @floatFromInt(n)) * extension;
    for (&sum) |*value| value.* *= extension;
    const c_green = std.math.pow(f64, 3.0, 5.0 / 3.0) /
        (5.0 * std.math.pow(f64, 2.0 * pi * pi, 1.0 / 3.0) * std.math.pow(f64, 4.0 * pi, 2.0 / 3.0));
    // log = centred - (N^2-N)/4, so the remainder avoids subtracting N^2.
    const delta_log = (sum[3] + 0.25 * total + @log(total) * total / 3.0) / total;
    const forces = summary(rows, order);
    return .{
        .count_extended = total,
        .riesz_1 = sum[0] / (total * total * (8.0 / (3.0 * pi))),
        .riesz_2 = sum[1] / (total * total),
        .riesz_3 = sum[2] / ((2.0 / (3.0 * pi)) * total * total * @log(total)),
        .green_ratio = sum[4] / (-c_green * std.math.pow(f64, total, 4.0 / 3.0)),
        .log_ratio = delta_log / (-5.0 / 12.0),
        .quey_energy = forces.quey,
        .separation_deg = 4.0 * std.math.asin(@min(1, @sqrt(near) / 2.0)) * 180.0 / pi,
        .force_rms = forces.rms,
    };
}

/// Current Quey/separation diagnostics without the expensive log/ratio path.
pub fn measureQuick(gpa: Allocator, g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32) !Measures {
    const rows = try forceRows(gpa, g, q, n, ops, order, true);
    defer gpa.free(rows);
    return quickMeasures(rows, order);
}

fn quickMeasures(rows: []const Row, order: u32) !Measures {
    const n: u32 = @intCast(rows.len);
    var near: f64 = 4;
    for (rows) |row| near = @min(near, row.extra[2]);
    if (near <= 0) return error.DuplicateOrientations;
    const stats = summary(rows, order);
    return .{
        .count_extended = @floatFromInt(@as(u64, 2) * order * n),
        .riesz_1 = std.math.nan(f64),
        .riesz_2 = std.math.nan(f64),
        .riesz_3 = std.math.nan(f64),
        .green_ratio = std.math.nan(f64),
        .log_ratio = std.math.nan(f64),
        .quey_energy = stats.quey,
        .separation_deg = 4.0 * std.math.asin(@min(1, @sqrt(near) / 2.0)) * 180.0 / pi,
        .force_rms = stats.rms,
    };
}

fn nearestDegrees(rows: []const Row, out: []f32) void {
    for (rows, out) |row, *o| o.* = @floatCast(4.0 * std.math.asin(@min(1, @sqrt(@as(f64, row.extra[2])) / 2.0)) * 180.0 / pi);
}

fn meanRadius(count: u64) f64 {
    // Same mean-neighborhood scale as the archived Quey relaxation.
    const target = pi * pi / @as(f64, @floatFromInt(count));
    var lo: f64 = 0;
    var hi: f64 = pi;
    for (0..60) |_| {
        const middle = 0.5 * (lo + hi);
        if (pi * (middle - @sin(middle)) < target) lo = middle else hi = middle;
    }
    return 0.25 * (lo + hi);
}

/// Barzilai-Borwein relaxation with every step on the GPU (`shaders/relax.wgsl`): forces,
/// BB sums, the exact median curvature and the move, with no readback between steps.
/// `finish` reads the last step's results once per batch.
pub const RelaxState = struct {
    g: ?*gpu.Gpu = null,
    n: u32 = 0,
    rows: gpu.Buf = .{},
    partial: gpu.Buf = .{},
    previous: gpu.Buf = .{},
    state: gpu.Buf = .{},
    sums: gpu.Buf = .{},
    histogram: gpu.Buf = .{},
    /// `rows` holds the forces of the current positions (the measure that ended the
    /// last relax call): the next step starts from them instead of a second force pass.
    current: bool = false,
    /// Group order of the last step, for `finish`.
    order: u32 = 0,
    /// Near-field forces (near.zig) instead of all pairs, when set (see `setNear`).
    near: ?near_field.Near = null,

    pub fn reset(self: *RelaxState, gpa: Allocator) void {
        _ = gpa;
        self.release();
    }

    fn release(self: *RelaxState) void {
        if (self.g) |g| {
            for ([_]*gpu.Buf{ &self.rows, &self.partial, &self.previous, &self.state, &self.sums, &self.histogram }) |b| g.release(b);
            if (self.near) |*nf| nf.release(g);
        }
        // The near-field settings stay (its buffers are gone, and it sizes them again).
        const near: ?near_field.Near = if (self.near) |nf| .{ .a = nf.a, .group = nf.group, .delta = nf.delta, .taper = nf.taper } else null;
        self.* = .{ .near = near };
    }

    /// Forces from the orbit points within delta mean orientation radii of each source
    /// (near.zig), or with `delta` null from all pairs. Kept across resets.
    pub fn setNear(self: *RelaxState, gpa: Allocator, g: *gpu.Gpu, group: Group, delta: ?f64) void {
        if (self.near) |*nf| {
            if (delta != null and nf.group == group and nf.delta == delta.?) return;
            nf.release(g);
            self.near = null;
            self.current = false;
        }
        if (delta) |d| {
            self.near = .{ .a = gpa, .group = group, .delta = d };
            self.current = false;
        }
    }

    /// Records the forces of the current positions into `rows`.
    fn recordForces(self: *RelaxState, g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32) !void {
        if (self.near) |*nf| {
            g.ensure(&self.rows, @as(u64, n) * @sizeOf(Row));
            try nf.pass(g, q, n, ops, order, meanRadius(@as(u64, n) * order), self.rows);
        } else try forcePass(g, q, n, ops, order, &self.partial, &self.rows);
    }

    /// Forget the current forces; call whenever the positions change outside `step`.
    pub fn dropCache(self: *RelaxState, gpa: Allocator) void {
        _ = gpa;
        self.current = false;
    }

    pub fn deinit(self: *RelaxState, gpa: Allocator) void {
        self.reset(gpa);
    }

    fn prepare(self: *RelaxState, g: *gpu.Gpu, n: u32) void {
        if (self.g != null and self.n == n) return;
        self.release();
        self.g = g;
        self.n = n;
        self.previous = g.storage(@as(u64, n) * 32);
        self.state = g.storage(16 * 4);
        g.writeSlice(self.state, 0, u32, &@as([16]u32, @splat(0)));
        self.sums = g.storage(@as(u64, (n + 63) / 64) * 32);
        self.histogram = g.storage(512 * 4);
        g.writeSlice(self.histogram, 0, u32, &@as([512]u32, @splat(0)));
    }

    /// Quey energy and separation of the current positions, keeping their forces for the next step.
    /// `nearest` (n values), when given, receives each source's nearest orbit point (SO(3) degrees).
    pub fn measure(self: *RelaxState, gpa: Allocator, g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32, nearest: ?[]f32) !Measures {
        self.prepare(g, n);
        self.current = false;
        try self.recordForces(g, q, n, ops, order);
        const rows = try gpa.alloc(Row, n);
        defer gpa.free(rows);
        g.read(self.rows, 0, std.mem.sliceAsBytes(rows));
        g.submit();
        try g.dev.wait();
        if (g.dev.takeError() != null) return error.GpuValidation;
        if (self.near) |*nf| if (nf.overflowed()) return self.measure(gpa, g, q, n, ops, order, nearest);
        const result = try quickMeasures(rows, order);
        if (nearest) |out| nearestDegrees(rows, out);
        self.current = true;
        return result;
    }

    /// Records one step; nothing is read back. A non-finite force or a duplicate orientation
    /// stops every later move and is reported by `finish`.
    pub fn step(self: *RelaxState, g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32, step_scale: f32, fixed: ?u32) !void {
        if (!(step_scale > 0) or !std.math.isFinite(step_scale)) return error.InvalidStepScale;
        self.prepare(g, n);
        if (!self.current) try self.recordForces(g, q, n, ops, order);
        self.current = false;
        self.order = order;
        const radius = meanRadius(@as(u64, n) * order);
        const groups = (n + 63) / 64;
        var params = RelaxParams{ .n = n, .fixed = if (fixed) |f| f + 1 else 0, .round = 0, .partials = groups, .radius = @floatCast(radius), .alpha0 = @floatCast((0.8 / 3.4) * radius * radius * radius), .step_scale = step_scale };
        const stats = try g.pipeline("relax/stats", shaders.relax, "relax_stats");
        const sums = try g.pipeline("relax/sums", shaders.relax, "relax_sums");
        const count = try g.pipeline("relax/median_count", shaders.relax, "median_count");
        const pick = try g.pipeline("relax/median_pick", shaders.relax, "median_pick");
        const move = try g.pipeline("relax/move", shaders.relax, "relax_move");
        g.dispatch(stats, groups, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), self.rows.at(2), self.previous.at(3), self.state.at(4), self.sums.at(5) });
        g.dispatch(sums, 1, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), self.state.at(4), self.sums.at(5), self.histogram.at(6) });
        // Enough workgroups to fill a GPU; each strides over the sites.
        const median_groups = @max(1, @min((n + 255) / 256, 256));
        for (0..4) |round| {
            params.round = @intCast(round);
            g.dispatch(count, median_groups, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), self.rows.at(2), self.state.at(4), self.histogram.at(6) });
            g.dispatch(pick, 1, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), self.state.at(4), self.histogram.at(6) });
        }
        g.dispatch(move, groups, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), q.at(1), self.rows.at(2), self.previous.at(3), self.state.at(4) });
    }

    /// Waits for the recorded steps and reports the last one.
    pub fn finish(self: *RelaxState, g: *gpu.Gpu) !StepResult {
        var values: [16]u32 = undefined;
        g.read(self.state, 0, std.mem.asBytes(&values));
        g.submit();
        try g.dev.wait();
        if (g.dev.takeError() != null) return error.GpuValidation;
        return self.stepResult(values);
    }

    /// Ends a batch with one wait: the last step's results (when steps were recorded since
    /// the last `finish`), the positions into `host`, and the Quey energy and separation
    /// of those positions, whose forces start the next step.
    pub fn complete(self: *RelaxState, gpa: Allocator, g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32, host: []f32, recorded: bool, nearest: ?[]f32) !struct { measures: Measures, step: ?StepResult } {
        self.prepare(g, n);
        self.current = false;
        try self.recordForces(g, q, n, ops, order);
        const rows = try gpa.alloc(Row, n);
        defer gpa.free(rows);
        var values: [16]u32 = undefined;
        if (recorded) g.read(self.state, 0, std.mem.asBytes(&values));
        g.read(q, 0, std.mem.sliceAsBytes(host));
        g.read(self.rows, 0, std.mem.sliceAsBytes(rows));
        g.submit();
        try g.dev.wait();
        if (g.dev.takeError() != null) return error.GpuValidation;
        const last: ?StepResult = if (recorded) try self.stepResult(values) else null;
        // A shell that outgrew its buffers is sized again and the forces redone.
        if (self.near) |*nf| if (nf.overflowed()) {
            const again = try self.complete(gpa, g, q, n, ops, order, host, false, nearest);
            return .{ .measures = again.measures, .step = last };
        };
        const measures = try quickMeasures(rows, order);
        if (nearest) |out| nearestDegrees(rows, out);
        self.current = true;
        return .{ .measures = measures, .step = last };
    }

    fn stepResult(self: *RelaxState, values: [16]u32) !StepResult {
        if (values[1] & 1 != 0) return error.NonFiniteForce;
        if (values[1] & 2 != 0) return error.DuplicateOrientations;
        const n: f64 = @floatFromInt(self.n);
        const step_max: f32 = @bitCast(values[2]);
        const forces: f32 = @bitCast(values[6]);
        const energy: f32 = @bitCast(values[7]);
        return .{
            .max_step_deg = @as(f64, step_max) * 360 / pi,
            .force_rms = @sqrt(@as(f64, forces) / n),
            .energy = @as(f64, energy) / (n * @as(f64, @floatFromInt(self.order)) * @max(n - 1, 1)),
        };
    }
};
const RelaxParams = extern struct { n: u32, fixed: u32, round: u32, partials: u32, radius: f32, alpha0: f32, step_scale: f32, pad: f32 = 0 };
