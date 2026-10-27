//! One GPU engine shared by the browser and Python. No symmetry-expanded cloud.
const std = @import("std");
pub const gpu = @import("gpu/gpu.zig");
const groups = @import("groups.zig");
const grids = @import("grids.zig");
const mtex = @import("mtex.zig");
const dedup = @import("dedup.zig");
const pairs = @import("pairs.zig");
const insert = @import("insert.zig");
const covering = @import("covering.zig");
const clock = @import("clock.zig");
const shaders = @import("shaders");
pub const version = "0.1.0";

pub const Engine = struct {
    allocator: std.mem.Allocator,
    dev: *gpu.Backend.Device,
    g: gpu.Gpu,
    group: groups.Group = .O,
    q: gpu.Buf = .{},
    ops: gpu.Buf = .{},
    host: []f32 = &.{},
    n: u32 = 0,
    iteration: u32 = 0,
    h: u32 = 0,
    seed: u32 = 0,
    generator: []const u8 = "none",
    removed_duplicates: u32 = 0,
    measures: ?pairs.Measures = null,
    /// The iteration `measures` describe (later steps may have moved the sample since).
    measure_iteration: u32 = 0,
    full_measures: bool = false,
    relax_state: pairs.RelaxState = .{},
    /// Relaxation forces from orbit points within this many mean orientation radii
    /// (Quey's delta, near.zig); 0 for all pairs. Energies measured with it set are the
    /// near-field sums.
    neighbourhood: f64 = 0,
    /// The neighbourhood `measures` were summed over (0: all pairs).
    measure_neighbourhood: f64 = 0,
    /// Per-point values for display, each with the iteration it describes (stats
    /// point_separation_iteration, point_covering_iteration): the nearest orbit
    /// neighbour (SO(3) degrees, from every energy pass), and from every covering pass the
    /// largest Delaunay cap each point touches (degrees) and its Voronoi faces.
    point_separation: []f32 = &.{},
    point_covering: []f32 = &.{},
    point_faces: []f32 = &.{},
    point_local: []u32 = &.{},
    point_local_gpu: gpu.Buf = .{},
    point_separation_iteration: ?u32 = null,
    point_covering_iteration: ?u32 = null,
    step_deg: f64 = 0,
    force_rms: f64 = 0,
    cancelled: std.atomic.Value(bool) = .init(false),
    fix_identity: bool = false,
    fixed_index: ?u32 = null,
    base_count: u32 = 0,
    inserted_count: u32 = 0,
    cover_result: ?covering.Result = null,
    cover_iteration: u32 = 0,
    /// GPU per-source trust radii for coverLocalGpu; reset whenever the sample is replaced.
    trust_gpu: gpu.Buf = .{},
    trust_n: u32 = 0,
    opt: CoverOpt = .{},
    mesh: covering.Mesh = .{},
    err: [2048]u8 = @splat(0),
    text: [8192]u8 = @splat(0),

    pub fn create(a: std.mem.Allocator, dev: *gpu.Backend.Device) !*Engine {
        const self = try a.create(Engine);
        errdefer a.destroy(self);
        self.* = .{ .allocator = a, .dev = dev, .g = try gpu.Gpu.init(a, dev) };
        self.g.cancel_flag = &self.cancelled;
        self.setGroup(.O);
        return self;
    }
    pub fn destroy(self: *Engine) void {
        self.mesh.deinit(self.allocator);
        for ([_]*[]f32{ &self.point_separation, &self.point_covering, &self.point_faces }) |s| self.allocator.free(s.*);
        self.allocator.free(self.point_local);
        self.g.release(&self.point_local_gpu);
        self.relax_state.deinit(self.allocator);
        self.allocator.free(self.host);
        self.g.release(&self.trust_gpu);
        self.g.release(&self.opt.best);
        self.g.release(&self.opt.start);
        self.opt.live.release(&self.g);
        self.g.release(&self.q);
        self.g.release(&self.ops);
        self.g.deinit();
        self.allocator.destroy(self);
    }
    fn setGroup(self: *Engine, group: groups.Group) void {
        self.group = group;
        const ops = groups.operations(group);
        self.g.ensure(&self.ops, ops.len * 16);
        self.g.writeSlice(self.ops, 0, [4]f32, ops);
    }
    pub fn fail(self: *Engine, comptime fmt: []const u8, args: anytype) i32 {
        _ = std.fmt.bufPrintZ(&self.err, fmt, args) catch {};
        return -1;
    }
    fn check(self: *Engine) !void {
        try self.dev.wait();
        if (self.dev.takeError()) |message| {
            _ = self.fail("{s}", .{message});
            return error.GpuValidation;
        }
    }
    fn fetch(self: *Engine, q: gpu.Buf, n: u32) ![]f32 {
        const host = try self.allocator.alloc(f32, @as(usize, n) * 4);
        errdefer self.allocator.free(host);
        self.g.read(q, 0, std.mem.sliceAsBytes(host));
        self.g.submit();
        try self.check();
        return host;
    }
    fn resetRun(self: *Engine) void {
        self.mesh.deinit(self.allocator);
        self.point_separation_iteration = null;
        self.point_covering_iteration = null;
        self.relax_state.reset(self.allocator);
        self.iteration = 0;
        self.step_deg = 0;
        self.force_rms = 0;
        self.measures = null;
        self.full_measures = false;
        self.cancelled.store(false, .release);
        self.cover_result = null;
        self.err[0] = 0;
    }
    fn adopt(self: *Engine, q: gpu.Buf, host: []f32, n: u32) void {
        self.g.release(&self.q);
        self.allocator.free(self.host);
        self.q = q;
        self.host = host;
        self.n = n;
        self.fixed_index = null;
        self.trust_n = 0;
        self.opt.active = false;
        // A new sample may be a lattice: cover it first unfenced, on a box-sized grid.
        self.g.gao_degenerate = true;
        self.g.gao_fill = 1;
        if (self.fix_identity) self.fixed_index = self.identityIndex();
        self.resetRun();
    }
    fn identityIndex(self: *Engine) ?u32 {
        for (0..self.n) |i| {
            const a = self.host[i * 4 ..][0..4];
            if (@abs(a[1]) + @abs(a[2]) + @abs(a[3]) < 1e-6) return @intCast(i);
        }
        return null;
    }
    pub fn identity(self: *Engine, add: bool, fixed: bool) !void {
        if (self.n == 0) return error.NoOrientations;
        var index = self.identityIndex();
        if (add and index == null) {
            const generator = self.generator;
            const h = self.h;
            const base_count = self.base_count;
            const inserted_count = self.inserted_count;
            const values = try self.allocator.alloc(f32, (self.n + 1) * 4);
            defer self.allocator.free(values);
            @memcpy(values[0..self.host.len], self.host);
            @memcpy(values[self.host.len..], &[_]f32{ 1, 0, 0, 0 });
            try self.set(self.group, values);
            self.generator = generator;
            self.h = h;
            self.base_count = base_count;
            self.inserted_count = inserted_count;
            index = self.n - 1;
        }
        if (fixed and index == null) return error.IdentityMissingEnableAddIdentity;
        if (self.fix_identity != fixed) self.relax_state.reset(self.allocator);
        self.fix_identity = fixed;
        self.fixed_index = if (fixed) index else null;
        if (fixed) {
            const i = index.?;
            @memcpy(self.host[i * 4 ..][0..4], &[_]f32{ 1, 0, 0, 0 });
            self.g.writeSlice(self.q, @as(u64, i) * 16, f32, self.host[i * 4 ..][0..4]);
            self.relax_state.dropCache(self.allocator);
        }
    }
    pub fn generateExact(self: *Engine, group: groups.Group, target: u32, add_identity: bool) !void {
        if (target < 32 or target > 1000000) return error.ExactCountRange32To1000000;
        var h: u32 = 1;
        var request: u32 = 8;
        if (@intFromEnum(group) < 8) {
            const budget = target - @as(u32, if (add_identity) 1 else 0);
            var best: u32 = 0;
            // Counts change only when either rounded angular subdivision changes.
            const order = @as(f64, @floatFromInt(groups.operations(group).len));
            const beta: f64 = if (@intFromEnum(group) < 4) std.math.pi else std.math.pi / 2.0;
            const gamma = if (@intFromEnum(group) < 4) std.math.pi / order else 2 * std.math.pi / order;
            for ([_]f64{ beta, gamma }) |axis| {
                var j: u32 = 0;
                while (true) : (j += 1) {
                    const radius = (@as(f64, @floatFromInt(j)) + 0.5) / axis;
                    const boundary = radius * radius * radius * beta * gamma;
                    if (boundary > @as(f64, @floatFromInt(target * 2)) + 3) break;
                    const centre: i64 = @intFromFloat(@floor(boundary));
                    for (0..5) |offset| {
                        const r = @max(8, centre + @as(i64, @intCast(offset)) - 2);
                        if (r > target * 2) continue;
                        const count = try mtex.rawCount(group, @intCast(r));
                        if (count <= budget and count > best) {
                            best = count;
                            request = @intCast(r);
                        }
                    }
                }
            }
        } else {
            while (h < 64 and try grids.count(group, h + 1) <= target) : (h += 1) {}
        }
        try self.generate(group, h, request, 0);
        const base_count = self.n;
        if (add_identity) try self.identity(true, self.fix_identity);
        const generator = self.generator;
        for (0..8) |_| {
            if (self.n == target) break;
            if (self.n > target) return error.BaseGridExceedsTarget;
            const additions = try insert.centres(self.allocator, &self.g, self.host, group, target - self.n);
            defer self.allocator.free(additions);
            const values = try self.allocator.alloc(f32, self.host.len + additions.len);
            defer self.allocator.free(values);
            @memcpy(values[0..self.host.len], self.host);
            @memcpy(values[self.host.len..], additions);
            const keep = try dedup.compact(self.allocator, values, groups.operations(group), self.n); // the current sample is already distinct
            if (keep <= self.n) return error.NoDistinctDelaunayCentres;
            try self.set(group, values[0 .. keep * 4]);
        }
        if (self.n != target) return error.InsufficientDelaunayCentres;
        self.generator = generator;
        self.h = if (@intFromEnum(group) < 8) 0 else h;
        self.base_count = base_count;
        self.inserted_count = target - base_count;
    }
    fn fold(self: *Engine, q: gpu.Buf, n: u32) !void {
        const u: [4]u32 = .{ n, @intCast(groups.operations(self.group).len), @intFromEnum(self.group), 0 };
        const pipe = try self.g.pipeline("state/fold", shaders.state, "fold");
        self.g.dispatch(pipe, (n + 255) / 256, 1, 1, &.{ q.at(0), self.ops.at(1), self.g.uniform(2, std.mem.asBytes(&u)) });
    }
    pub fn generate(self: *Engine, group: groups.Group, h: u32, target: u32, seed: u32) !void {
        if (h > 64 or target > 1000000) return error.GridTooLarge;
        const cyclic = @intFromEnum(group) < 8;
        const r = if (cyclic) blk: {
            const v = try mtex.generate(self.allocator, &self.g, group, if (target == 0) 2048 else target);
            break :blk grids.Result{ .quats = v.quats, .count = v.count };
        } else try grids.generate(self.allocator, &self.g, group, @max(1, h));
        var q = r.quats;
        errdefer self.g.release(&q);
        var host = try self.fetch(q, r.count);
        errdefer self.allocator.free(host);
        var n = r.count;
        self.removed_duplicates = 0;
        if (cyclic) {
            const keep = try dedup.compact(self.allocator, host, groups.operations(group), 0);
            if (keep != n) {
                self.removed_duplicates = n - @as(u32, @intCast(keep));
                n = @intCast(keep);
                const compact = try self.allocator.dupe(f32, host[0 .. keep * 4]);
                self.allocator.free(host);
                host = compact;
                self.g.writeSlice(q, 0, f32, host);
            }
        }
        self.setGroup(group);
        self.adopt(q, host, n);
        self.h = if (cyclic) 0 else @max(1, h);
        self.seed = seed;
        self.base_count = n;
        self.inserted_count = 0;
        self.generator = if (cyclic) "MTEX equispacedSO3Grid (seam repair)" else switch (group) {
            .T => "FCC octahedron transport",
            .O => "FCC truncated cube transport",
            .I => "FCC cube transport",
            else => unreachable,
        };
    }
    pub fn set(self: *Engine, group: groups.Group, values: []const f32) !void {
        if (values.len == 0 or values.len % 4 != 0 or values.len > 4000000) return error.InvalidCount;
        for (0..values.len / 4) |i| {
            var norm: f64 = 0;
            for (values[i * 4 ..][0..4]) |x| {
                if (!std.math.isFinite(x)) return error.NonFiniteQuaternion;
                norm += @as(f64, x) * x;
            }
            if (norm < 1e-20) return error.ZeroQuaternion;
        }
        const old_group = self.group;
        self.setGroup(group);
        errdefer self.setGroup(old_group);
        var q = self.g.storage(values.len * 4);
        errdefer self.g.release(&q);
        self.g.writeSlice(q, 0, f32, values);
        const n: u32 = @intCast(values.len / 4);
        try self.fold(q, n);
        const host = try self.fetch(q, n);
        self.adopt(q, host, n);
        self.h = 0;
        self.generator = "Imported quaternions";
        self.removed_duplicates = 0;
        self.base_count = n;
        self.inserted_count = 0;
    }
    pub fn measure(self: *Engine) !void {
        if (self.n == 0) return error.NoOrientations;
        const order: u32 = @intCast(groups.operations(self.group).len);
        // The Riesz, Green and log sums cost about 10x the Quey sum per pair; past
        // ratioLimit only Quey energy and separation are refreshed.
        if (self.n > self.ratioLimit()) {
            self.syncNeighbourhood();
            self.measures = try self.relax_state.measure(self.allocator, &self.g, self.q, self.n, self.ops, order, try self.separationOut());
            self.full_measures = false;
        } else {
            self.measures = try pairs.measure(self.allocator, &self.g, self.q, self.n, self.ops, order);
            self.full_measures = true;
            self.measure_neighbourhood = 0;
            // Per-point separations from one force pass (cheap below the ratio limit).
            self.syncNeighbourhood();
            _ = try self.relax_state.measure(self.allocator, &self.g, self.q, self.n, self.ops, order, try self.separationOut());
            self.measure_neighbourhood = 0;
        }
        self.measure_iteration = self.iteration;
        try self.check();
    }
    /// Largest sample of this group whose energy ratios Refresh computes.
    pub fn ratioLimit(self: *Engine) u32 {
        const order: f64 = @floatFromInt(groups.operations(self.group).len);
        return @intFromFloat(@floor(@sqrt(full_measure_pairs / order)));
    }
    pub fn buildMesh(self: *Engine) !void {
        self.cancelled.store(false, .release);
        const result = try covering.measure(self.allocator, &self.g, self.q, self.n, self.ops, self.group, null, &self.mesh, null, try self.localOut());
        self.cover_result = result;
        self.cover_iteration = self.iteration;
        try self.adoptLocal();
    }
    pub fn generateRandom(self: *Engine, group: groups.Group, count: u32, seed: u32) !void {
        if (count < 1 or count > 1000000) return error.InvalidCount;
        var q = self.g.storage(@as(u64, count) * 16);
        errdefer self.g.release(&q);
        const params = [4]u32{ count, seed, 0, 0 };
        const pipe = try self.g.pipeline("state/haar", shaders.haar, "haar");
        self.g.dispatch(pipe, (count + 255) / 256, 1, 1, &.{ q.at(0), self.g.uniform(1, std.mem.asBytes(&params)) });
        self.setGroup(group);
        try self.fold(q, count);
        const host = try self.fetch(q, count);
        self.adopt(q, host, count);
        self.generator = "Haar random";
        self.h = 0;
        self.seed = seed;
        self.base_count = count;
        self.inserted_count = 0;
    }
    pub fn measureCovering(self: *Engine) !void {
        if (self.n < 4) return error.TooFewOrientations;
        self.cancelled.store(false, .release);
        const hint = if (self.cover_result) |r| r.radius_deg * std.math.pi / 360 else null;
        const result = try covering.measure(self.allocator, &self.g, self.q, self.n, self.ops, self.group, hint, null, null, try self.localOut());
        self.cover_result = result;
        self.cover_iteration = self.iteration;
        try self.adoptLocal();
    }
    /// Covering optimization (shaders/local.wgsl). Each step is a local
    /// sequential-LP move: disjoint jobs around the worst caps, each solved in its
    /// own workgroup, then an exact re-check that reverts jobs whose local maximum
    /// did not drop. After `cover_patience` steps without a new best, the best
    /// sample is restored and kicked. `finish` leaves the best sample in place.
    /// Covering measures are current on return.
    pub fn optimizeCovering(self: *Engine, steps: u32, mode: u32, finish: bool, floor_deg: f32) !void {
        if (self.n < 4) return error.TooFewOrientations;
        if (mode & 3 > 1 or mode & ~@as(u32, 0xff001f) != 0) return error.InvalidCoveringMode;
        if (!(floor_deg >= 0 and floor_deg < 180)) return error.InvalidSeparationFloor;
        // The floor is an SO(3) angle; jobs work with the S3 angle, half of it, in radians.
        const floor: f32 = floor_deg * std.math.pi / 360;
        if (self.fix_identity and self.fixed_index == null) return error.IdentityMissingEnableAddIdentity;
        const g = &self.g;
        const bytes = @as(u64, self.n) * 16;
        if (!self.opt.active or self.opt.iteration != self.iteration) {
            // A new run, or the sample moved since (relaxation, import, identity).
            g.ensure(&self.opt.best, bytes);
            g.ensure(&self.opt.start, bytes);
            try self.endLive();
            g.copy(self.q, 0, self.opt.best, 0, bytes);
            self.opt = .{ .best = self.opt.best, .start = self.opt.start, .active = true };
            self.trust_n = 0;
        }
        self.cancelled.store(false, .release);
        for (0..@min(steps, 1000000)) |_| {
            if (self.cancelled.load(.acquire)) break;
            g.copy(self.q, 0, self.opt.start, 0, bytes);
            // Lattice-like or co-spherical samples overflow local jobs; full passes
            // handle them exactly until the sample is generic again.
            const result = try self.coverLocalGpu(cover_box, cover_band, cover_solver_iterations, cover_rounds, if (self.opt.fallback > 0) mode | 4 else mode, floor);
            if (self.opt.fallback > 0) {
                self.opt.fallback -= 1;
                self.opt.full_steps += 1;
            } else if (mode & 4 == 0 and result[0] > 0 and 2 * result[2] > result[0]) self.opt.fallback = cover_fallback_steps;
            // The step measured `start` exactly before moving it.
            const radius = self.cover_result.?.radius_deg;
            if (radius < self.opt.best_radius) {
                self.opt.best_radius = radius;
                g.copy(self.opt.start, 0, self.opt.best, 0, bytes);
            }
            // A descent has plateaued after cover_patience steps without a
            // relative gain of cover_progress; creeping gains do not count.
            if (radius < self.opt.reference * (1 - cover_progress)) {
                self.opt.reference = radius;
                self.opt.stall = 0;
            } else self.opt.stall += 1;
            self.opt.steps += 1;
            self.opt.jobs = result[0];
            self.opt.improved = result[1];
            if (self.opt.stall >= cover_patience) {
                // Every source moves: the live pass is built again from the kicked sample.
                try self.endLive();
                g.copy(self.opt.best, 0, self.q, 0, bytes);
                const rho: f32 = @floatCast(self.opt.best_radius * std.math.pi / 360);
                // Every source is kicked: at a plateau the caps within 1% of the largest
                // touch every source, so a kick aimed at the largest caps is global anyway.
                try self.localPass("local/kick", "kick", cover_kick * rho, self.opt.kicks);
                try self.fold(self.q, self.n);
                try self.localPass("local/fill_trust", "fill_trust", cover_kick_trust, 0);
                self.opt.kicks += 1;
                self.opt.reference = std.math.inf(f64);
                self.opt.stall = 0;
            }
        }
        // A stop ends the steps, not the run: the sample is still measured below.
        self.cancelled.store(false, .release);
        if (finish) {
            try self.endLive();
            g.copy(self.opt.best, 0, self.q, 0, bytes);
            // The best sample may be from a live stretch, unfolded as q was then.
            try self.fold(self.q, self.n);
        }
        // The points moved: relaxation's Barzilai-Borwein history no longer applies,
        // and its first step from stale differences can be huge.
        self.relax_state.reset(self.allocator);
        // The host copy is folded into the zone even while a live run keeps q unfolded.
        if (self.opt.unfolded) {
            g.copy(self.q, 0, self.opt.start, 0, bytes);
            try self.fold(self.opt.start, self.n);
            self.g.read(self.opt.start, 0, std.mem.sliceAsBytes(self.host));
        } else self.g.read(self.q, 0, std.mem.sliceAsBytes(self.host));
        self.g.submit();
        try self.check();
        // The covering radius: from the live covering while it is exact, else a full pass.
        if (finish or !try self.liveCovering()) try self.measureCovering();
        // Energy and separation (pair sums over the whole orbit) at most about a tenth of
        // the run's wall time: again once 9x their last cost has passed, and at the end.
        const now = clock.ms();
        if (finish or self.measures == null or self.opt.energy_ms == null or now - self.opt.energy_at >= 9 * self.opt.energy_ms.?) {
            self.syncNeighbourhood();
            self.measures = try self.relax_state.measure(self.allocator, &self.g, self.q, self.n, self.ops, @intCast(groups.operations(self.group).len), try self.separationOut());
            self.full_measures = false;
            self.measure_iteration = self.iteration;
            self.opt.energy_at = clock.ms();
            self.opt.energy_ms = self.opt.energy_at - now;
        }
        self.opt.iteration = self.iteration;
    }
    /// One per-source local.wgsl kernel over q and the trust radii.
    fn localPass(self: *Engine, key: []const u8, entry: []const u8, value: f32, seed: u32) !void {
        const params = LocalParams{ .n = self.n, .rows = 0, .jobs = 0, .iterations = 0, .band = 0, .top = 0, .grow = 0, .shrink = 0, .fixed = self.fixedParam(), .seed = seed, .value = value };
        const pipe = try self.g.pipeline(key, shaders.local, entry);
        const binds = if (std.mem.eql(u8, entry, "kick")) &[_]gpu.Bind{ self.g.uniform(0, std.mem.asBytes(&params)), self.q.at(16) } else &[_]gpu.Bind{ self.g.uniform(0, std.mem.asBytes(&params)), self.trust_gpu.at(15) };
        self.g.dispatch(pipe, (self.n + 255) / 256, 1, 1, binds);
    }
    fn fixedParam(self: *Engine) u32 {
        return if (self.fix_identity) if (self.fixed_index) |f| f + 1 else 0 else 0;
    }
    /// One local covering step; see optimizeCovering. Returns jobs, improved, and
    /// whether any job overflowed its free-source or cap slots.
    fn coverLocalGpu(self: *Engine, box: f32, band: f32, iterations: u32, rounds: u32, variant: u32, floor: f32) ![3]u32 {
        const g = &self.g;
        const n = self.n;
        const mode = variant & 3;
        const t0 = clock.ms();
        // Default: re-tessellate each job locally. Bit 4: full exact pass after
        // the move (the earlier path). Bit 8: re-tessellate, then check every
        // job's local maximum against a full exact pass.

        const full = variant & 4 != 0;
        const check_local = variant & 8 != 0;
        if (self.trust_n != n) {
            g.ensure(&self.trust_gpu, @as(u64, n) * 4);
            try self.localPass("local/fill_trust", "fill_trust", box, 0);
            self.trust_n = n;
        }
        var owned_rows: gpu.Buf = .{};
        defer g.release(&owned_rows);
        var count: u32 = 0;
        var owned_shell: covering.Shell = .{};
        defer owned_shell.release(g);
        const radius = if (self.cover_result) |r| r.radius_deg * std.math.pi / 360 else null;
        // Local steps read caps from the live pass (bit 16: a full pass every step instead).
        if (full or variant & 16 != 0) try self.endLive();
        const live_caps = if (full or variant & 16 != 0) null else try self.liveCaps(&owned_rows, &count, &owned_shell);
        // The shell must reach every orbit point a job gathers: 2.5 R around its free sources.
        const hint = if (radius) |r| (if (full) r else r * cover_shell_reach) else null;
        var before = if (live_caps) |result| result else try covering.measure(self.allocator, g, self.q, n, self.ops, self.group, hint, null, .{ .band = 0, .keep = &owned_rows, .count = &count, .shell = if (full) null else &owned_shell }, null);
        // Without a current radius (a new or relaxed sample) the shell is sized from
        // density and can be narrower than the jobs' 2.2 R gathers: measure again,
        // or every job is skipped and the step falls back to a full pass.
        if (live_caps == null and !full and 2.21 * before.radius_deg * std.math.pi / 360 > owned_shell.margin) {
            const reach = before.radius_deg * std.math.pi / 360 * cover_shell_reach;
            before = try covering.measure(self.allocator, g, self.q, n, self.ops, self.group, reach, null, .{ .band = 0, .keep = &owned_rows, .count = &count, .shell = &owned_shell }, null);
        }
        const live_on = live_caps != null;
        const rows = owned_rows;
        const shell = owned_shell;
        self.cover_result = before;
        const t1 = clock.ms();
        if (count == 0) return error.NoCoveringCircumcaps;
        const cap_jobs = n / 16 + 64;
        var params = LocalParams{ .n = n, .rows = count, .jobs = cap_jobs, .iterations = iterations, .band = band, .top = @floatCast(before.radius_deg * std.math.pi / 360), .grow = 1.25, .shrink = 0.5, .ring = mode, .fixed = self.fixedParam() };
        const per_source = @as(u64, n) * 4;
        const per_job = @as(u64, cap_jobs) * 4;
        // delta: one rotation per source, then one result per job.
        const sizes = [17]u64{ per_source, per_source, per_source, @as(u64, count) * 4, per_source, per_source, per_job * 24, per_job, per_job * 256, per_job, per_job, per_job, (@as(u64, n) + cap_jobs) * 16, @as(u64, n) * 16, per_source, per_source, per_source };
        var scratch: [17]gpu.Buf = undefined;
        for (&scratch, sizes) |*b, size| b.* = g.storage(size);
        defer for (&scratch) |*b| g.release(b);
        const core_claim, const free_claim, const region_claim, const lose, const job_of, const job_core, const job_free, const job_nfree, const job_rows, const job_n, const job_old, const job_new, const delta, const q_old, const taken, const near, const near2 = scratch;
        var counters = g.storage(64);
        defer g.release(&counters);
        for ([_]gpu.Buf{ job_of, job_core, job_nfree, job_n, job_old, job_new, delta, taken, near, near2, counters }) |b| g.clear(b);
        const code = shaders.local;
        const u = g.uniform(0, std.mem.asBytes(&params));
        const rg = (count + 255) / 256;
        const sg = (n + 255) / 256;
        for (0..rounds) |_| {
            for ([_]gpu.Buf{ core_claim, free_claim, region_claim, lose }) |b| g.clear(b);
            g.dispatch(try g.pipeline("local/mark_near", code, "mark_near"), rg, 1, 1, &.{ u, rows.at(1), taken.at(19), near.at(20) });
            if (mode != 0) g.dispatch(try g.pipeline("local/mark_near2", code, "mark_near2"), rg, 1, 1, &.{ u, rows.at(1), near.at(20), near2.at(21) });
            g.dispatch(try g.pipeline("local/claim_core", code, "claim_core"), rg, 1, 1, &.{ u, rows.at(1), core_claim.at(2), near.at(20), near2.at(21) });
            g.dispatch(try g.pipeline("local/claim_free", code, "claim_free"), rg, 1, 1, &.{ u, rows.at(1), core_claim.at(2), free_claim.at(3) });
            g.dispatch(try g.pipeline("local/claim_region", code, "claim_region"), rg, 1, 1, &.{ u, rows.at(1), free_claim.at(3), region_claim.at(4) });
            g.dispatch(try g.pipeline("local/mark_losers", code, "mark_losers"), rg, 1, 1, &.{ u, rows.at(1), core_claim.at(2), free_claim.at(3), region_claim.at(4), lose.at(5), taken.at(19) });
            g.dispatch(try g.pipeline("local/pick_jobs", code, "pick_jobs"), rg, 1, 1, &.{ u, rows.at(1), core_claim.at(2), lose.at(5), job_core.at(7), counters.at(18) });
            g.dispatch(try g.pipeline("local/assign_free", code, "assign_free"), rg, 1, 1, &.{ u, rows.at(1), job_of.at(6), job_core.at(7) });
            g.dispatch(try g.pipeline("local/mark_taken", code, "mark_taken"), rg, 1, 1, &.{ u, rows.at(1), job_of.at(6), taken.at(19) });
        }
        g.dispatch(try g.pipeline("local/list_free", code, "list_free"), sg, 1, 1, &.{ u, job_of.at(6), job_free.at(8), job_nfree.at(9), counters.at(18) });
        g.dispatch(try g.pipeline("local/gather_rows", code, "gather_rows"), rg, 1, 1, &.{ u, rows.at(1), job_of.at(6), job_rows.at(10), job_n.at(11), job_old.at(12), counters.at(18) });
        // The job count stays on the GPU: job_launch writes the per-job dispatch sizes.
        var launch = g.storage(32);
        defer g.release(&launch);
        g.dispatch(try g.pipeline("local/job_launch", code, "job_launch"), 1, 1, 1, &.{ u, counters.at(18), launch.at(25) });
        // A storage write read back as indirect counts needs a pass boundary on native wgpu
        // (same pass: the indirect dispatch saw stale counts). Submitting closes the pass; no wait.
        g.submit();
        var tally: [16]u32 = undefined;
        if (!full) {
            params.iterations = cover_local_solver_iterations;
            params.floor = floor;
            params.lower = .{ shell.lower[0], shell.lower[1], shell.lower[2], shell.scale };
            // Bits 16-23 of the variant override the local LP rounds per job.
            const inner = (variant >> 16) & 0xff;
            params.grid = .{ shell.dim, shell.count, if (inner != 0) inner else cover_local_rounds, 0 };
            params.value = @floatCast(shell.margin);
            const ul = g.uniform(0, std.mem.asBytes(&params));
            var table: [60][4]f32 = @splat(@splat(0));
            for (groups.operations(self.group), 0..) |op, k| for (0..4) |c| {
                table[k][c] = @floatCast(op[c]);
            };
            const ut = g.uniform(24, std.mem.asBytes(&table));
            g.dispatchIndirect(try g.pipeline("local/retessellate_jobs", code, "retessellate_jobs"), launch, 0, &.{ ul, ut, job_of.at(6), job_free.at(8), job_nfree.at(9), job_old.at(12), delta.at(14), self.trust_gpu.at(15), self.q.at(16), shell.index.at(23) });
            g.dispatch(try g.pipeline("local/apply_rotations", code, "apply_rotations"), sg, 1, 1, &.{ ul, job_of.at(6), delta.at(14), self.q.at(16) });
            if (live_on) self.opt.unfolded = true else try self.fold(self.q, n);
            g.dispatchIndirect(try g.pipeline("local/tally_local", code, "tally_local"), launch, 12, &.{ ul, delta.at(14), counters.at(18) });
            if (check_local) {
                var after_rows: gpu.Buf = .{};
                defer g.release(&after_rows);
                var after_count: u32 = 0;
                _ = try covering.measure(self.allocator, g, self.q, n, self.ops, self.group, before.radius_deg * std.math.pi / 360, null, .{ .band = 0, .keep = &after_rows, .count = &after_count }, null);
                var after_params = params;
                after_params.rows = after_count;
                g.dispatch(try g.pipeline("local/new_local", code, "new_local"), (after_count + 255) / 256, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&after_params)), after_rows.at(1), job_of.at(6), job_new.at(13) });
                g.dispatchIndirect(try g.pipeline("local/verify_local", code, "verify_local"), launch, 12, &.{ ul, job_new.at(13), delta.at(14), counters.at(18) });
            }
            // The live update is recorded before the tally readback, so its counters
            // land at the same wait.
            if (live_on) try self.liveBegin(job_of);
            g.read(counters, 0, std.mem.asBytes(&tally));
            g.submit();
            const t2 = clock.ms();
            try self.check();
            const t3 = clock.ms();
            if (live_on) self.liveFinish();
            for (&self.opt.times, [_]f64{ t1 - t0, t2 - t1, t3 - t2 }) |*total, x| total.* += x;
            self.opt.skipped += tally[2];
            self.opt.overflows += @intFromBool(tally[7] != 0);
            self.opt.local_rounds += tally[3];
            for (&self.opt.skip_reasons, tally[8..16]) |*total, add| total.* += add;
            if (check_local) {
                self.opt.mismatches += tally[4];
                self.opt.local_error = @max(self.opt.local_error, @as(f32, @bitCast(tally[5])));
            }
            self.iteration += 1;
            return .{ @min(tally[0], cap_jobs), tally[1], tally[2] };
        }
        const uj = g.uniform(0, std.mem.asBytes(&params));
        g.dispatchIndirect(try g.pipeline("local/solve_jobs", code, "solve_jobs"), launch, 0, &.{ uj, rows.at(1), job_free.at(8), job_nfree.at(9), job_rows.at(10), job_n.at(11), delta.at(14), self.trust_gpu.at(15) });
        g.copy(self.q, 0, q_old, 0, @as(u64, n) * 16);
        g.dispatch(try g.pipeline("local/apply_moves", code, "apply_moves"), sg, 1, 1, &.{ uj, job_of.at(6), delta.at(14), self.q.at(16) });
        try self.fold(self.q, n);
        var after_rows: gpu.Buf = .{};
        defer g.release(&after_rows);
        var after_count: u32 = 0;
        _ = try covering.measure(self.allocator, g, self.q, n, self.ops, self.group, before.radius_deg * std.math.pi / 360, null, .{ .band = 0, .keep = &after_rows, .count = &after_count }, null);
        var after_params = params;
        after_params.rows = after_count;
        g.dispatch(try g.pipeline("local/new_local", code, "new_local"), (after_count + 255) / 256, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&after_params)), after_rows.at(1), job_of.at(6), job_new.at(13) });
        g.dispatch(try g.pipeline("local/accept_jobs", code, "accept_jobs"), sg, 1, 1, &.{ uj, job_of.at(6), job_old.at(12), job_new.at(13), self.trust_gpu.at(15), self.q.at(16), q_old.at(17) });
        g.dispatchIndirect(try g.pipeline("local/count_improved", code, "count_improved"), launch, 12, &.{ uj, job_old.at(12), job_new.at(13), counters.at(18) });
        g.read(counters, 0, std.mem.asBytes(&tally));
        g.read(self.q, 0, std.mem.sliceAsBytes(self.host));
        g.submit();
        try self.check();
        self.opt.overflows += @intFromBool(tally[7] != 0);
        self.cover_result = before;
        self.iteration += 1;
        return .{ @min(tally[0], cap_jobs), tally[1], tally[2] };
    }
    /// Caps of the live pass for a local step, building it first when there is none, or
    /// when its margin m is less than the 2.21 R a job gathers plus the 0.1 R it may move
    /// a source (caps up to m are exact; see covering.Live). Null: a full pass this step
    /// (the margin cannot grow further).
    fn liveCaps(self: *Engine, rows: *gpu.Buf, count: *u32, shell: *covering.Shell) !?covering.Result {
        const g = &self.g;
        const order: u32 = @intCast(groups.operations(self.group).len);
        const density = @as(f64, @floatFromInt(@as(u64, self.n) * order));
        var wider: f64 = 0;
        for (0..6) |_| {
            if (!self.opt.live_on) {
                var margin = @max(wider, @min(1.569, 2.5 * std.math.cbrt(3 * std.math.pi / (4 * density))));
                if (self.cover_result) |r| margin = @max(margin, @min(1.569, r.radius_deg * std.math.pi / 360 * cover_live_reach));
                self.opt.live = covering.build(self.allocator, g, self.q, self.n, self.ops, self.group, margin, true) catch |err| {
                    if (err != error.CertificateFailed or margin >= 1.569) return err;
                    wider = @min(1.569, margin * 1.5);
                    continue;
                };
                self.opt.live_on = true;
                self.opt.live_builds += 1;
            }
            const live = &self.opt.live;
            const result = live.caps(self.allocator, g, self.q, self.ops, null, .{ .band = 0, .keep = rows, .count = count, .shell = shell }, null) catch |err| {
                if (err != error.CertificateFailed or live.margin >= 1.569) return err;
                wider = @min(1.569, live.margin * 1.5);
                self.dropLive();
                continue;
            };
            const r = result.radius_deg * std.math.pi / 360;
            if (live.margin >= (2.21 + 0.11) * r) return result;
            if (live.margin >= 1.569) {
                self.dropLive();
                return null;
            }
            // Built again around this radius (the next build reads it).
            self.dropLive();
            self.cover_result = result;
        }
        return null;
    }
    /// Records the live update for this step's moved sources (job_of nonzero).
    fn liveBegin(self: *Engine, moved: gpu.Buf) !void {
        try self.opt.live.beginMove(&self.g, self.q, self.ops, moved);
    }
    /// Checks the live update (after the step's wait); one it could not finish is
    /// dropped, and the next step builds again.
    fn liveFinish(self: *Engine) void {
        self.opt.live.finishMove() catch {
            self.opt.live_failures += 1;
            self.dropLive();
        };
    }
    /// The covering radius of the current sample from the live covering, when it has one
    /// and the radius is within its exact range (at most the margin; see covering.Live). Returns
    /// whether `cover_result` is now current.
    fn liveCovering(self: *Engine) !bool {
        if (!self.opt.live_on) return false;
        const live = &self.opt.live;
        const result = live.caps(self.allocator, &self.g, self.q, self.ops, null, null, try self.localOut()) catch |err| switch (err) {
            error.CertificateFailed, error.NoCoveringCircumcaps, error.CoveringWitnessMismatch => {
                self.dropLive();
                return false;
            },
            else => return err,
        };
        if (result.radius_deg * std.math.pi / 360 > live.margin) return false;
        self.cover_result = result;
        self.cover_iteration = self.iteration;
        try self.adoptLocal();
        return true;
    }
    fn dropLive(self: *Engine) void {
        if (!self.opt.live_on) return;
        self.opt.live.release(&self.g);
        self.opt.live_on = false;
    }
    /// Ends live updates for this run: q is folded into the zone again.
    fn endLive(self: *Engine) !void {
        self.dropLive();
        if (self.opt.unfolded) try self.fold(self.q, self.n);
        self.opt.unfolded = false;
    }
    /// Development check of live covering updates (covering.Live.move): builds the live
    /// pass, then `steps` times moves `moves` random sources by left rotations of up to
    /// `angle_deg` (S3) and updates it locally; compares the counted caps with a new
    /// build at the same margin. Returns caps before, caps after (live), caps of the new
    /// build, caps that differ between the two, caps the moves changed, and whether the
    /// largest radii agree (1), differ (0) or exceed the margin (2).
    pub fn liveCheck(self: *Engine, moves: u32, angle_deg: f32, seed: u32, steps: u32) ![6]u64 {
        if (self.n < 4) return error.TooFewOrientations;
        const a = self.allocator;
        const g = &self.g;
        if (self.cover_result == null) try self.measureCovering();
        const margin = @min(1.569, self.cover_result.?.radius_deg * std.math.pi / 360 * cover_live_reach);
        var live = try covering.build(a, g, self.q, self.n, self.ops, self.group, margin, true);
        defer live.release(g);
        const before = try live.capSet(a, g, self.ops);
        defer a.free(before);
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();
        const flags = try a.alloc(u32, self.n);
        defer a.free(flags);
        const angle: f64 = @as(f64, angle_deg) * std.math.pi / 180;
        var moved = g.storage(@as(u64, self.n) * 4);
        defer g.release(&moved);
        for (0..steps) |_| {
            @memset(flags, 0);
            for (0..@min(moves, self.n)) |_| {
                const s = random.uintLessThan(u32, self.n);
                flags[s] = 1;
                var v: [3]f64 = undefined;
                for (&v) |*x| x.* = random.floatNorm(f64);
                const length = @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                const theta = angle * random.float(f64);
                const k = @sin(theta) / length;
                const r = [4]f64{ @cos(theta), v[0] * k, v[1] * k, v[2] * k };
                const q = self.host[s * 4 ..][0..4];
                const x = [4]f64{ q[0], q[1], q[2], q[3] };
                const out = [4]f64{ r[0] * x[0] - r[1] * x[1] - r[2] * x[2] - r[3] * x[3], r[0] * x[1] + r[1] * x[0] + r[2] * x[3] - r[3] * x[2], r[0] * x[2] - r[1] * x[3] + r[2] * x[0] + r[3] * x[1], r[0] * x[3] + r[1] * x[2] - r[2] * x[1] + r[3] * x[0] };
                for (q, out) |*y, z| y.* = @floatCast(z);
            }
            g.writeSlice(self.q, 0, f32, self.host);
            g.writeSlice(moved, 0, u32, flags);
            try live.move(g, self.q, self.ops, moved);
        }
        if (@import("builtin").target.os.tag != .freestanding) std.debug.print("live stars: certified {d}, gave up (capacity {d}, candidates {d}, topology {d}, certificate {d})\n", .{ live.live_stats[0], live.live_stats[1], live.live_stats[2], live.live_stats[3], live.live_stats[4] });
        const after = try live.capSet(a, g, self.ops);
        defer a.free(after);
        var fresh = try covering.build(a, g, self.q, self.n, self.ops, self.group, margin, false);
        defer fresh.release(g);
        const rebuilt = try fresh.capSet(a, g, self.ops);
        defer a.free(rebuilt);
        const largest = struct {
            fn of(keys: []const covering.CapKey) f32 {
                var m: f32 = 0;
                for (keys) |k| m = @max(m, k.rho);
                return m;
            }
        }.of;
        // Caps right at the margin may fall either side of it in the two passes.
        const limit: f32 = @floatCast(margin * (1 - 1e-5));
        const top = largest(after);
        self.cover_result = null;
        self.relax_state.dropCache(a);
        return .{ before.len, after.len, rebuilt.len, keyDifference(after, rebuilt, limit), keyDifference(before, after, limit), if (top > limit) 2 else @intFromBool(top == largest(rebuilt)) };
    }
    /// Development: see zo_debug_shell.
    pub fn debugShell(self: *Engine, out: [][4]f32, counts: *[2]u32) !void {
        if (self.cover_result == null) try self.measureCovering();
        const order: u32 = @intCast(groups.operations(self.group).len);
        const density = @as(f64, @floatFromInt(@as(u64, self.n) * order));
        const margin = @max(@min(1.569, 2.5 * std.math.cbrt(3 * std.math.pi / (4 * density))), @min(1.569, self.cover_result.?.radius_deg * std.math.pi / 360 * 1.5));
        var live = try covering.build(self.allocator, &self.g, self.q, self.n, self.ops, self.group, margin, false);
        defer live.release(&self.g);
        const total = live.count + live.fence;
        if (total > out.len) return error.DebugBufferTooSmall;
        self.g.read(live.sites, 0, std.mem.sliceAsBytes(out[0..total]));
        self.g.submit();
        try self.check();
        counts.* = .{ live.count, live.fence };
    }
    /// `slice` sized for the current sample.
    fn sized(self: *Engine, comptime T: type, slice: *[]T, count: usize) ![]T {
        if (slice.len != count) {
            self.allocator.free(slice.*);
            slice.* = &.{};
            slice.* = try self.allocator.alloc(T, count);
        }
        return slice.*;
    }
    /// Receives per-point separations of the energy pass about to run (this iteration).
    fn separationOut(self: *Engine) ![]f32 {
        const out = try self.sized(f32, &self.point_separation, self.n);
        self.point_separation_iteration = self.iteration;
        return out;
    }
    fn localOut(self: *Engine) !covering.Local {
        return .{ .buf = &self.point_local_gpu, .host = try self.sized(u32, &self.point_local, 2 * @as(usize, self.n)) };
    }
    /// The covering pass's per-source words, after its readback, as degrees and faces.
    fn adoptLocal(self: *Engine) !void {
        const n = self.n;
        const radius = try self.sized(f32, &self.point_covering, n);
        const faces = try self.sized(f32, &self.point_faces, n);
        for (radius, faces, self.point_local[0..n], self.point_local[n .. 2 * n]) |*r, *f, bits, tets| {
            r.* = @as(f32, @bitCast(bits)) * 360 / std.math.pi;
            f.* = @floatFromInt(2 + tets / 2);
        }
        self.point_covering_iteration = self.iteration;
    }
    /// Per-point values for display: 0 separation, 1 local covering radius, 2 Voronoi
    /// faces; null before the first pass that makes them.
    pub fn pointValues(self: *Engine, kind: u32) ?[]const f32 {
        const values = switch (kind) {
            0 => if (self.point_separation_iteration != null) self.point_separation else return null,
            1 => if (self.point_covering_iteration != null) self.point_covering else return null,
            2 => if (self.point_covering_iteration != null) self.point_faces else return null,
            else => return null,
        };
        return if (values.len == self.n) values else null;
    }
    /// Sets the relaxation neighbourhood (Quey's delta; 0 for all pairs).
    pub fn setNeighbourhood(self: *Engine, delta: f64) !void {
        if (!(delta == 0 or (delta >= 2 and delta <= 1000))) return error.NeighbourhoodOutside2To1000;
        self.neighbourhood = delta;
    }
    /// The relaxation state's forces follow the neighbourhood setting and group; energies
    /// it measures from here on are summed over that neighbourhood.
    fn syncNeighbourhood(self: *Engine) void {
        self.relax_state.setNear(self.allocator, &self.g, self.group, if (self.neighbourhood > 0) self.neighbourhood else null);
        self.measure_neighbourhood = self.neighbourhood;
    }
    pub fn relax(self: *Engine, steps: u32) !void {
        if (self.n < 2) return error.TooFewOrientations;
        self.cancelled.store(false, .release);
        self.syncNeighbourhood();
        const order: u32 = @intCast(groups.operations(self.group).len);
        // Steps are recorded without reading anything back. Every `sync_every` steps (about
        // 2e10 pair evaluations, 0.2 s on a fast GPU) the engine waits, so Stop stays responsive
        // however many steps a caller asks for; the app's own batches are about 0.1 s. A
        // near-field neighbourhood holds about 0.94 delta^3 orbit points per source.
        const n_f: f64 = @floatFromInt(self.n);
        const neighbours = if (self.neighbourhood > 0) @min(n_f * @as(f64, @floatFromInt(order)), 0.94 * std.math.pow(f64, self.neighbourhood, 3) + 64) else n_f * @as(f64, @floatFromInt(order));
        const evaluations = n_f * neighbours;
        const sync_every: usize = @intFromFloat(@max(1, @floor(2e10 / evaluations)));
        var recorded = false;
        for (0..@min(steps, 1000)) |index| {
            if (self.cancelled.load(.acquire)) break;
            if (self.fix_identity and self.fixed_index == null) return error.IdentityMissingEnableAddIdentity;
            try self.relax_state.step(&self.g, self.q, self.n, self.ops, order, 1.0, self.fixed_index);
            // Right-action folding preserves the left tangent frame used by BB.
            // No force/history rotation is needed when a representative crosses a seam.
            try self.fold(self.q, self.n);
            self.iteration += 1;
            recorded = true;
            if ((index + 1) % sync_every == 0 and index + 1 < @min(steps, 1000)) {
                const result = try self.relax_state.finish(&self.g);
                self.step_deg = result.max_step_deg;
                self.force_rms = result.force_rms;
                recorded = false;
            }
        }
        // One wait ends the batch: last step, positions and the current energy and separation.
        // These measure forces are also the next step's.
        self.measures = null;
        const ending = try self.relax_state.complete(self.allocator, &self.g, self.q, self.n, self.ops, order, self.host, recorded, try self.separationOut());
        if (ending.step) |result| {
            self.step_deg = result.max_step_deg;
            self.force_rms = result.force_rms;
        }
        self.measures = ending.measures;
        self.full_measures = false;
        self.measure_iteration = self.iteration;
    }
    pub fn stats(self: *Engine) [*:0]const u8 {
        const m = self.measures;
        const bound = @import("covering_reference.zig").simplexDegrees(@as(u64, 2) * groups.operations(self.group).len * self.n);
        const covering_current = self.cover_result != null and self.cover_iteration == self.iteration;
        const measures_current = m != null and self.measure_iteration == self.iteration;
        var w: std.Io.Writer = .fixed(self.text[0 .. self.text.len - 1]);
        std.json.Stringify.value(.{
            .version = version,
            .group = groups.name(self.group),
            .order = groups.operations(self.group).len,
            .count = self.n,
            .count_extended = @as(u64, 2) * groups.operations(self.group).len * self.n,
            .generator = self.generator,
            .h = self.h,
            .seed = self.seed,
            .iteration = self.iteration,
            .removed_duplicates = self.removed_duplicates,
            .fixed_identity = self.fix_identity,
            .identity_index = self.identityIndex(),
            .base_count = self.base_count,
            .inserted_count = self.inserted_count,
            .triangulation_backend = "wgsl",
            .simplex_bound_deg = bound,
            .covering_excess = if (self.cover_result) |r| (if (bound) |b| finite(r.radius_deg / b - 1) else null) else null,
            .mesh_ratio = if (covering_current and measures_current and m.?.separation_deg > 0) finite(self.cover_result.?.radius_deg / m.?.separation_deg) else null,
            .covering_radius_deg = if (self.cover_result) |r| finite(r.radius_deg) else null,
            .covering_current = self.cover_result != null and self.cover_iteration == self.iteration,
            .covering_iteration = if (self.cover_result != null) self.cover_iteration else @as(?u32, null),
            .covering_witness = if (self.cover_result) |r| @as(?[4]f32, r.witness) else null,
            .covering_domain_bound_deg = if (self.cover_result) |r| finite(r.domain_bound_deg) else null,
            .covering_shell_points = if (self.cover_result) |r| @as(?u32, r.shell_points) else null,
            .covering_fence_points = if (self.cover_result) |r| @as(?u32, r.fence_points) else null,
            .covering_tetrahedra = if (self.cover_result) |r| @as(?u32, r.tetrahedra) else null,
            .covering_certificate_probes = if (self.cover_result) |r| @as(?u32, r.certificate_probes) else null,
            .covering_exact_predicates = if (self.cover_result) |r| @as(?u64, r.exact_predicates) else null,
            .covering_rounds = if (self.cover_result) |r| @as(?u32, r.rounds) else null,
            .measured = m != null,
            // Energy and separation describe this iteration (else an earlier one: covering
            // runs measure them on a time budget).
            .measures_current = measures_current,
            .measures_iteration = if (m != null) @as(?u32, self.measure_iteration) else null,
            .energy_ratios_current = self.full_measures and measures_current,
            // Relaxation forces within this many mean orientation radii (null: all pairs);
            // energy_neighbourhood: the same for the reported energy.
            .relax_neighbourhood = if (self.neighbourhood > 0) @as(?f64, self.neighbourhood) else null,
            .point_separation_iteration = self.point_separation_iteration,
            .point_covering_iteration = self.point_covering_iteration,
            .energy_neighbourhood = if (m != null and self.measure_neighbourhood > 0) @as(?f64, self.measure_neighbourhood) else null,
            .ratio_limit = self.ratioLimit(),
            .precision = if (m == null) "Not measured" else if (!measures_current) "Energy and separation from an earlier iteration" else if (self.full_measures) "All measures current" else if (self.n > self.ratioLimit()) "Energy and separation current; this sample exceeds the ratio limit" else "Energy and separation current; refresh energy ratios",
            .energy = if (m) |x| finite(x.quey_energy) else null,
            .quey_energy = if (m) |x| finite(x.quey_energy) else null,
            .riesz_1 = if (m) |x| finite(x.riesz_1) else null,
            .riesz_2 = if (m) |x| finite(x.riesz_2) else null,
            .riesz_3 = if (m) |x| finite(x.riesz_3) else null,
            .green_ratio = if (m) |x| finite(x.green_ratio) else null,
            .log_ratio = if (m) |x| finite(x.log_ratio) else null,
            .separation_deg = if (m) |x| finite(x.separation_deg) else null,
            .step_deg = self.step_deg,
            .force_rms = if (m) |x| finite(x.force_rms) else finite(self.force_rms),
            .covering_optimizer_steps = self.opt.steps,
            .covering_optimizer_kicks = self.opt.kicks,
            .covering_optimizer_best_deg = if (self.opt.active) finite(self.opt.best_radius) else null,
            .covering_optimizer_jobs = self.opt.jobs,
            .covering_optimizer_improved = self.opt.improved,
            .covering_optimizer_skipped = self.opt.skipped,
            .covering_optimizer_local_rounds = self.opt.local_rounds,
            .covering_optimizer_mismatches = self.opt.mismatches,
            .covering_optimizer_local_error = self.opt.local_error,
            .covering_optimizer_skip_reasons = self.opt.skip_reasons,
            .covering_optimizer_overflows = self.opt.overflows,
            .covering_optimizer_full_steps = self.opt.full_steps,
            .covering_optimizer_live_builds = self.opt.live_builds,
            .covering_optimizer_live_failures = self.opt.live_failures,
            .covering_optimizer_phase_ms = self.opt.times,
            .adapter = self.dev.info(),
        }, .{}, &w) catch {};
        self.text[w.end] = 0;
        return @ptrCast(&self.text);
    }
};
fn finite(x: f64) ?f64 {
    return if (std.math.isFinite(x)) x else null;
}
const LocalParams = extern struct { n: u32, rows: u32, jobs: u32, iterations: u32, band: f32, top: f32, grow: f32, shrink: f32, ring: u32 = 0, fixed: u32 = 0, seed: u32 = 0, zero: u32 = 0, value: f32 = 0, floor: f32 = 0, pad1: [2]f32 = @splat(0), lower: [4]f32 = @splat(0), grid: [4]u32 = @splat(0) };
/// Covering-optimization run: the best sample so far and the kick schedule.
/// Keys of radius at most `limit` in exactly one of two sorted cap sets (vertices only;
/// radii are compared through the largest).
fn keyDifference(x: []const covering.CapKey, y: []const covering.CapKey, limit: f32) u64 {
    var i: usize = 0;
    var j: usize = 0;
    var differ: u64 = 0;
    while (true) {
        while (i < x.len and x[i].rho > limit) i += 1;
        while (j < y.len and y[j].rho > limit) j += 1;
        if (i == x.len and j == y.len) break;
        if (j == y.len or (i < x.len and lessKey(x[i].v, y[j].v))) {
            differ += 1;
            i += 1;
        } else if (i == x.len or lessKey(y[j].v, x[i].v)) {
            differ += 1;
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    return differ;
}
fn lessKey(a: [4]u32, b: [4]u32) bool {
    for (a, b) |x, y| if (x != y) return x < y;
    return false;
}
const CoverOpt = struct {
    best: gpu.Buf = .{},
    start: gpu.Buf = .{},
    best_radius: f64 = std.math.inf(f64),
    /// Lowest radius of the current descent; the stall count measures progress
    /// against it, so each kicked sample is compacted to its own plateau.
    reference: f64 = std.math.inf(f64),
    /// Engine iteration this run state belongs to; any other motion restarts the run.
    iteration: u32 = 0,
    stall: u32 = 0,
    steps: u32 = 0,
    kicks: u32 = 0,
    jobs: u32 = 0,
    improved: u32 = 0,
    /// Jobs skipped (incomplete stars or gathers), accepted local rounds, and with
    /// the check variant, jobs whose local maximum disagreed with the exact pass.
    skipped: u32 = 0,
    local_rounds: u32 = 0,
    mismatches: u32 = 0,
    local_error: f32 = 0,
    skip_reasons: [8]u32 = @splat(0),
    /// Steps in which some job had more caps or free sources than its slots.
    overflows: u32 = 0,
    /// Full-pass steps left after local jobs were mostly skipped, and their total.
    fallback: u32 = 0,
    full_steps: u32 = 0,
    active: bool = false,
    /// The live covering pass local steps update in place (covering.Live), and how many
    /// were built. While one is kept, sources are not folded into the zone (a fold would
    /// renumber their images); `unfolded` says q needs a fold before the run ends.
    live: covering.Live = .{},
    live_on: bool = false,
    live_builds: u32 = 0,
    /// Live updates that could not complete (the live covering was built again).
    live_failures: u32 = 0,
    unfolded: bool = false,
    /// Energy measures of this run: when the last one ended and what it cost (ms).
    energy_at: f64 = 0,
    energy_ms: ?f64 = null,
    /// Development: milliseconds per phase of local steps (caps with their wait, recording
    /// the jobs and live update, the step's wait).
    times: [3]f64 = @splat(0),
};
// Measured on O/T/I at N around 1000 (docs/methods.md): initial trust radius and
// candidate band as fractions of the covering radius, solver iterations,
// selection rounds, stall steps before a kick, relative gain that resets the
// stall count, kick RMS angle, post-kick trust.
const cover_box: f32 = 0.15;
const cover_band: f32 = 0.3;
const cover_solver_iterations: u32 = 400;
const cover_rounds: u32 = 4;
const cover_patience: u32 = 150;
const cover_progress: f64 = 5e-4;
const cover_kick: f32 = 0.04;
const cover_kick_trust: f32 = 0.03;
// Shell reach in units of the covering radius (a job gathers within 2.5 R), and
// local LP rounds per job and step.
const cover_shell_reach: f64 = 1.75;
// Live shell margin in units of the covering radius: jobs need 2.21 R plus this step's
// 0.1 R move; the rest is room for the radius to grow before a new build.
const cover_live_reach: f64 = 2.5;
/// Largest sample whose Riesz, Green and log ratios Refresh computes.
/// Pair evaluations (n^2 |K|) up to which Refresh computes the Riesz, Green and log
/// ratios: a few seconds on a discrete GPU (T up to about 64,000 orientations).
const full_measure_pairs: f64 = 5e10;
const cover_local_rounds: u32 = 6;
const cover_local_solver_iterations: u32 = 120;
// Full-pass steps after a step whose local jobs were mostly skipped.
const cover_fallback_steps: u32 = 50;
