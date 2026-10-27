//! Near-field Quey forces (Quey, Villani & Maurice 2018, section 2.3): each source feels
//! only the orbit points within an S3 angle r = delta * d_r of it, d_r the mean orientation
//! radius; delta = 20 holds about 7.5k neighbours, at a cost linear in N. Pair forces fade
//! smoothly over the outer half of r (see `taper`). Without the far field, density
//! differences larger than r even out more slowly than with all pairs: from a random start
//! the relaxed separation and covering radius take about 4x the steps to match. On the GPU, the covering shell kernels stream every orbit image within r
//! of the zone (the only images a zone source can feel), counting sorts place images in a
//! grid of the stereographic chart and sources in Morton order, and workgroups sum batches
//! of 64 nearby sources' forces over shares of the image runs their box can reach
//! (near_force in pairs.wgsl), reduced as all-pairs partial rows are. Nothing is read
//! back per step; the first pass sizes the buffers and the chart box once.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const covering = @import("covering.zig");
const groups = @import("groups.zig");
const shaders = @import("shaders");
const A = std.mem.Allocator;

const ShellParams = extern struct { n: u32, capacity: u32, order: u32, planes: u32, fence: u32 = 0, zero: u32 = 0, pad0: u32 = 0, pad1: u32 = 0, config: [4]f32, padding: [4]f32 = @splat(0) };
const SortParams = extern struct { count: u32, capacity: u32, dim: u32, flags: u32, lower: [4]f32 };
const NearParams = extern struct { slices: u32, sdim: u32, n: u32, first: u32 = 0, count: u32 = 0, pad: [3]u32 = @splat(0), lower: [4]f32, cut: [4]f32 };
/// The sources' Morton grid (a power of two per side).
const tdim = 32;
/// Workgroups a pass aims for: batches of 64 sources times slices of their site rows.
/// Short workgroups balance far better than long ones.
const target_groups = 65536;
const max_slices = 64;
/// Partial rows (48 bytes each) a chunk of the pass may hold.
const partial_rows = 1 << 19;

pub const Near = struct {
    a: A,
    group: groups.Group,
    delta: f64,
    /// The window on pair forces starts falling at this fraction of the cutoff (1: none).
    /// With a hard cutoff (Quey's) the force jumps as points cross the edge and relaxed
    /// separations come out about 4% smaller than with all pairs; from half the cutoff
    /// they match.
    taper: f64 = 0.5,
    n: u32 = 0,
    order: u32 = 0,
    /// Cutoff (S3 radians) and its chord.
    radius: f64 = 0,
    capacity: u32 = 0,
    ready: bool = false,
    planes: u32 = 0,
    halfspaces: gpu.Buf = .{},
    info: gpu.Buf = .{},
    blocks: gpu.Buf = .{},
    sites: gpu.Buf = .{},
    origin: gpu.Buf = .{},
    sorted_sites: gpu.Buf = .{},
    images: gpu.Buf = .{},
    codes: gpu.Buf = .{},
    site_offsets: gpu.Buf = .{},
    site_cursors: gpu.Buf = .{},
    points: gpu.Buf = .{},
    sorted_sources: gpu.Buf = .{},
    source_offsets: gpu.Buf = .{},
    source_cursors: gpu.Buf = .{},
    partial: gpu.Buf = .{},
    lower: [3]f32 = @splat(0),
    width: f32 = 1,
    sdim: u32 = 1,
    slices: u32 = 1,
    /// Sources per chunk (a multiple of 64).
    chunk: u32 = 64,
    /// The last pass's shell count and overflow flag (they land at the caller's wait).
    status: [2]u32 = @splat(0),

    pub fn release(self: *Near, g: *gpu.Gpu) void {
        for ([_]*gpu.Buf{ &self.halfspaces, &self.info, &self.blocks, &self.sites, &self.origin, &self.sorted_sites, &self.images, &self.codes, &self.site_offsets, &self.site_cursors, &self.points, &self.sorted_sources, &self.source_offsets, &self.source_cursors, &self.partial }) |b| g.release(b);
        self.* = .{ .a = self.a, .group = self.group, .delta = self.delta, .taper = self.taper };
    }

    /// Records the shell of every orbit image within the cutoff of the zone.
    fn shell(self: *Near, g: *gpu.Gpu, q: gpu.Buf, ops: gpu.Buf) !void {
        var state = [_]u32{ self.n, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0, 0, 0xffffffff, 0, 0, 0, 0, @bitCast(@as(f32, 4)), 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        g.writeSlice(self.info, 0, u32, &state);
        // A source may sit a rounding outside the zone after its fold.
        const params = ShellParams{ .n = self.n, .capacity = self.capacity, .order = self.order, .planes = self.planes, .config = .{ @floatCast(@sin(self.radius + 1e-4)), 0, 0, 0 } };
        const shape = gpu.Gpu.flat((self.n * self.order * 2 + 127) / 128);
        g.ensure(&self.blocks, @as(u64, shape[0]) * shape[1] * 8);
        g.dispatch(try g.pipeline("cover/shell_count", shaders.shell, "shell_count"), shape[0], shape[1], 1, &.{ g.uniform(0, std.mem.asBytes(&params)), self.halfspaces.at(11), ops.at(12), q.at(15), self.blocks.at(17) });
        g.dispatch(try g.pipeline("cover/shell_scan", shaders.shell, "shell_scan"), 1, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), self.info.at(14), self.blocks.at(17) });
        g.dispatch(try g.pipeline("cover/shell", shaders.shell, "shell_build"), shape[0], shape[1], 1, &.{ g.uniform(0, std.mem.asBytes(&params)), self.sites.at(1), self.halfspaces.at(11), ops.at(12), self.info.at(14), q.at(15), self.origin.at(16), self.blocks.at(17) });
    }

    /// Sizes the buffers and the chart box from one shell pass (one wait).
    fn setup(self: *Near, g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32, mean_radius: f64) !void {
        const a = self.a;
        self.release(g);
        self.n = n;
        self.order = order;
        self.radius = @min(1.5, self.delta * mean_radius);
        var planes = try covering.zonePlanes(a, self.group);
        defer planes.deinit(a);
        self.planes = @intCast(planes.items.len);
        self.halfspaces = g.storage(planes.items.len * 16);
        g.writeSlice(self.halfspaces, 0, [4]f32, planes.items);
        self.info = g.storage(24 * 4);
        var capacity: u32 = @max(n + 1024, n * 4);
        var state: [24]u32 = undefined;
        for (0..4) |_| {
            self.capacity = capacity;
            g.ensure(&self.sites, @as(u64, capacity) * 16);
            g.ensure(&self.origin, @as(u64, capacity) * 4);
            try self.shell(g, q, ops);
            g.read(self.info, 0, std.mem.asBytes(&state));
            g.submit();
            try g.dev.wait();
            if (g.dev.takeError() != null) return error.GpuValidation;
            if (state[1] == 0) break;
            capacity = @max(capacity * 2, state[0] + state[0] / 4);
        } else return error.NearShellTooLarge;
        // Room for the shell to change as the sources move.
        const count = state[0];
        if (count + count / 4 + 1024 > self.capacity) {
            self.capacity = count + count / 4 + 1024;
            g.release(&self.sites);
            g.release(&self.origin);
            self.sites = g.storage(@as(u64, self.capacity) * 16);
            self.origin = g.storage(@as(u64, self.capacity) * 4);
        }
        var lo: [3]f32 = undefined;
        var hi: [3]f32 = undefined;
        for (0..3) |k| {
            lo[k] = unordered(state[k + 2]);
            hi[k] = unordered(state[k + 5]);
        }
        var width: f32 = 1e-6;
        for (lo, hi) |l, h| width = @max(width, h - l);
        // Points that drift outside the box land in its edge cells: still found, only slower.
        for (&self.lower, lo) |*l, x| l.* = x - 0.05 * width;
        self.width = width * 1.1;
        self.sdim = @intFromFloat(std.math.clamp(@round(std.math.cbrt(@as(f64, @floatFromInt(count)) / 8)), 1, 128));
        const batches = (n + 63) / 64;
        self.slices = std.math.clamp((target_groups + batches - 1) / batches, 1, max_slices);
        const scells = @as(u64, self.sdim) * self.sdim * self.sdim;
        const tcells: u64 = tdim * tdim * tdim;
        self.sorted_sites = g.storage(@as(u64, self.capacity) * 16);
        self.images = g.storage(@as(u64, self.capacity) * 16);
        self.codes = g.storage(@as(u64, self.capacity) * 4);
        self.site_offsets = g.storage((scells + 1) * 4);
        self.site_cursors = g.storage((scells + 1) * 4);
        self.points = g.storage(@as(u64, n) * 16);
        self.sorted_sources = g.storage(@as(u64, n) * 16);
        self.source_offsets = g.storage((tcells + 1) * 4);
        self.source_cursors = g.storage((tcells + 1) * 4);
        self.chunk = @min((n + 63) / 64 * 64, @max(64, partial_rows / self.slices / 64 * 64));
        self.partial = g.storage(@as(u64, self.chunk) * self.slices * 48);
        self.ready = true;
    }

    fn sort(g: *gpu.Gpu, params: SortParams, input: gpu.Buf, info: gpu.Buf, offsets: gpu.Buf, cursors: gpu.Buf, sorted: gpu.Buf, items: u32) !void {
        const cells = params.dim * params.dim * params.dim;
        g.dispatch(try g.pipeline("near/clear", shaders.near, "near_clear"), (cells + 128) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), offsets.at(2) });
        g.dispatch(try g.pipeline("near/count", shaders.near, "near_count"), (items + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), input.at(1), offsets.at(2), info.at(5) });
        g.dispatch(try g.pipeline("near/prefix", shaders.near, "near_prefix"), 1, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), offsets.at(2), cursors.at(4) });
        g.dispatch(try g.pipeline("near/scatter", shaders.near, "near_scatter"), (items + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), input.at(1), sorted.at(3), cursors.at(4), info.at(5) });
        g.dispatch(try g.pipeline("near/order", shaders.near, "near_order"), (cells + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), offsets.at(2), sorted.at(3) });
    }

    /// Records the near-field force rows of the sources `q` into `rows` (one Row each), as
    /// pairs.forcePass does for all pairs; `mean_radius` is d_r for n * order points.
    pub fn pass(self: *Near, g: *gpu.Gpu, q: gpu.Buf, n: u32, ops: gpu.Buf, order: u32, mean_radius: f64, rows: gpu.Buf) !void {
        if (!self.ready or self.n != n or self.order != order) try self.setup(g, q, n, ops, order, mean_radius);
        try self.shell(g, q, ops);
        const box = [4]f32{ self.lower[0], self.lower[1], self.lower[2], @as(f32, @floatFromInt(self.sdim)) / self.width };
        const tbox = [4]f32{ self.lower[0], self.lower[1], self.lower[2], tdim / self.width };
        try sort(g, .{ .count = 0, .capacity = self.capacity, .dim = self.sdim, .flags = 1, .lower = box }, self.sites, self.info, self.site_offsets, self.site_cursors, self.sorted_sites, self.capacity);
        const image_params = [4]u32{ n, self.capacity, 0, 0 };
        g.dispatch(try g.pipeline("near/images", shaders.near, "near_images"), (self.capacity + 127) / 128, 1, 1, &.{ g.uniform(8, std.mem.asBytes(&image_params)), self.sorted_sites.at(3), self.info.at(5), q.at(6), ops.at(9), self.origin.at(10), self.images.at(11), self.codes.at(12) });
        const source_params = SortParams{ .count = n, .capacity = n, .dim = tdim, .flags = 2, .lower = tbox };
        g.dispatch(try g.pipeline("near/points", shaders.near, "near_points"), (n + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&source_params)), q.at(6), self.points.at(7) });
        try sort(g, source_params, self.points, self.info, self.source_offsets, self.source_cursors, self.sorted_sources, n);
        const chord = 2 * @sin(self.radius / 2);
        const pair_params = [8]u32{ n, order, 0, 0, 0, 0, 0, 0 };
        const force = try g.pipeline("pairs/near-force", shaders.pairs, "near_force");
        const reduce = try g.pipeline("pairs/near-reduce", shaders.pairs, "near_reduce");
        var first: u32 = 0;
        while (first < n) : (first += self.chunk) {
            const count = @min(self.chunk, n - first);
            const params = NearParams{ .slices = self.slices, .sdim = self.sdim, .n = n, .first = first, .count = count, .lower = box, .cut = .{ @floatCast(chord * chord), @floatCast(chord), @floatCast(self.taper * self.radius), @floatCast(1 / @max(1e-9, (1 - self.taper) * self.radius)) } };
            g.dispatch(force, (count + 63) / 64, self.slices, 1, &.{ g.uniform(0, std.mem.asBytes(&pair_params)), q.at(1), self.partial.at(3), g.uniform(4, std.mem.asBytes(&params)), self.images.at(8), self.site_offsets.at(9), self.codes.at(10), self.sorted_sources.at(11) });
            g.dispatch(reduce, (count + 63) / 64, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&pair_params)), g.uniform(4, std.mem.asBytes(&params)), self.partial.at(6), rows.at(7), self.sorted_sources.at(11) });
        }
        g.read(self.info, 0, std.mem.asBytes(&self.status));
    }
    /// Whether the last pass's shell outgrew its buffers (after the caller's wait); the
    /// next pass sizes them again.
    pub fn overflowed(self: *Near) bool {
        if (self.status[1] == 0) return false;
        self.ready = false;
        return true;
    }
};

fn unordered(key: u32) f32 {
    return @bitCast(if (key & 0x80000000 != 0) key ^ 0x80000000 else ~key);
}
