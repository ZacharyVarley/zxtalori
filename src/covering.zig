//! GPU symmetry shell, parallel DT, circumcaps and independent domain certificate.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const dt = @import("gpu_delaunay.zig");
const gao_dt = @import("gao_delaunay.zig");
const groups = @import("groups.zig");
const shaders = @import("shaders");
const A = std.mem.Allocator;
const Params = extern struct { n: u32, seeds: u32, first: u32, cap: u32, dim: u32 = 0, zero: u32 = 0, budget: u32 = 0, pad: u32 = 0, lower: [4]f32 = @splat(0), scale: [4]f32 = @splat(0) };
pub const Mesh = struct {
    vertices: [][4]f32 = &.{},
    tets: [][4]u32 = &.{},
    pub fn deinit(self: *Mesh, a: A) void {
        a.free(self.vertices);
        a.free(self.tets);
        self.* = .{};
    }
};
/// One linearized cap: four source ids, then per vertex (d rho / d left move, rho).
pub const Row = extern struct { src: [4]u32, grad: [4][4]f32 };
/// Cap rows within `band` of the largest stay on the GPU in `keep` (caller-owned).
/// `shell`, when given, also receives the shell sites and their grid index.
pub const Rows = struct { band: f32, keep: *gpu.Buf, count: *u32, shell: ?*Shell = null };
/// Per-source results read back with the caps (2n words in `host`): the S3 radius bits of
/// the largest cap each source's orbit touches, then its incident tetrahedra (its
/// Voronoi cell has 2 + that / 2 faces). `buf` is caller-owned scratch.
pub const Local = struct { buf: *gpu.Buf, host: []u32 };
/// Stereographic shell sites and their uniform-grid index, caller-owned: `index`
/// holds the grid heads (dim^3), next pointers (count), then each site's origin
/// (count), source + n * (2 * operator + negative).
pub const Shell = struct {
    sites: gpu.Buf = .{},
    index: gpu.Buf = .{},
    count: u32 = 0,
    dim: u32 = 0,
    lower: [3]f32 = @splat(0),
    scale: f32 = 0,
    margin: f64 = 0,
    pub fn release(self: *Shell, g: *gpu.Gpu) void {
        g.release(&self.sites);
        g.release(&self.index);
        self.* = .{};
    }
};
pub const Result = struct { radius_deg: f64, domain_bound_deg: f64, witness: [4]f32, shell_points: u32, fence_points: u32 = 0, tetrahedra: u32, certificate_probes: u32, exact_predicates: u64, rounds: u32 };
/// Fence band beyond the shell margin m, out to (1 + fence_width) m: real images that
/// bound the empty spheres of outer shell sites without being DT seeds themselves.
/// Omitted while the last Gao pass was degenerate (`Gpu.gao_degenerate`).
const fence_width = 0.5;
/// `Gpu.delaunay_mode` bit that omits the fence (diagnostic comparison only).
pub const no_fence: u32 = 1 << 20;
const code = shaders.dt_common ++ shaders.dt_math ++ shaders.dt_double ++ shaders.covering;
fn check(g: *gpu.Gpu) !void {
    g.submit();
    try g.dev.wait();
    if (g.dev.takeError() != null) return error.CoveringGpuValidation;
}
fn unordered(key: u32) f32 {
    return @bitCast(if (key & 0x80000000 != 0) key ^ 0x80000000 else ~key);
}
/// A covering pass kept on the GPU: the shell (sites, their origins, its grid), the shell
/// certificate and the Gao stars of the shell. `caps` reads the covering radius from the
/// current triangulation.
///
/// A live build (`build` with full) also keeps the triangulation current as sources move
/// (`beginMove`, `finishMove`; shaders/live.wgsl, gao_mark.wgsl, gao_live.wgsl). Its shell
/// and fence images keep stable slots; each move rewrites the slots of the moved sources'
/// images, gives images that entered either band a free slot and frees the slots of images
/// that left both, lists the grid again from the current positions, and recomputes exactly
/// the stars those changes can alter, each exact (every face certified). So the shell is
/// always every image within the margin m of the zone at its current position, and every
/// cap centred in the zone with radius at most m is exact, with no state carried between
/// steps beyond the triangulation itself.
pub const Live = struct {
    n: u32 = 0,
    order: u32 = 0,
    count: u32 = 0,
    fence: u32 = 0,
    margin: f64 = 0,
    lower: [3]f32 = @splat(0),
    upper: [3]f32 = @splat(0),
    halfspaces: gpu.Buf = .{},
    info: gpu.Buf = .{},
    sites: gpu.Buf = .{},
    origin: gpu.Buf = .{},
    heads: gpu.Buf = .{},
    next: gpu.Buf = .{},
    dim: u32 = 0,
    scale: f32 = 0,
    /// Live builds: slots (shell and fence images, free ones), the fence band, the box
    /// every position must stay in, and live.wgsl's and gao_mark.wgsl's buffers.
    listed: bool = false,
    capacity: u32 = 0,
    fenced: bool = false,
    fence_margin: f64 = 0,
    box_lower: [3]f32 = @splat(0),
    box_upper: [3]f32 = @splat(0),
    links: gpu.Buf = .{},
    /// The moved sources of a step, listed ([0] count).
    moved: gpu.Buf = .{},
    scratch: gpu.Buf = .{},
    flags: gpu.Buf = .{},
    pending: gpu.Buf = .{},
    launch: gpu.Buf = .{},
    /// The last move's summary counters, then scratch[0..4] and the wide count; they
    /// land at the caller's wait.
    move_stats: [24]u32 = @splat(0),
    move_status: [5]u32 = @splat(0),
    /// Development: stars the live kernel certified, then ones it gave up on for vertex
    /// or triangle capacity, candidates, topology, and certification (gao_main redoes them).
    live_stats: [5]u64 = @splat(0),
    gao: gao_dt.Live = .{},
    /// Output tetrahedra of the current triangulation, exact predicates so far, rounds.
    tets: u32 = 0,
    cold: u64 = 0,
    rounds: u32 = 0,
    probes: u32 = 0,
    subdivisions: u32 = 0,
    pub fn release(self: *Live, g: *gpu.Gpu) void {
        for ([_]*gpu.Buf{ &self.halfspaces, &self.info, &self.sites, &self.origin, &self.heads, &self.next, &self.flags, &self.links, &self.moved, &self.scratch, &self.pending, &self.launch }) |b| g.release(b);
        self.gao.release(g);
        self.* = .{};
    }

    /// A live build's triangulation: the sorted shell and fence sites become stable slots
    /// (spare ones free), the grid is listed, the stars are computed, and their reaches.
    fn buildLive(self: *Live, g: *gpu.Gpu, fenced: bool, fence_margin: f64, margin: f64) !gao_dt.Built {
        const total = self.count + self.fence;
        self.capacity = total + total / 4 + 1024;
        self.fenced = fenced;
        self.fence_margin = fence_margin;
        // Images in either band stay within the zone's neighbourhood; a margin of a fifth
        // of the box keeps them in it (else the move asks for a new build).
        var width: f32 = 0;
        for (self.lower, self.upper) |l, h| width = @max(width, h - l);
        for (&self.box_lower, &self.box_upper, self.lower, self.upper) |*bl, *bu, l, h| {
            bl.* = l - 0.2 * width;
            bu.* = h + 0.2 * width;
        }
        self.gao = try gao_dt.prepare(g, self.sites, total, self.capacity, self.count, self.box_lower, self.box_upper, margin, true);
        self.links = g.storage((@as(u64, self.n) + self.capacity) * 4);
        g.clear(self.links);
        self.scratch = g.storage((4 + 10 * @as(u64, self.capacity)) * 4);
        g.writeSlice(self.scratch, 0, u32, &[_]u32{ self.capacity - total, 0, 0, 0 });
        self.flags = g.storage((2 * @as(u64, self.capacity) + 2 + 1024) * 4);
        g.clear(self.flags);
        // Whole 32-byte records: the live kernel reads the list through gao_records.
        self.pending = g.storage(std.mem.alignForward(u64, (@as(u64, self.capacity) + 1) * 4, 32));
        self.launch = g.storage(48);
        var by_slot = g.storage(@as(u64, self.capacity) * 4);
        const lp = self.liveParams();
        g.dispatch(try g.pipeline("live/adopt", shaders.live, "live_adopt"), (self.capacity + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&lp)), self.gao.coordinates.at(2), self.origin.at(16), by_slot.at(17), self.links.at(18), self.scratch.at(19) });
        g.release(&self.origin);
        self.origin = by_slot;
        self.listed = true;
        self.gao.params.reserved |= 8;
        self.gao.params.seeds = 0x80000000;
        try self.gao.relist(g);
        const built = try gao_dt.solve(g, &self.gao);
        self.gao = built.live;
        try self.gao.reach(g, self.flags, self.pending, null, self.reachCap());
        return built;
    }
    /// The coordinates of the ids `emit` outputs: the shell sites, or a live build's slots.
    fn points(self: *const Live) gpu.Buf {
        return if (self.listed) self.gao.coordinates else self.sites;
    }
    fn capParams(self: *const Live) Params {
        return .{ .n = self.count, .seeds = self.tets, .first = self.order, .cap = self.subdivisions, .dim = self.dim, .budget = self.n, .lower = .{ self.lower[0], self.lower[1], self.lower[2], 0 }, .scale = .{ self.scale, @floatCast(self.margin), @floatCast(@sqrt(@as(f64, 3)) / @as(f64, @floatFromInt(self.subdivisions))), 0 } };
    }

    /// The covering radius of the current triangulation, its witness, and optionally the
    /// near-maximal cap rows, the shell grid and the mesh. error.CertificateFailed: the
    /// shell did not enclose every cap (build again with a wider margin).
    pub fn caps(self: *Live, a: A, g: *gpu.Gpu, core: gpu.Buf, operations: gpu.Buf, mesh: ?*Mesh, rows: ?Rows, local: ?Local) !Result {
        // Fresh cap results; words 11 and 12 keep the certificate.
        g.writeSlice(self.info, 8 * 4, u32, &[_]u32{ 0, 0xffffffff, 0 });
        g.writeSlice(self.info, 14 * 4, u32, &[_]u32{ @bitCast(@as(f32, 4)), 0, 0, 0, 0, 0 });
        var tets = try self.gao.emit(g, self.tets);
        defer g.release(&tets);
        var params = self.capParams();
        var caps_buf = g.storage(@as(u64, self.tets) * 32);
        defer g.release(&caps_buf);
        const cap_kernel = try g.pipeline("cover/caps", code, "circumcaps");
        const choose = try g.pipeline("cover/select", code, "select_cap");
        const witness_kernel = try g.pipeline("cover/witness", code, "check_witness");
        g.dispatch(cap_kernel, (self.tets + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), self.points().at(1), tets.at(10), operations.at(12), caps_buf.at(13), self.info.at(14) });
        g.dispatch(choose, (self.tets + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), caps_buf.at(13), self.info.at(14) });
        g.dispatch(witness_kernel, (self.n * self.order + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), operations.at(12), caps_buf.at(13), self.info.at(14), core.at(15) });
        if (local) |out| {
            g.ensure(out.buf, @as(u64, self.n) * 8);
            g.clearRange(out.buf.*, 0, @as(u64, self.n) * 8);
            g.dispatch(try g.pipeline("cover/local", code, "local_caps"), (self.tets + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), tets.at(10), caps_buf.at(13), self.origin.at(16), out.buf.at(19) });
            g.read(out.buf.*, 0, std.mem.sliceAsBytes(out.host[0 .. 2 * self.n]));
        }
        // Cap rows need only GPU-side results, so they are built before the one readback.
        var counter: gpu.Buf = .{};
        defer g.release(&counter);
        var out: gpu.Buf = .{};
        defer g.release(&out);
        var found: [4]u32 = @splat(0);
        if (rows) |request| {
            counter = g.storage(16);
            g.writeSlice(counter, 0, u32, &[_]u32{ 0, 0, 0, 0 });
            out = g.storage(@as(u64, self.tets) * @sizeOf(Row));
            var row_params = params;
            row_params.scale[3] = request.band;
            const row_kernel = try g.pipeline("cover/rows", code, "cap_rows_kernel");
            g.dispatch(row_kernel, (self.tets + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&row_params)), tets.at(10), operations.at(12), caps_buf.at(13), self.info.at(14), core.at(15), self.origin.at(16), out.at(17), counter.at(18) });
            g.read(counter, 0, std.mem.asBytes(&found));
        }
        var state: [24]u32 = undefined;
        g.read(self.info, 0, std.mem.asBytes(&state));
        try check(g);
        if (state[11] != 0) return error.CertificateFailed;
        if (state[10] == 0 or state[9] == 0xffffffff) return error.NoCoveringCircumcaps;
        var witness: [4]f32 = undefined;
        for (&witness, state[16..20]) |*w, bits| w.* = @bitCast(bits);
        const proposed: f32 = @bitCast(state[8]);
        const chord2: f32 = @bitCast(state[14]);
        const radius = 2 * std.math.asin(@min(1, @sqrt(@as(f64, chord2)) * 0.5));
        if (@abs(radius - proposed) > 3e-6) return error.CoveringWitnessMismatch;
        if (rows) |request| {
            request.count.* = found[0];
            g.release(request.keep);
            request.keep.* = out;
            out = .{};
            if (request.shell) |kept| {
                kept.release(g);
                const cells = @as(u64, self.dim) * self.dim * self.dim;
                if (self.listed) {
                    // The slots' grid at their current positions (free slots unlisted).
                    kept.index = g.storage((cells + 2 * @as(u64, self.capacity)) * 4);
                    const lp = self.liveParams();
                    g.dispatch(try g.pipeline("live/index_clear", shaders.live, "live_index_clear"), @intCast((cells + 127) / 128), 1, 1, &.{ g.uniform(0, std.mem.asBytes(&lp)), kept.index.at(20) });
                    g.dispatch(try g.pipeline("live/index", shaders.live, "live_index"), (self.capacity + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&lp)), self.gao.coordinates.at(2), self.origin.at(17), kept.index.at(20) });
                    kept.* = .{ .index = kept.index, .count = self.capacity, .dim = self.dim, .lower = self.lower, .scale = self.scale, .margin = self.margin };
                    return .{ .radius_deg = radius * 360 / std.math.pi, .domain_bound_deg = self.margin * 360 / std.math.pi, .witness = witness, .shell_points = self.count, .fence_points = self.fence, .tetrahedra = self.tets, .certificate_probes = state[12], .exact_predicates = self.cold, .rounds = self.rounds };
                }
                kept.sites = g.storage(@as(u64, self.count) * 16);
                g.copy(self.sites, 0, kept.sites, 0, @as(u64, self.count) * 16);
                kept.index = g.storage((cells + 2 * @as(u64, self.count)) * 4);
                g.copy(self.heads, 0, kept.index, 0, cells * 4);
                g.copy(self.next, 0, kept.index, cells * 4, @as(u64, self.count) * 4);
                g.copy(self.origin, 0, kept.index, (cells + self.count) * 4, @as(u64, self.count) * 4);
                kept.* = .{ .sites = kept.sites, .index = kept.index, .count = self.count, .dim = self.dim, .lower = self.lower, .scale = self.scale, .margin = self.margin };
            }
        }
        if (mesh) |destination| {
            const vertices = try a.alloc([4]f32, self.count);
            errdefer a.free(vertices);
            var mesh_tets = try a.alloc([4]u32, self.tets);
            errdefer a.free(mesh_tets);
            g.read(self.sites, 0, std.mem.sliceAsBytes(vertices));
            g.read(tets, 0, std.mem.sliceAsBytes(mesh_tets));
            try check(g);
            // The shell's part of the DT: drop tetrahedra on fence sites.
            var kept_tets: usize = 0;
            for (mesh_tets) |t| if (@max(@max(t[0], t[1]), @max(t[2], t[3])) < self.count) {
                mesh_tets[kept_tets] = t;
                kept_tets += 1;
            };
            mesh_tets = try a.realloc(mesh_tets, kept_tets);
            destination.deinit(a);
            destination.* = .{ .vertices = vertices, .tets = mesh_tets };
        }
        return .{ .radius_deg = radius * 360 / std.math.pi, .domain_bound_deg = self.margin * 360 / std.math.pi, .witness = witness, .shell_points = self.count, .fence_points = self.fence, .tetrahedra = self.tets, .certificate_probes = state[12], .exact_predicates = self.cold, .rounds = self.rounds };
    }

    /// Updates the triangulation for the sources `moved` flags (one u32 per source), now
    /// at their positions in `core`: `beginMove`, a wait, then `finishMove`.
    pub fn move(self: *Live, g: *gpu.Gpu, core: gpu.Buf, operations: gpu.Buf, moved: gpu.Buf) !void {
        try self.beginMove(g, core, operations, moved);
        try check(g);
        try self.finishMove();
    }
    /// Records the update for the sources `moved` flags (nonzero per source) without
    /// waiting on the GPU; its counters land at the caller's next wait, after which
    /// `finishMove` checks them.
    pub fn beginMove(self: *Live, g: *gpu.Gpu, core: gpu.Buf, operations: gpu.Buf, moved: gpu.Buf) !void {
        const lp = self.liveParams();
        // Uniforms are made at each dispatch: a submit recycles the staged ones.
        // The moved sources' images: slots rewritten, taken and freed, changes listed.
        g.clearRange(self.scratch, 8, 8);
        const shell_params = Params{ .n = self.n, .seeds = self.capacity, .first = self.order, .cap = @intCast(self.halfspaces.size / 16), .dim = @intFromBool(self.fenced), .lower = .{ @floatCast(@sin(self.margin)), @floatCast(@sin(self.fence_margin)), self.box_lower[0], self.box_lower[1] }, .scale = .{ self.box_lower[2], self.box_upper[0], self.box_upper[1], self.box_upper[2] } };
        g.ensure(&self.moved, (@as(u64, self.n) + 1) * 4);
        g.clearRange(self.moved, 0, 4);
        g.dispatch(try g.pipeline("live/moved", shaders.live, "live_moved"), (self.n + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&lp)), moved.at(21), self.moved.at(22) });
        g.dispatch(try g.pipeline("live/launch_moved", shaders.live, "live_launch_moved"), 1, 1, 1, &.{ self.launch.at(11), self.moved.at(22) });
        g.submit();
        g.dispatchIndirect(try g.pipeline("cover/live_images", shaders.shell, "shell_live_images"), self.launch, 32, &.{ g.uniform(0, std.mem.asBytes(&shell_params)), self.halfspaces.at(11), operations.at(12), core.at(15), self.moved.at(18), self.gao.coordinates.at(21), self.origin.at(22), self.links.at(23), self.scratch.at(24) });
        g.dispatch(try g.pipeline("live/release", shaders.live, "live_release"), 1, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&lp)), self.scratch.at(19) });
        try self.gao.relist(g);
        // The stars the changes can alter, recomputed once each (indirect sizes need a pass
        // boundary on native wgpu: each launch is followed by a submit).
        g.clearRange(self.pending, 0, 4);
        g.dispatch(try g.pipeline("live/launch_changes", shaders.live, "live_launch_changes"), 1, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&lp)), self.launch.at(11), self.scratch.at(19) });
        g.submit();
        try self.gao.mark(g, self.flags, self.pending, self.scratch, self.launch, self.reachCap());
        g.dispatch(try g.pipeline("live/launch", shaders.live, "live_launch"), 1, 1, 1, &.{ self.pending.at(10), self.launch.at(11) });
        g.submit();
        try self.gao.restar(g, .{ self.pending, self.launch }, &self.move_stats);
        try self.gao.reach(g, self.flags, self.pending, self.launch, self.reachCap());
        g.read(self.scratch, 0, std.mem.asBytes(self.move_status[0..4]));
        g.read(self.flags, (2 * @as(u64, self.capacity) + 1) * 4, std.mem.asBytes(self.move_status[4..5]));
    }
    /// Checks the counters of the last `beginMove` (after a wait) and takes its count.
    /// error.LiveRebuild: the free slots, the change list, the wide-star list or the box
    /// ran out (or a star outgrew its local mesh); the caller builds again.
    pub fn finishMove(self: *Live) !void {
        if (self.move_status[3] != 0 or self.move_status[4] > 1024) return error.LiveRebuild;
        try gao_dt.Live.restarResult(&self.move_stats);
        for (&self.live_stats, self.move_stats[19..24]) |*total, x| total.* += x;
        self.tets = self.move_stats[0];
        self.cold += self.move_stats[7];
        self.rounds += 1;
    }
    fn liveParams(self: *const Live) LiveParams {
        return .{ .n = self.n, .total = self.count + self.fence, .capacity = self.capacity, .count = self.count, .dim = self.dim, .lower = .{ self.lower[0], self.lower[1], self.lower[2], self.scale } };
    }
    /// The reach beyond which a star is listed as wide: six grid cells.
    fn reachCap(self: *const Live) f32 {
        return 6 / self.gao.params.scale[0];
    }

    /// Counted caps of the current triangulation (centred in the zone, radius at most the
    /// margin) as sorted vertex orbit codes (source + n * image) with the radius; for
    /// validating live updates against a new build.
    pub fn capSet(self: *Live, a: A, g: *gpu.Gpu, operations: gpu.Buf) ![]CapKey {
        g.writeSlice(self.info, 8 * 4, u32, &[_]u32{ 0, 0xffffffff, 0 });
        g.writeSlice(self.info, 14 * 4, u32, &[_]u32{ @bitCast(@as(f32, 4)), 0, 0, 0, 0, 0 });
        var tets = try self.gao.emit(g, self.tets);
        defer g.release(&tets);
        var params = self.capParams();
        var caps_buf = g.storage(@as(u64, self.tets) * 32);
        defer g.release(&caps_buf);
        g.dispatch(try g.pipeline("cover/caps", code, "circumcaps"), (self.tets + 127) / 128, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), self.points().at(1), tets.at(10), operations.at(12), caps_buf.at(13), self.info.at(14) });
        const t = try a.alloc([4]u32, self.tets);
        defer a.free(t);
        const c = try a.alloc([8]f32, self.tets);
        defer a.free(c);
        const o = try a.alloc(u32, if (self.listed) self.capacity else self.count + self.fence);
        defer a.free(o);
        g.read(tets, 0, std.mem.sliceAsBytes(t));
        g.read(caps_buf, 0, std.mem.sliceAsBytes(c));
        g.read(self.origin, 0, std.mem.sliceAsBytes(o));
        try check(g);
        var keys: std.ArrayList(CapKey) = .empty;
        errdefer keys.deinit(a);
        for (t, c) |tet, cap| {
            if (!(cap[4] > 0)) continue;
            var v: [4]u32 = undefined;
            for (tet, &v) |site, *code_out| code_out.* = o[site];
            std.mem.sort(u32, &v, {}, std.sort.asc(u32));
            try keys.append(a, .{ .v = v, .rho = cap[4] });
        }
        const out = try keys.toOwnedSlice(a);
        std.mem.sort(CapKey, out, {}, CapKey.less);
        return out;
    }
};
pub const CapKey = struct {
    v: [4]u32,
    rho: f32,
    pub fn less(_: void, x: CapKey, y: CapKey) bool {
        for (x.v, y.v) |a, b| if (a != b) return a < b;
        return x.rho < y.rho;
    }
};
const LiveParams = extern struct { n: u32, total: u32, capacity: u32, count: u32, dim: u32, pad: [3]u32 = @splat(0), lower: [4]f32 };

/// A new live covering pass at shell margin `margin` (S3 radians).
/// The zone's bisector halfspaces (normalized), as the shell kernels take them.
pub fn zonePlanes(a: A, group: groups.Group) !std.ArrayList([4]f32) {
    var planes: std.ArrayList([4]f32) = .empty;
    errdefer planes.deinit(a);
    for (groups.operations(group), 0..) |op, i| for ([_]f64{ 1, -1 }) |sign| {
        if (i == 0 and sign == 1) continue;
        var normal = [4]f64{ 1 - sign * op[0], -sign * op[1], -sign * op[2], -sign * op[3] };
        var length: f64 = 0;
        for (normal) |x| length += x * x;
        length = @sqrt(length);
        var value: [4]f32 = undefined;
        for (&normal, &value) |x, *y| y.* = @floatCast(x / length);
        try planes.append(a, value);
    };
    return planes;
}

/// The shell's images (no fence) at `margin`: of every source, or with `moved` only of
/// the sources it flags (nonzero per source). Coordinates are bit-identical to the
/// shell build's. Caller frees.
pub const Image = struct { code: u32, x: [3]f32 };
pub fn shellImages(a: A, g: *gpu.Gpu, core: gpu.Buf, n: u32, operations: gpu.Buf, group: groups.Group, margin: f64, moved: ?gpu.Buf, expected: u32) ![]Image {
    var planes = try zonePlanes(a, group);
    defer planes.deinit(a);
    var halfspaces = g.storage(planes.items.len * 16);
    defer g.release(&halfspaces);
    g.writeSlice(halfspaces, 0, [4]f32, planes.items);
    var counter = g.storage(16);
    defer g.release(&counter);
    var dummy: gpu.Buf = .{};
    defer g.release(&dummy);
    if (moved == null) dummy = g.storage(16);
    const order: u32 = @intCast(groups.operations(group).len);
    const kernel = try g.pipeline("cover/shell_images", shaders.shell, "shell_images");
    const shape = gpu.Gpu.flat((n * order * 2 + 127) / 128);
    var capacity: u32 = @max(expected, 64);
    for (0..3) |_| {
        var out = g.storage(@as(u64, capacity) * 16);
        defer g.release(&out);
        g.writeSlice(counter, 0, u32, &[_]u32{ 0, 0, 0, 0 });
        const params = Params{ .n = n, .seeds = 0, .first = order, .cap = @intCast(planes.items.len), .budget = @intFromBool(moved != null), .pad = capacity, .lower = .{ @floatCast(@sin(margin)), 0, 0, 0 } };
        g.dispatch(kernel, shape[0], shape[1], 1, &.{ g.uniform(0, std.mem.asBytes(&params)), halfspaces.at(11), operations.at(12), core.at(15), (moved orelse dummy).at(18), out.at(19), counter.at(20) });
        var count: [4]u32 = undefined;
        g.read(counter, 0, std.mem.asBytes(&count));
        try check(g);
        if (count[0] > capacity) {
            capacity = count[0];
            continue;
        }
        const raw = try a.alloc([4]u32, count[0]);
        defer a.free(raw);
        if (count[0] > 0) {
            g.read(out, 0, std.mem.sliceAsBytes(raw));
            try check(g);
        }
        const images = try a.alloc(Image, count[0]);
        for (raw, images) |r, *image| image.* = .{ .code = r[0], .x = .{ @bitCast(r[1]), @bitCast(r[2]), @bitCast(r[3]) } };
        return images;
    }
    return error.ShellImagesOverflow;
}

/// `full`: a live build (exact whole stars and stable slots, for `beginMove`; see Live).
pub fn build(a: A, g: *gpu.Gpu, core: gpu.Buf, n: u32, operations: gpu.Buf, group: groups.Group, margin_in: f64, full: bool) !Live {
    if (n < 4) return error.TooFewCoveringSites;
    const ops = groups.operations(group);
    const order: u32 = @intCast(ops.len);
    var planes = try zonePlanes(a, group);
    defer planes.deinit(a);
    var live = Live{ .n = n, .order = order, .margin = margin_in };
    errdefer live.release(g);
    live.halfspaces = g.storage(planes.items.len * 16);
    g.writeSlice(live.halfspaces, 0, [4]f32, planes.items);
    live.info = g.storage(24 * 4);
    var blocks: gpu.Buf = .{};
    defer g.release(&blocks);
    const margin = margin_in;
    var capacity: u32 = @max(n + 4, n * @min(order * 2, 8));
    for (0..4) |_| {
        if (g.cancel_flag) |flag| if (flag.load(.acquire)) return error.Cancelled;
        if (@as(u64, capacity + 4) * 16 > g.lim.max_storage_binding) return error.CoveringShellTooLarge;
        g.ensure(&live.sites, @as(u64, capacity + 4) * 16);
        g.ensure(&live.origin, @as(u64, capacity + 4) * 4);
        // Words 16..19: the witness centre; 20..23 spare.
        var state = [_]u32{ n, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0, 0, 0xffffffff, 0, 0, 0, 0, @bitCast(@as(f32, 4)), 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        g.writeSlice(live.info, 0, u32, &state);
        // Same 64-byte layout as ShellParams (n,capacity,order,planes,fence,...).
        // On lattice-like input the fence's co-spherical images add exact predicates
        // (T h=8: 14k against 1.3k), costing more than its shorter searches save.
        const fenced = g.delaunay_mode & no_fence == 0 and !g.gao_degenerate;
        const fence_margin = @min(1.569, margin * (1 + fence_width));
        const shell_params = Params{ .n = n, .seeds = capacity, .first = order, .cap = @intCast(planes.items.len), .dim = @intFromBool(fenced), .lower = .{ @floatCast(@sin(margin)), @floatCast(@sin(fence_margin)), 0, 0 } };
        const shell_count = try g.pipeline("cover/shell_count", shaders.shell, "shell_count");
        const shell_scan = try g.pipeline("cover/shell_scan", shaders.shell, "shell_scan");
        const shell = try g.pipeline("cover/shell", shaders.shell, "shell_build");
        const shape = gpu.Gpu.flat((n * order * 2 + 127) / 128);
        g.ensure(&blocks, @as(u64, shape[0]) * shape[1] * 8);
        g.dispatch(shell_count, shape[0], shape[1], 1, &.{ g.uniform(0, std.mem.asBytes(&shell_params)), live.halfspaces.at(11), operations.at(12), core.at(15), blocks.at(17) });
        g.dispatch(shell_scan, 1, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&shell_params)), live.info.at(14), blocks.at(17) });
        g.dispatch(shell, shape[0], shape[1], 1, &.{ g.uniform(0, std.mem.asBytes(&shell_params)), live.sites.at(1), live.halfspaces.at(11), operations.at(12), live.info.at(14), core.at(15), live.origin.at(16), blocks.at(17) });
        g.read(live.info, 0, std.mem.asBytes(&state));
        try check(g);
        if (state[1] != 0) {
            capacity = @max(capacity * 2, state[0] + state[13] + 4);
            continue;
        }
        // Shell sites [0, count), then fence sites [count, count + fence).
        live.count = state[0];
        live.fence = state[13];
        for (0..3) |k| {
            live.lower[k] = unordered(state[k + 2]);
            live.upper[k] = unordered(state[k + 5]);
        }
        const index = try dt.makeIndex(g, live.sites, live.count, live.lower, live.upper);
        live.heads = index.heads;
        live.next = index.next;
        live.dim = index.dim;
        live.scale = index.scale;
        const subdivisions: u32 = @intFromFloat(@max(4, @ceil(@sqrt(@as(f64, 3)) / (0.4 * margin))));
        if (subdivisions > 512) return error.CoveringCertificateTooLarge;
        live.subdivisions = subdivisions;
        const params = live.capParams();
        const probes = 4 * (subdivisions + 1) * (subdivisions + 1) * (subdivisions + 1);
        const certificate = try g.pipeline("cover/certificate", code, "certify_shell");
        const probe_shape = gpu.Gpu.flat((probes + 127) / 128);
        g.dispatch(certificate, probe_shape[0], probe_shape[1], 1, &.{ g.uniform(0, std.mem.asBytes(&params)), live.sites.at(1), live.heads.at(2), live.next.at(3), live.halfspaces.at(11), operations.at(12), live.info.at(14) });
        // The certificate's verdict is read with the caps (Live.caps), not on its own. A
        // triangulation that fails on a shell that fails the certificate also asks for a
        // wider margin.
        const built = (if (full) live.buildLive(g, fenced, fence_margin, margin) else gao_dt.build(g, live.sites, live.count + live.fence, live.count, live.lower, live.upper, margin, false)) catch |err| {
            g.read(live.info, 0, std.mem.asBytes(&state));
            try check(g);
            if (state[11] != 0) return error.CertificateFailed;
            return err;
        };
        live.gao = built.live;
        live.tets = built.stats[0];
        live.cold = built.cold;
        live.rounds = built.rounds;
        return live;
    }
    return error.CoveringShellRetryLimit;
}

pub fn measure(a: A, g: *gpu.Gpu, core: gpu.Buf, n: u32, operations: gpu.Buf, group: groups.Group, hint: ?f64, mesh: ?*Mesh, rows: ?Rows, local: ?Local) !Result {
    if (n < 4) return error.TooFewCoveringSites;
    const order: u32 = @intCast(groups.operations(group).len);
    const density = @as(f64, @floatFromInt(@as(u64, n) * order));
    var margin = @min(1.569, 2.5 * std.math.cbrt(3 * std.math.pi / (4 * density)));
    if (hint) |radius| margin = @max(margin, @min(1.569, radius * 1.5));
    for (0..9) |_| {
        var live = build(a, g, core, n, operations, group, margin, false) catch |err| {
            if (err != error.CertificateFailed) return err;
            if (margin >= 1.569) return error.CoveringDomainNotEnclosed;
            margin = @min(1.569, margin * 1.5);
            continue;
        };
        defer live.release(g);
        return live.caps(a, g, core, operations, mesh, rows, local) catch |err| {
            if (err != error.CertificateFailed) return err;
            if (margin >= 1.569) return error.CoveringDomainNotEnclosed;
            margin = @min(1.569, margin * 1.5);
            continue;
        };
    }
    return error.CoveringShellRetryLimit;
}
