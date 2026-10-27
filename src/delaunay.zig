//! Incremental 3D DT for grid augmentation. Float filters, exact binary32 cold
//! predicates and global-ID lift ties; no jitter or unresolved-site fallback.
//! This CPU scheduler is deliberately separate from the future GPU scheduler.
const std = @import("std");
const exact = @import("predicates.zig");
const A = std.mem.Allocator;
pub const Point = exact.Point;
pub const Tet = [4]u32;
const none = std.math.maxInt(u32);
pub const Cell = struct { v: Tet, nb: [4]u32 = @splat(none), live: bool = true, seen: u32 = 0, bad: u32 = 0 };
pub const Boundary = struct { face: [3]u32, outside: u32, slot: usize };
pub const Link = struct { cell: u32, slot: usize };

fn det3(m: [3][3]f64) f64 {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
// Shewchuk's fast predicates (Robust Adaptive Floating-Point Geometric Predicates,
// 1997) with his forward error bounds: f32 inputs are exact doubles, and the bounds
// cover the rounded differences and products. Undecided cases take the exact path.
const epsilon = 0x1p-53;
const orient_bound = (7.0 + 56.0 * epsilon) * epsilon;
const sphere_bound = (16.0 + 224.0 * epsilon) * epsilon;
/// Sign of (b-a) dot ((c-a) cross (d-a)): minus Shewchuk's orient3d(a, b, c, d).
pub fn orient(p: [4]Point) !i8 {
    var a: [3]f64 = undefined;
    var b: [3]f64 = undefined;
    var c: [3]f64 = undefined;
    for (0..3) |k| {
        a[k] = @as(f64, p[0][k]) - p[3][k];
        b[k] = @as(f64, p[1][k]) - p[3][k];
        c[k] = @as(f64, p[2][k]) - p[3][k];
    }
    const bc = b[0] * c[1];
    const cb = c[0] * b[1];
    const ca = c[0] * a[1];
    const ac = a[0] * c[1];
    const ab = a[0] * b[1];
    const ba = b[0] * a[1];
    const d = a[2] * (bc - cb) + b[2] * (ca - ac) + c[2] * (ab - ba);
    const permanent = (@abs(bc) + @abs(cb)) * @abs(a[2]) + (@abs(ca) + @abs(ac)) * @abs(b[2]) + (@abs(ab) + @abs(ba)) * @abs(c[2]);
    if (@abs(d) > orient_bound * permanent) return if (d < 0) 1 else -1;
    return @intFromEnum(try exact.orient3d(p));
}
/// Whether `query` is inside the sphere of positively oriented `v`: the sign of
/// det[p_i - q, |p_i - q|^2] (Shewchuk's insphere), negative inside.
pub fn inside(points: []const Point, v: Tet, query: u32) !bool {
    var x: [4]f64 = undefined;
    var y: [4]f64 = undefined;
    var z: [4]f64 = undefined;
    var lift: [4]f64 = undefined;
    const e = points[query];
    for (v, 0..) |index, i| {
        x[i] = @as(f64, points[index][0]) - e[0];
        y[i] = @as(f64, points[index][1]) - e[1];
        z[i] = @as(f64, points[index][2]) - e[2];
        lift[i] = x[i] * x[i] + y[i] * y[i] + z[i] * z[i];
    }
    const aby = x[0] * y[1];
    const bay = x[1] * y[0];
    const bcy = x[1] * y[2];
    const cby = x[2] * y[1];
    const cdy = x[2] * y[3];
    const dcy = x[3] * y[2];
    const day = x[3] * y[0];
    const ady = x[0] * y[3];
    const acy = x[0] * y[2];
    const cay = x[2] * y[0];
    const bdy = x[1] * y[3];
    const dby = x[3] * y[1];
    const ab = aby - bay;
    const bc = bcy - cby;
    const cd = cdy - dcy;
    const da = day - ady;
    const ac = acy - cay;
    const bd = bdy - dby;
    const abc = z[0] * bc - z[1] * ac + z[2] * ab;
    const bcd = z[1] * cd - z[2] * bd + z[3] * bc;
    const cda = z[2] * da + z[3] * ac + z[0] * cd;
    const dab = z[3] * ab + z[0] * bd + z[1] * da;
    const d = (lift[3] * abc - lift[2] * dab) + (lift[1] * cda - lift[0] * bcd);
    const az = @abs(z[0]);
    const bz = @abs(z[1]);
    const cz = @abs(z[2]);
    const dz = @abs(z[3]);
    const permanent = ((@abs(cdy) + @abs(dcy)) * bz + (@abs(dby) + @abs(bdy)) * cz + (@abs(bcy) + @abs(cby)) * dz) * lift[0] +
        ((@abs(day) + @abs(ady)) * cz + (@abs(acy) + @abs(cay)) * dz + (@abs(cdy) + @abs(dcy)) * az) * lift[1] +
        ((@abs(aby) + @abs(bay)) * dz + (@abs(bdy) + @abs(dby)) * az + (@abs(day) + @abs(ady)) * bz) * lift[2] +
        ((@abs(bcy) + @abs(cby)) * az + (@abs(cay) + @abs(acy)) * bz + (@abs(aby) + @abs(bay)) * cz) * lift[3];
    if (@abs(d) > sphere_bound * permanent) return d < 0;
    return (try exact.insphereSos(.{ points[v[0]], points[v[1]], points[v[2]], points[v[3]], points[query] }, .{ v[0], v[1], v[2], v[3], query })) == .negative;
}
pub fn face(v: Tet, skip: usize) [3]u32 {
    var f: [3]u32 = undefined;
    var at: usize = 0;
    for (v, 0..) |x, i| {
        if (i != skip) {
            f[at] = x;
            at += 1;
        }
    }
    return f;
}
pub fn key(f: [3]u32) [3]u32 {
    var sorted = f;
    if (sorted[0] > sorted[1]) std.mem.swap(u32, &sorted[0], &sorted[1]);
    if (sorted[1] > sorted[2]) std.mem.swap(u32, &sorted[1], &sorted[2]);
    if (sorted[0] > sorted[1]) std.mem.swap(u32, &sorted[0], &sorted[1]);
    return sorted;
}
pub fn locate(points: []const Point, cells: []const Cell, start: u32, query: u32) !u32 {
    var current = start;
    for (0..@max(128, cells.len * 2)) |_| {
        const c = cells[current];
        var crossed = false;
        for (0..4) |k| {
            var p = [4]Point{ points[c.v[0]], points[c.v[1]], points[c.v[2]], points[c.v[3]] };
            p[k] = points[query];
            if (try orient(p) < 0) {
                if (c.nb[k] == none) return error.PointOutsideSuperTetrahedron;
                current = c.nb[k];
                crossed = true;
                break;
            }
        }
        if (!crossed) return current;
    }
    return error.LocationDidNotConverge;
}
pub fn triangulate(a: A, input: []const Point) ![]Tet {
    // Very thin represented hull facets can have circumcentres far beyond a
    // fixed enclosing tetrahedron. Grow that enclosure only when the exact
    // convex-hull completion check asks for it; the DT algorithm is unchanged.
    for ([_]f32{ 0x1p20, 0x1p40, 0x1p80 }) |scale| {
        return triangulateEnclosed(a, input, scale) catch |err| switch (err) {
            error.IncompleteConvexHull => continue,
            else => return err,
        };
    }
    return error.IncompleteConvexHull;
}
fn triangulateEnclosed(a: A, input: []const Point, scale: f32) ![]Tet {
    if (input.len < 4 or input.len > 1000000) return error.InvalidCount;
    var points = try a.alloc(Point, input.len + 4);
    defer a.free(points);
    @memcpy(points[0..input.len], input);
    var bound: f32 = 1;
    var unique = std.AutoHashMap([3]u32, void).init(a);
    defer unique.deinit();
    for (input) |p| {
        var bits: [3]u32 = undefined;
        for (p, 0..) |x, k| {
            if (!std.math.isFinite(x)) return error.NonFinite;
            bound = @max(bound, @abs(x));
            bits[k] = if (x == 0) 0 else @bitCast(x);
        }
        const entry = try unique.getOrPut(bits);
        if (entry.found_existing) return error.DuplicateCoordinates;
    }
    bound *= scale;
    if (!std.math.isFinite(bound)) return error.CoordinateRange;
    const n: u32 = @intCast(input.len);
    points[n..][0..4].* = .{ .{ -bound, -bound, -bound }, .{ bound, bound, -bound }, .{ bound, -bound, bound }, .{ -bound, bound, bound } };
    var cells: std.ArrayList(Cell) = .empty;
    defer cells.deinit(a);
    var initial = Tet{ n, n + 1, n + 2, n + 3 };
    if (try orient(.{ points[n], points[n + 1], points[n + 2], points[n + 3] }) < 0) std.mem.swap(u32, &initial[0], &initial[1]);
    try cells.append(a, .{ .v = initial });
    var free: std.ArrayList(u32) = .empty;
    defer free.deinit(a);
    var queue: std.ArrayList(u32) = .empty;
    defer queue.deinit(a);
    var bad: std.ArrayList(u32) = .empty;
    defer bad.deinit(a);
    var rim: std.ArrayList(Boundary) = .empty;
    defer rim.deinit(a);
    var fresh: std.ArrayList(u32) = .empty;
    defer fresh.deinit(a);
    var links = std.AutoHashMap([3]u32, Link).init(a);
    defer links.deinit();
    const order = try a.alloc(u32, n);
    defer a.free(order);
    for (order, 0..) |*x, i| x.* = @intCast(i);
    var rng = std.Random.DefaultPrng.init(193);
    rng.random().shuffle(u32, order);
    var last: u32 = 0;
    for (order, 1..) |query, generation_usize| {
        const generation: u32 = @intCast(generation_usize);
        const containing = try locate(points, cells.items, last, query);
        queue.clearRetainingCapacity();
        bad.clearRetainingCapacity();
        rim.clearRetainingCapacity();
        fresh.clearRetainingCapacity();
        links.clearRetainingCapacity();
        try queue.append(a, containing);
        cells.items[containing].seen = generation;
        var at: usize = 0;
        while (at < queue.items.len) : (at += 1) {
            const index = queue.items[at];
            const c = cells.items[index];
            if (!try inside(points, c.v, query)) continue;
            cells.items[index].bad = generation;
            try bad.append(a, index);
            for (c.nb) |neighbor| if (neighbor != none and cells.items[neighbor].seen != generation) {
                cells.items[neighbor].seen = generation;
                try queue.append(a, neighbor);
            };
        }
        if (bad.items.len == 0) return error.EmptyCavity;
        for (bad.items) |index| {
            const c = cells.items[index];
            for (c.nb, 0..) |neighbor, k| if (neighbor == none or cells.items[neighbor].bad != generation) {
                var slot: usize = 0;
                if (neighbor != none) {
                    slot = std.mem.indexOfScalar(u32, &cells.items[neighbor].nb, index) orelse return error.BrokenAdjacency;
                }
                try rim.append(a, .{ .face = face(c.v, k), .outside = neighbor, .slot = slot });
            };
        }
        for (bad.items) |index| {
            cells.items[index].live = false;
            try free.append(a, index);
        }
        for (rim.items) |b| {
            var v = Tet{ b.face[0], b.face[1], b.face[2], query };
            const orientation = try orient(.{ points[v[0]], points[v[1]], points[v[2]], points[v[3]] });
            if (orientation == 0) return error.DegenerateCavity;
            if (orientation < 0) std.mem.swap(u32, &v[0], &v[1]);
            const index = free.pop() orelse blk: {
                const index: u32 = @intCast(cells.items.len);
                try cells.append(a, undefined);
                break :blk index;
            };
            cells.items[index] = .{ .v = v };
            cells.items[index].nb[3] = b.outside;
            if (b.outside != none) cells.items[b.outside].nb[b.slot] = index;
            try fresh.append(a, index);
        }
        for (fresh.items) |index| for (0..3) |slot| {
            const f = key(face(cells.items[index].v, slot));
            if (links.fetchRemove(f)) |other| {
                cells.items[index].nb[slot] = other.value.cell;
                cells.items[other.value.cell].nb[other.value.slot] = index;
            } else try links.put(f, .{ .cell = index, .slot = slot });
        };
        if (links.count() != 0) return error.OpenCavity;
        last = fresh.items[0];
    }
    var result: std.ArrayList(Tet) = .empty;
    errdefer result.deinit(a);
    for (cells.items) |c| if (c.live and std.mem.max(u32, &c.v) < n) {
        // Finite enclosing vertices must not truncate the actual convex hull.
        // Check each exposed output face against every represented input site.
        for (c.nb, 0..) |neighbor, slot| {
            if (neighbor != none and std.mem.max(u32, &cells.items[neighbor].v) < n) continue;
            var p = [4]Point{ points[c.v[0]], points[c.v[1]], points[c.v[2]], points[c.v[3]] };
            for (input) |query| {
                p[slot] = query;
                if (try orient(p) < 0) return error.IncompleteConvexHull;
            }
        }
        try result.append(a, c.v);
    };
    if (result.items.len == 0) return error.LowerDimensional;
    return result.toOwnedSlice(a);
}

test "fast predicate filters agree with the exact predicates" {
    var rng = std.Random.DefaultPrng.init(7);
    const random = rng.random();
    var points: [5]Point = undefined;
    for (0..20000) |round| {
        // Generic points, and nearly degenerate ones (a scaled lattice plus noise).
        for (&points) |*p| for (p) |*x| {
            x.* = if (round % 2 == 0) random.float(f32) - 0.5 else @as(f32, @floatFromInt(random.intRangeAtMost(i32, -2, 2))) * 0.1 + (random.float(f32) - 0.5) * 1e-6;
        };
        const o = try orient(points[0..4].*);
        try std.testing.expectEqual(@as(i8, @intFromEnum(try exact.orient3d(points[0..4].*))), o);
        if (o <= 0) continue;
        const want = (try exact.insphereSos(points, .{ 0, 1, 2, 3, 4 })) == .negative;
        try std.testing.expectEqual(want, try inside(&points, .{ 0, 1, 2, 3 }, 4));
    }
}
