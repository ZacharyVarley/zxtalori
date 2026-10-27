//! Fill a grid from the largest empty spherical Delaunay circumcaps.
//! Only representatives and a boundary shell are stored, never the full orbit.
const std = @import("std");
const dt = @import("delaunay.zig");
const gpu_dt = @import("gpu_delaunay.zig");
const gpu = @import("gpu/gpu.zig");
const groups = @import("groups.zig");
const A = std.mem.Allocator;
const Q = [4]f64;
const Candidate = struct { q: Q, radius: f64 };
const Site = struct { p: dt.Point, next: ?usize };
fn dot(a: Q, b: Q) f64 {
    var result: f64 = 0;
    for (a, b) |x, y| result += x * y;
    return result;
}
fn normalized(a: Q) Q {
    var q = a;
    const n = @sqrt(dot(q, q));
    for (&q) |*x| x.* /= n;
    return q;
}
fn mul(a: Q, b: Q) Q {
    return .{ a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0] };
}
fn quaternion(p: dt.Point) Q {
    const r = @as(f64, p[0]) * p[0] + @as(f64, p[1]) * p[1] + @as(f64, p[2]) * p[2];
    return .{ (1 - r) / (1 + r), 2 * @as(f64, p[0]) / (1 + r), 2 * @as(f64, p[1]) / (1 + r), 2 * @as(f64, p[2]) / (1 + r) };
}
fn determinant(m: [3][3]f64) f64 {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
fn circumcap(p: [4]Q) ?Candidate {
    var e: [3]Q = undefined;
    for (0..3) |i| for (0..4) |k| {
        e[i][k] = p[i + 1][k] - p[0][k];
    };
    var c: Q = undefined;
    for (0..4) |k| {
        var minor: [3][3]f64 = undefined;
        for (0..3) |i| {
            var at: usize = 0;
            for (0..4) |j| if (j != k) {
                minor[i][at] = e[i][j];
                at += 1;
            };
        }
        c[k] = determinant(minor) * @as(f64, if (k % 2 == 0) 1 else -1);
    }
    const norm = @sqrt(dot(c, c));
    if (norm < 1e-16) return null;
    for (&c) |*x| x.* /= norm;
    var height: f64 = 0;
    for (p) |q| height += dot(c, q) * 0.25;
    // Select the bounded stereographic ball, whose cap excludes the south pole.
    if (c[0] + height < 0) {
        for (&c) |*x| x.* = -x.*;
        height = -height;
    }
    if (c[0] <= 0 or height <= 0 or height >= 1) return null;
    for (p) |q| if (@abs(dot(c, q) - height) > 1e-9) return null;
    return .{ .q = c, .radius = std.math.acos(height) };
}
fn inZone(c: Q, ops: []const Q, tolerance: f64) bool {
    for (ops) |op| if (@abs(dot(c, op)) > c[0] + tolerance) return false;
    return c[0] >= 0;
}
/// Orientations of both signs under a spatial hash of their vector part: a cap query
/// then touches the few cells within its chord, not every orientation.
const Cover = struct {
    cell: f64,
    points: std.ArrayList(Q) = .empty,
    next: std.ArrayList(usize) = .empty,
    heads: std.AutoHashMap([3]i32, usize),
    fn init(a: A, cell: f64) Cover {
        return .{ .cell = cell, .heads = std.AutoHashMap([3]i32, usize).init(a) };
    }
    fn deinit(self: *Cover, a: A) void {
        self.points.deinit(a);
        self.next.deinit(a);
        self.heads.deinit();
    }
    fn key(self: Cover, q: Q) [3]i32 {
        var k: [3]i32 = undefined;
        for (0..3) |i| k[i] = @intFromFloat(@floor(q[i + 1] / self.cell));
        return k;
    }
    fn add(self: *Cover, a: A, q: Q) !void {
        for ([_]f64{ 1, -1 }) |sign| {
            var signed = q;
            for (&signed) |*x| x.* *= sign;
            const entry = try self.heads.getOrPut(self.key(signed));
            try self.next.append(a, if (entry.found_existing) entry.value_ptr.* else none);
            entry.value_ptr.* = self.points.items.len;
            try self.points.append(a, signed);
        }
    }
    /// Whether some stored orientation has |dot(query, q)| > threshold.
    fn near(self: Cover, query: Q, threshold: f64) bool {
        const chord = @sqrt(@max(0, 2 - 2 * threshold)) * (1 + 1e-9) + 1e-12;
        var lo: [3]i32 = undefined;
        var hi: [3]i32 = undefined;
        for (0..3) |i| {
            lo[i] = @intFromFloat(@floor((query[i + 1] - chord) / self.cell));
            hi[i] = @intFromFloat(@floor((query[i + 1] + chord) / self.cell));
        }
        var x = lo[0];
        while (x <= hi[0]) : (x += 1) {
            var y = lo[1];
            while (y <= hi[1]) : (y += 1) {
                var z = lo[2];
                while (z <= hi[2]) : (z += 1) {
                    var at = self.heads.get(.{ x, y, z }) orelse continue;
                    while (at != none) : (at = self.next.items[at]) {
                        if (dot(query, self.points.items[at]) > threshold) return true;
                    }
                }
            }
        }
        return false;
    }
};
const none = std.math.maxInt(usize);
fn appendSite(a: A, sites: *std.ArrayList(Site), cells: *std.AutoHashMap([3]i32, usize), p: dt.Point) !void {
    const tolerance = 1e-6;
    var key: [3]i32 = undefined;
    for (p, 0..) |x, k| key[k] = @intFromFloat(@floor(x / tolerance));
    for (0..3) |dx| for (0..3) |dy| for (0..3) |dz| {
        var found: ?usize = cells.get(.{ key[0] + @as(i32, @intCast(dx)) - 1, key[1] + @as(i32, @intCast(dy)) - 1, key[2] + @as(i32, @intCast(dz)) - 1 });
        while (found) |i| {
            var distance: f64 = 0;
            for (p, sites.items[i].p) |x, y| distance += (@as(f64, x) - y) * (@as(f64, x) - y);
            if (distance < tolerance * tolerance) return;
            found = sites.items[i].next;
        }
    };
    const entry = try cells.getOrPut(key);
    const previous = if (entry.found_existing) entry.value_ptr.* else null;
    entry.value_ptr.* = sites.items.len;
    try sites.append(a, .{ .p = p, .next = previous });
}
pub fn centres(a: A, g: *gpu.Gpu, values: []const f32, group: groups.Group, wanted: u32) ![]f32 {
    if (values.len < 16 or wanted == 0) return error.InvalidCount;
    const ops_f32 = groups.operations(group);
    const ops = try a.alloc(Q, ops_f32.len);
    defer a.free(ops);
    for (ops_f32, ops) |op, *q| q.* = normalized(.{ op[0], op[1], op[2], op[3] });
    const core = try a.alloc(Q, values.len / 4);
    defer a.free(core);
    for (core, 0..) |*q, i| q.* = normalized(.{ values[4 * i], values[4 * i + 1], values[4 * i + 2], values[4 * i + 3] });
    var planes: std.ArrayList(Q) = .empty;
    defer planes.deinit(a);
    for (ops, 0..) |op, i| for ([_]f64{ 1, -1 }) |sign| {
        if (i == 0 and sign == 1) continue;
        try planes.append(a, normalized(.{ 1 - sign * op[0], -sign * op[1], -sign * op[2], -sign * op[3] }));
    };
    const density = @as(f64, @floatFromInt(core.len * ops.len));
    var threshold = @min(0.8, 2.5 * std.math.cbrt(3 * std.math.pi / (4 * density)));
    for (0..4) |_| {
        const sine = @sin(threshold);
        var sites: std.ArrayList(Site) = .empty;
        defer sites.deinit(a);
        var hash = std.AutoHashMap([3]i32, usize).init(a);
        defer hash.deinit();
        for (core) |q| {
            if (g.cancel_flag) |flag| if (flag.load(.acquire)) return error.Cancelled;
            try appendSite(a, &sites, &hash, .{ @floatCast(q[1] / (1 + q[0])), @floatCast(q[2] / (1 + q[0])), @floatCast(q[3] / (1 + q[0])) });
            var rim = false;
            for (planes.items) |plane| if (dot(q, plane) <= sine) {
                rim = true;
                break;
            };
            if (!rim) continue;
            for (ops, 0..) |op, k| for ([_]f64{ 1, -1 }) |sign| {
                if (k == 0 and sign == 1) continue;
                var image = mul(q, op);
                for (&image) |*x| x.* *= sign;
                var near = true;
                for (planes.items) |plane| if (dot(image, plane) < -sine) {
                    near = false;
                    break;
                };
                if (!near or image[0] < -0.9) continue;
                try appendSite(a, &sites, &hash, .{ @floatCast(image[1] / (1 + image[0])), @floatCast(image[2] / (1 + image[0])), @floatCast(image[3] / (1 + image[0])) });
            };
        }
        const points = try a.alloc(dt.Point, sites.items.len);
        defer a.free(points);
        for (sites.items, points) |site, *p| p.* = site.p;
        const tets = try gpu_dt.triangulateHostRegional(a, g, points, threshold);
        defer a.free(tets);
        var candidates: std.ArrayList(Candidate) = .empty;
        defer candidates.deinit(a);
        for (tets) |t| {
            const c = circumcap(.{ quaternion(points[t[0]]), quaternion(points[t[1]]), quaternion(points[t[2]]), quaternion(points[t[3]]) }) orelse continue;
            if (!inZone(c.q, ops, 1e-7)) continue;
            try candidates.append(a, c);
        }
        std.mem.sort(Candidate, candidates.items, {}, struct {
            fn less(_: void, x: Candidate, y: Candidate) bool {
                return x.radius > y.radius;
            }
        }.less);
        if (candidates.items.len == 0) return error.NoDelaunayCentres;
        if (candidates.items[0].radius > threshold * 0.98) {
            threshold = @min(1.2, threshold * 1.5);
            continue;
        }
        var selected: std.ArrayList(Q) = .empty;
        defer selected.deinit(a);
        // Every original orbit and every cap centre already chosen, in one hash.
        var occupied = Cover.init(a, threshold);
        defer occupied.deinit(a);
        for (core) |q| try occupied.add(a, q);
        candidate: for (candidates.items) |c| {
            if (g.cancel_flag) |flag| if (flag.load(.acquire)) return error.Cancelled;
            // Co-spherical subdivisions have the same centre; insert it once.
            if (occupied.near(c.q, 1 - 1e-12)) continue :candidate;
            const height = @cos(c.radius);
            // Verify against every original orbit, streamed operator by operator.
            // This rejects projection roundoff/slivers without manufacturing caps.
            for (ops) |op| {
                var inverse = op;
                for (inverse[1..]) |*x| x.* = -x.*;
                const query = mul(c.q, inverse);
                // Adjacent cells of nearly co-spherical grids can have distinct
                // circumcentres only 1e-3 radians apart. Once a hole is filled,
                // do not fill another cap already occupied by that insertion.
                // Continue down the radius ordering, retriangulating if needed.
                if (occupied.near(query, height + 3e-7)) continue :candidate;
            }
            try selected.append(a, c.q);
            try occupied.add(a, c.q);
            if (selected.items.len == wanted) break;
        }
        if (selected.items.len == 0) return error.NoEmptyDelaunayCentres;
        const output = try a.alloc(f32, selected.items.len * 4);
        for (selected.items, 0..) |q, i| for (q, 0..) |x, k| {
            output[i * 4 + k] = @floatCast(x);
        };
        return output;
    }
    return error.ShellDidNotClose;
}
