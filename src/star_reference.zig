//! Exhaustive small-input Delaunay one-ring oracle, outside the product engine.
//! This deliberately scans every candidate and uses exact predicates throughout.
//! It is a correctness experiment for at most 64 sites, not a scalable scheduler.
const std = @import("std");
const exact = @import("predicates.zig");
pub const Point = exact.Point;
pub const Tet = [4]u32;
pub const Result = struct { tets: []Tet, candidates: u64, insphere_tests: u64 };

pub fn oneRing(allocator: std.mem.Allocator, points: []const Point, ids: []const u64, seed: u32) !Result {
    if (points.len < 4 or points.len > 64 or seed >= points.len or ids.len != points.len) return error.InvalidInput;
    for (points, 0..) |point, i| {
        for (point) |value| if (!std.math.isFinite(value)) return error.NonFinite;
        for (i + 1..points.len) |j| {
            if (ids[i] == ids[j]) return error.DuplicateId;
            if (point[0] == points[j][0] and point[1] == points[j][1] and point[2] == points[j][2]) return error.DuplicateCoordinates;
        }
    }
    var tets: std.ArrayList(Tet) = .empty;
    errdefer tets.deinit(allocator);
    var candidates: u64 = 0;
    var tests: u64 = 0;
    for (0..points.len) |a| {
        if (a == seed) continue;
        for (a + 1..points.len) |b| {
            if (b == seed) continue;
            for (b + 1..points.len) |c| {
                if (c == seed) continue;
                candidates += 1;
                var tet: Tet = .{ seed, @intCast(a), @intCast(b), @intCast(c) };
                const orientation = try exact.orient3d(.{ points[seed], points[a], points[b], points[c] });
                if (orientation == .zero) continue;
                if (orientation == .negative) std.mem.swap(u32, &tet[1], &tet[2]);
                var p: [5]Point = undefined;
                var global_ids: [5]u64 = undefined;
                for (tet, 0..) |index, i| {
                    p[i] = points[index];
                    global_ids[i] = ids[index];
                }
                var empty = true;
                for (points, 0..) |query, j| {
                    if (std.mem.indexOfScalar(u32, &tet, @intCast(j)) != null) continue;
                    p[4] = query;
                    global_ids[4] = ids[j];
                    tests += 1;
                    const side = try exact.insphereSos(p, global_ids);
                    if (side == .zero) return error.UnresolvedPredicate;
                    // The defining tetrahedron is positive; raw determinant is
                    // negative inside. Exact cospheres use the same global IDs.
                    if (side == .negative) {
                        empty = false;
                        break;
                    }
                }
                if (empty) try tets.append(allocator, tet);
            }
        }
    }
    if (tets.items.len == 0) return error.LowerDimensional;
    return .{ .tets = try tets.toOwnedSlice(allocator), .candidates = candidates, .insphere_tests = tests };
}
