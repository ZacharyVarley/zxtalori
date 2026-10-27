//! Orbit-aware seam deduplication without storing a symmetry-expanded cloud.
const std = @import("std");
const tolerance: f64 = 2e-6;

/// Keeps the first of every orbit-equal group, in place. The first `unique` entries are
/// known to be distinct already: they are indexed without being checked.
pub fn compact(allocator: std.mem.Allocator, q: []f32, operations: []const [4]f32, unique: usize) !u32 {
    var cells = std.AutoHashMap([3]i32, usize).init(allocator);
    defer cells.deinit();
    const next = try allocator.alloc(usize, q.len / 4);
    defer allocator.free(next);
    var count: usize = 0;
    outer: for (0..q.len / 4) |i| {
        const source = q[i * 4 ..][0..4].*;
        for (operations) |op| {
            if (i < unique) break;
            const a: [4]f64 = .{ source[0], source[1], source[2], source[3] };
            const b: [4]f64 = .{ op[0], op[1], op[2], op[3] };
            var p: [4]f64 = .{
                a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
            };
            var norm: f64 = 0;
            for (p) |x| norm += x * x;
            norm = @sqrt(norm);
            for (&p) |*x| x.* /= norm;
            for ([_]f64{ 1, -1 }) |sign| {
                // Cells are twice the tolerance wide: a match lies in one of at most 8.
                const lo = cell(.{ p[1] * sign - tolerance, p[2] * sign - tolerance, p[3] * sign - tolerance });
                const hi = cell(.{ p[1] * sign + tolerance, p[2] * sign + tolerance, p[3] * sign + tolerance });
                var x = lo[0];
                while (x <= hi[0]) : (x += 1) {
                    var y = lo[1];
                    while (y <= hi[1]) : (y += 1) {
                        var z = lo[2];
                        while (z <= hi[2]) : (z += 1) {
                            var found = cells.get(.{ x, y, z }) orelse continue;
                            while (found != std.math.maxInt(usize)) {
                                var distance: f64 = 0;
                                for (0..4) |k| {
                                    const d = p[k] * sign - q[found * 4 + k];
                                    distance += d * d;
                                }
                                if (distance < tolerance * tolerance) continue :outer;
                                found = next[found];
                            }
                        }
                    }
                }
            }
        }
        @memmove(q[count * 4 ..][0..4], &source);
        const key = cell(.{ source[1], source[2], source[3] });
        const entry = try cells.getOrPut(key);
        next[count] = if (entry.found_existing) entry.value_ptr.* else std.math.maxInt(usize);
        entry.value_ptr.* = count;
        count += 1;
    }
    return @intCast(count);
}
fn cell(p: [3]f64) [3]i32 {
    const width = 2 * tolerance;
    return .{ @intFromFloat(@floor(p[0] / width)), @intFromFloat(@floor(p[1] / width)), @intFromFloat(@floor(p[2] / width)) };
}
