//! T/O/I source lattices and GPU measure-preserving transports.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const groups = @import("groups.zig");
const shaders = @import("shaders");

pub const Result = struct { quats: gpu.Buf, count: u32 };
const Params = extern struct { count: u32, group: u32, h: u32, pad: u32 = 0, beta: f32, p0: f32 = 0, p1: f32 = 0, p2: f32 = 0 };

fn cutIndex(h: u32) i32 {
    const wanted = (1.0 + @sqrt(@as(f64, 2))) * (@as(f64, @floatFromInt(h)) + 0.5);
    const lower: i32 = @intFromFloat(@floor(wanted));
    const odd = if (@mod(lower, 2) == 1) lower else lower - 1;
    return if (wanted - @as(f64, @floatFromInt(odd)) <= @as(f64, @floatFromInt(odd + 2)) - wanted) odd else odd + 2;
}

pub fn count(group: groups.Group, h: u32) !u32 {
    if (h < 1 or h > 128) return error.InvalidGridResolution;
    const id = @intFromEnum(group);
    if (id == 8) return (16 * h * h * h + 24 * h * h + 14 * h + 3) / 3;
    if (id != 9 and id != 10) return error.UnsupportedGroup;
    const bound: i32 = @intCast(h);
    const cut = cutIndex(h);
    var n: u32 = 0;
    var i = -bound;
    while (i <= bound) : (i += 1) {
        var j = -bound;
        while (j <= bound) : (j += 1) {
            var k = -bound;
            while (k <= bound) : (k += 1) {
                if (@mod(i + j + k, 2) != 0) continue;
                if (id == 9 and @as(i32, @intCast(@abs(i) + @abs(j) + @abs(k))) >= cut) continue;
                n += 1;
            }
        }
    }
    return n;
}

pub fn generate(allocator: std.mem.Allocator, g: *gpu.Gpu, group: groups.Group, h: u32) !Result {
    const n = try count(group, h);
    const id = @intFromEnum(group);
    const points = try allocator.alloc([4]f32, n);
    defer allocator.free(points);
    const bound: i32 = @intCast(if (id == 8) 2 * h else h);
    const step: f32 = @floatCast(if (id == 8) std.math.pow(f64, std.math.pi * std.math.pi / 16.0, 1.0 / 3.0) / @as(f64, @floatFromInt(2 * h + 1)) else std.math.pow(f64, std.math.pi, 2.0 / 3.0) / @as(f64, @floatFromInt(2 * h + 1)));
    const cut = cutIndex(h);
    var at: usize = 0;
    var i = -bound;
    while (i <= bound) : (i += 1) {
        var j = -bound;
        while (j <= bound) : (j += 1) {
            var k = -bound;
            while (k <= bound) : (k += 1) {
                if (@mod(i + j + k, 2) != 0) continue;
                const sum: i32 = @intCast(@abs(i) + @abs(j) + @abs(k));
                if (id == 8 and sum > bound) continue;
                if (id == 9 and sum >= cut) continue;
                points[at] = .{ @as(f32, @floatFromInt(i)) * step, @as(f32, @floatFromInt(j)) * step, @as(f32, @floatFromInt(k)) * step, 0 };
                at += 1;
            }
        }
    }
    std.debug.assert(at == n);
    var source = g.storage(@as(u64, n) * 16);
    // A recorded dispatch may still read source, so submit before releasing it.
    defer g.release(&source);
    g.writeSlice(source, 0, [4]f32, points);
    var output = g.storage(@as(u64, n) * 16);
    errdefer g.release(&output);
    const code = shaders.grid_tables ++ shaders.grids;
    const pipe = try g.pipeline("grids/transport", code, "grid_generate");
    const params = Params{ .count = n, .group = id, .h = h, .beta = @floatCast(@as(f64, @floatFromInt(cut)) / (@as(f64, @floatFromInt(h)) + 0.5) - 1.0) };
    g.dispatch(pipe, (n + 63) / 64, 1, 1, &.{ g.uniform(0, std.mem.asBytes(&params)), source.at(1), output.at(2) });
    g.submit();
    return .{ .quats = output, .count = n };
}
