// SPDX-License-Identifier: MIT
//! MTEX equispacedSO3Grid C/D branch, from the archived equispaced.py port.
//! Bachmann, Hielscher & Schaeben (2010); original MTEX commit a7896a4.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const groups = @import("groups.zig");
const shaders = @import("shaders");
const pi: f64 = std.math.pi;
pub const Result = struct { quats: gpu.Buf, count: u32 };

// Exact raw ring-count formula used to enumerate the MTEX count jumps without
// dispatching a grid. Orbit seam compaction can only decrease this upper bound.
pub fn rawCount(group: groups.Group, target: u32) !u32 {
    const index = @intFromEnum(group);
    if (index >= 8) return error.UnsupportedGroup;
    const mz: f64 = switch (group) {
        .C2, .D2 => 2,
        .C3, .D3 => 3,
        .C4, .D4 => 4,
        .C6, .D6 => 6,
        else => unreachable,
    };
    const beta: f64 = if (index < 4) pi else pi / 2;
    const gamma = pi / mz;
    const res = 2 / std.math.cbrt(@as(f64, @floatFromInt(@max(target, 8))) / (beta * gamma));
    const no_center = @as(u64, @intFromFloat(@round(if (index < 4) 2 * pi / res else pi / res))) % 2 == 0;
    const snapped = @max(1.0, @round(2 * beta / res));
    const ring_res = 2 * beta / snapped;
    const ntheta: u32 = @intFromFloat(@trunc(@round(2 * beta / ring_res + @as(f64, if (no_center) 1 else 0)) / 2));
    const rings = ntheta + @as(u32, if (no_center) 0 else 1);
    const gammas: u32 = @intFromFloat(@max(1.0, @round(2 * gamma / res)));
    var total: u32 = 0;
    for (0..rings) |ring| {
        const theta = (@as(f64, @floatFromInt(ring)) + @as(f64, if (no_center) 0.5 else 0)) * ring_res;
        const steps: u32 = @intFromFloat(@max(1.0, @round(@sin(theta) * (2 * pi / beta) * @as(f64, @floatFromInt(@max(ntheta, 1))))));
        total += steps * gammas;
    }
    return total;
}

pub fn generate(allocator: std.mem.Allocator, g: *gpu.Gpu, group: groups.Group, target: u32) !Result {
    const index = @intFromEnum(group);
    if (index >= 8) return error.UnsupportedGroup;
    const mz: f64 = switch (group) {
        .C2, .D2 => 2,
        .C3, .D3 => 3,
        .C4, .D4 => 4,
        .C6, .D6 => 6,
        else => unreachable,
    };
    const max_beta: f64 = if (index < 4) pi else pi / 2;
    const max_gamma: f64 = pi / mz;
    const res = 2 / std.math.cbrt(@as(f64, @floatFromInt(@max(target, 8))) / (max_beta * max_gamma));
    // The archived port's skip_dihedral_equator=True repair prevents orbit twins.
    const snap = if (index < 4) 2 * pi / res else pi / res;
    const no_center = @as(u64, @intFromFloat(@round(snap))) % 2 == 0;
    const snapped = @max(1.0, @round(2 * max_beta / res));
    const ring_res = 2 * max_beta / snapped;
    const ntheta: u32 = @intFromFloat(@trunc(@round(2 * max_beta / ring_res + @as(f64, if (no_center) 1 else 0)) / 2));
    const ring_count = ntheta + @as(u32, if (no_center) 0 else 1);
    const gamma_count: u32 = @intFromFloat(@max(1.0, @round(2 * max_gamma / res)));
    var eulers: std.ArrayList([4]f32) = .empty;
    defer eulers.deinit(allocator);
    for (0..ring_count) |ring| {
        const theta = (@as(f64, @floatFromInt(ring)) + @as(f64, if (no_center) 0.5 else 0)) * ring_res;
        const steps: u32 = @intFromFloat(@max(1.0, @round(@sin(theta) * (2 * pi / max_beta) * @as(f64, @floatFromInt(@max(ntheta, 1))))));
        const width: f64 = @floatFromInt(steps);
        const stagger: f64 = @as(f64, @floatFromInt((ring + 1) % 2)) * pi / width;
        const beta = std.math.acos(std.math.clamp(@cos(theta), -1, 1));
        for (0..steps) |k| {
            const rho = @as(f64, @floatFromInt(k)) * (2 * pi / width) + stagger;
            const alpha = std.math.atan2(@sin(theta) * @sin(rho), @sin(theta) * @cos(rho));
            const dg = std.math.atan2(-(@cos(beta) + 1) * @sin(alpha), (@cos(beta) + 1) * @cos(alpha));
            for (0..gamma_count) |j| {
                try eulers.append(allocator, .{ @floatCast(alpha), @floatCast(beta), @floatCast(dg - max_gamma + @as(f64, @floatFromInt(j)) * (2 * max_gamma / @as(f64, @floatFromInt(gamma_count)))), 0 });
            }
        }
    }
    if (eulers.items.len == 0) return error.EmptyGrid;
    var input = g.storage(eulers.items.len * 16);
    defer g.release(&input);
    g.writeSlice(input, 0, [4]f32, eulers.items);
    const ops = groups.operations(group);
    var operators = g.storage(ops.len * 16);
    defer g.release(&operators);
    g.writeSlice(operators, 0, [4]f32, ops);
    const count: u32 = @intCast(eulers.items.len);
    const output = g.storage(eulers.items.len * 16);
    const pipe = try g.pipeline("mtex/zyz", shaders.mtex, "main");
    const u: [4]u32 = .{ count, @intCast(ops.len), 0, 0 };
    g.dispatch(pipe, (count + 255) / 256, 1, 1, &.{ input.at(0), operators.at(1), output.at(2), g.uniform(3, std.mem.asBytes(&u)) });
    return .{ .quats = output, .count = count };
}
