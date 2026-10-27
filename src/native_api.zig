const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const Engine = @import("engine.zig").Engine;
comptime {
    _ = @import("exports.zig");
}
const a = std.heap.smp_allocator;
var create_error: [1024]u8 = @splat(0);
export fn zo_create(path: [*:0]const u8) ?*Engine {
    const dev = gpu.Backend.Device.init(a, std.mem.span(path)) catch |e| {
        _ = std.fmt.bufPrintZ(&create_error, "GPU init: {s}", .{@errorName(e)}) catch {};
        return null;
    };
    return Engine.create(a, dev) catch |e| {
        dev.deinit();
        _ = std.fmt.bufPrintZ(&create_error, "Engine init: {s}", .{@errorName(e)}) catch {};
        return null;
    };
}
export fn zo_create_error() [*:0]const u8 {
    return @ptrCast(&create_error);
}
export fn zo_destroy(e: ?*Engine) void {
    if (e) |engine| {
        const dev = engine.dev;
        engine.destroy();
        dev.deinit();
    }
}
// Development instrumentation: GPU timestamps, independent of Python wall time.
export fn zo_profile(e: *Engine, enabled: bool) bool {
    return e.dev.setProfile(enabled);
}
export fn zo_profile_report(e: *Engine, output: [*]u8, capacity: usize) usize {
    return e.g.profileReport(output[0..capacity]);
}
export fn zo_delaunay_mode(e: *Engine, mode: u32) void {
    e.g.delaunay_mode = mode;
}
/// Scales the pair-sum dispatch budget (tests shrink it to force chunking).
export fn zo_pair_budget(e: *Engine, scale: f32) void {
    e.g.pair_budget_scale = scale;
}
/// Development check of live covering updates; see Engine.liveCheck.
export fn zo_live_check(e: *Engine, moves: u32, angle_deg: f32, seed: u32, steps: u32, out: *[6]u64) i32 {
    out.* = e.liveCheck(moves, angle_deg, seed, steps) catch |err| return e.fail("Live check: {s}", .{@errorName(err)});
    return 0;
}
/// Development: the covering shell's sites (shell, then fence) of a new build at the
/// current covering margin, into `out` (4 floats each, up to `capacity`); returns the
/// shell count in `counts[0]` and the fence count in `counts[1]`.
export fn zo_debug_shell(e: *Engine, out: [*][4]f32, capacity: u32, counts: *[2]u32) i32 {
    e.debugShell(out[0..capacity], counts) catch |err| return e.fail("Debug shell: {s}", .{@errorName(err)});
    return 0;
}
