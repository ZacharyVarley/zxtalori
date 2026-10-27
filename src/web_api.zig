const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const Engine = @import("engine.zig").Engine;
comptime {
    _ = @import("exports.zig");
}
const a = std.heap.wasm_allocator;
extern "env" fn zo_log(ptr: [*]const u8, len: usize) void;
pub fn panic(message: []const u8, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    zo_log(message.ptr, message.len);
    @trap();
}
export fn zo_alloc(n: usize) ?[*]u8 {
    const p = a.alignedAlloc(u8, .@"16", n) catch return null;
    return p.ptr;
}
export fn zo_free(p: [*]align(16) u8, n: usize) void {
    a.free(p[0..n]);
}
export fn zo_create() ?*Engine {
    const dev = gpu.Backend.Device.init(a) catch return null;
    return Engine.create(a, dev) catch {
        a.destroy(dev);
        return null;
    };
}
export fn zo_destroy(e: *Engine) void {
    const dev = e.dev;
    e.destroy();
    a.destroy(dev);
}
