//! Infrastructure diagnostic, not an orientation algorithm or performance benchmark.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const shaders = @import("shaders");

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(init.gpa);
    defer args.deinit();
    _ = args.next();
    const path = args.next() orelse ".deps/wgpu-native/win_amd64/lib";
    const dev = try gpu.Backend.Device.init(init.gpa, path);
    defer dev.deinit();
    var g = try gpu.Gpu.init(init.gpa, dev);
    defer g.deinit();
    const n = 1000;
    const pipe = try g.pipeline("setup/probe", shaders.probe, "main");
    var out = g.storage(n * 4);
    defer g.release(&out);
    g.dispatch(pipe, (n + 255) / 256, 1, 1, &.{out.at(0)});
    var host: [n]f32 = undefined;
    g.read(out, 0, std.mem.sliceAsBytes(&host));
    g.submit();
    try dev.wait();
    if (dev.takeError()) |message| {
        std.debug.print("GPU error: {s}\n", .{message});
        return error.GpuValidation;
    }
    for (host, 0..) |value, i| {
        if (value != @as(f32, @floatFromInt(i)) * 2 + 1) return error.WrongReadback;
    }
    std.debug.print("PASS: {s}; {d} WGSL results verified\n", .{ dev.info(), n });
    std.debug.print("limits: storage {d} MiB, buffer {d} MiB, workgroup memory {d} B\n", .{
        g.lim.max_storage_binding >> 20, g.lim.max_buffer >> 20, g.lim.max_workgroup_storage,
    });
}
