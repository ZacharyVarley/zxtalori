//! Wall-clock milliseconds for engine time budgets: the std.Io clock natively, the
//! page's performance.now() in the browser (the env.zo_now import).
const std = @import("std");
const builtin = @import("builtin");

const web = struct {
    extern "env" fn zo_now() f64;
};

pub fn ms() f64 {
    if (builtin.target.os.tag == .freestanding) return web.zo_now();
    const ns = std.Io.Clock.awake.now(std.Io.Threaded.global_single_threaded.io()).nanoseconds;
    return @as(f64, @floatFromInt(ns)) * 1e-6;
}
