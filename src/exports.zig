const std = @import("std");
const Engine = @import("engine.zig").Engine;
const groups = @import("groups.zig");
export fn zo_version() [*:0]const u8 {
    return @import("engine.zig").version;
}
export fn zo_generate(e: *Engine, id: u32, h: u32, target: u32, seed: u32) i32 {
    if (id >= 11) return e.fail("Unknown group", .{});
    const group: groups.Group = @enumFromInt(id);
    e.generate(group, h, target, seed) catch |err| return e.fail("Generation failed: {s}", .{@errorName(err)});
    return 0;
}
export fn zo_generate_exact(e: *Engine, id: u32, target: u32, flags: u32) i32 {
    if (id >= 11) return e.fail("Unknown group", .{});
    e.generateExact(@enumFromInt(id), target, flags & 1 != 0) catch |err| return e.fail("Exact-count generation: {s}", .{@errorName(err)});
    return 0;
}
export fn zo_set(e: *Engine, id: u32, p: [*]const f32, n: u32) i32 {
    if (id >= 11) return e.fail("Unknown group", .{});
    const group: groups.Group = @enumFromInt(id);
    e.set(group, p[0 .. @as(usize, n) * 4]) catch |err| return e.fail("Import failed: {s}", .{@errorName(err)});
    return 0;
}
export fn zo_count(e: *Engine) u32 {
    return e.n;
}
export fn zo_order(e: *Engine) u32 {
    return @intCast(groups.operations(e.group).len);
}
export fn zo_operations(e: *Engine) [*]const [4]f32 {
    return groups.operations(e.group).ptr;
}
export fn zo_quaternions(e: *Engine) [*]const f32 {
    return e.host.ptr;
}
export fn zo_measure(e: *Engine) i32 {
    e.measure() catch |err| return e.fail("Measure failed: {s}", .{@errorName(err)});
    return 0;
}
export fn zo_cover(e: *Engine) i32 {
    e.measureCovering() catch |err| return e.fail("Covering radius: {s}", .{@errorName(err)});
    return 0;
}
export fn zo_relax(e: *Engine, steps: u32) i32 {
    e.relax(steps) catch |err| return e.fail("Relaxation failed: {s}", .{@errorName(err)});
    return 0;
}
/// Relaxation forces from orbit points within delta mean orientation radii (Quey's
/// near field; delta 20 holds about 7.5k neighbours); 0 for all pairs.
export fn zo_relax_neighbourhood(e: *Engine, delta: f32) i32 {
    e.setNeighbourhood(delta) catch |err| return e.fail("Relaxation neighbourhood: {s}", .{@errorName(err)});
    return 0;
}
/// Per-point display values (kind 0 nearest-neighbour degrees, 1 local covering radius
/// degrees, 2 Voronoi faces), zo_count() of them; null when not yet measured.
export fn zo_point_values(e: *Engine, kind: u32) ?[*]const f32 {
    return if (e.pointValues(kind)) |values| values.ptr else null;
}
export fn zo_identity(e: *Engine, flags: u32) i32 {
    e.identity(flags & 1 != 0, flags & 2 != 0) catch |err| return e.fail("Identity constraint: {s}", .{@errorName(err)});
    return 0;
}
export fn zo_stats(e: *Engine) [*:0]const u8 {
    return e.stats();
}
export fn zo_error(e: *Engine) [*:0]const u8 {
    return @ptrCast(&e.err);
}
export fn zo_cancel(e: *Engine) void {
    e.cancelled.store(true, .release);
}

export fn zo_random(e: *Engine, id: u32, count: u32, seed: u32) i32 {
    if (id >= 11) return -1;
    e.generateRandom(@enumFromInt(id), count, seed) catch |err| return e.fail("Haar initialization: {s}", .{@errorName(err)});
    return 0;
}
export fn zo_mesh(e: *Engine) i32 {
    e.buildMesh() catch |err| return e.fail("DT display: {s}", .{@errorName(err)});
    return 0;
}
export fn zo_mesh_vertices(e: *Engine) [*][4]f32 {
    return e.mesh.vertices.ptr;
}
export fn zo_mesh_tets(e: *Engine) [*][4]u32 {
    return e.mesh.tets.ptr;
}
export fn zo_mesh_vertex_count(e: *Engine) u32 {
    return @intCast(e.mesh.vertices.len);
}
export fn zo_mesh_tet_count(e: *Engine) u32 {
    return @intCast(e.mesh.tets.len);
}
/// Covering optimization: `steps` local steps; mode 0 frees the offending tet,
/// 1 also its face neighbours. Bit 4: full exact pass after each move instead of
/// local re-tessellation; bit 8: re-tessellate and check against a full pass.
/// `finish` restores the best sample. Local steps keep every pair of orientations at
/// least `floor_deg` apart (an SO(3) angle; 0 for none) and never bring pairs already
/// below it nearer.
export fn zo_cover_optimize(e: *Engine, steps: u32, mode: u32, finish: u32, floor_deg: f32) i32 {
    e.optimizeCovering(steps, mode, finish != 0, floor_deg) catch |err| return e.fail("Covering optimization: {s}", .{@errorName(err)});
    return 0;
}
