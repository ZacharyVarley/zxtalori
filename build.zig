const std = @import("std");

// Shared native/Python and browser engine, plus a small GPU diagnostic.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const wgpu_dir = b.option([]const u8, "wgpu", "unpacked wgpu-native release") orelse ".deps/wgpu-native/win_amd64";
    const tc = b.addTranslateC(.{
        .root_source_file = b.path("src/gpu/webgpu_c.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = false,
    });
    tc.addIncludePath(b.path("src/gpu/cshim"));
    tc.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ wgpu_dir, "include" }) });
    const shaders = b.createModule(.{ .root_source_file = b.path("shaders/shaders.zig") });
    const probe_mod = b.createModule(.{
        .root_source_file = b.path("src/toolchain_probe.zig"),
        .target = target,
        .optimize = optimize,
    });
    const webgpu_c = tc.createModule();
    probe_mod.addImport("webgpu_c", webgpu_c);
    probe_mod.addImport("shaders", shaders);
    if (target.result.os.tag != .windows) probe_mod.link_libc = true;
    const probe = b.addExecutable(.{ .name = "zxtalori-probe", .root_module = probe_mod });
    b.installArtifact(probe);
    const run = b.addRunArtifact(probe);
    run.addArg(b.pathJoin(&.{ wgpu_dir, "lib" }));
    b.step("probe", "verify native GPU dispatch and readback").dependOn(&run.step);

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/native_api.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_mod.addImport("webgpu_c", webgpu_c);
    lib_mod.addImport("shaders", shaders);
    if (target.result.os.tag != .windows) lib_mod.link_libc = true;
    const lib = b.addLibrary(.{ .name = "zxtalori", .root_module = lib_mod, .linkage = .dynamic });
    b.installArtifact(lib);

    const wasm_mod = b.createModule(.{
        .root_source_file = b.path("src/web_api.zig"),
        .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
        .optimize = .ReleaseSmall,
    });
    wasm_mod.addImport("shaders", shaders);
    const wasm = b.addExecutable(.{ .name = "zxtalori", .root_module = wasm_mod });
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    const install = b.addInstallArtifact(wasm, .{ .dest_dir = .{ .override = .{ .custom = "web" } } });
    const wasm_step = b.step("wasm", "build browser engine");
    wasm_step.dependOn(&install.step);

    const wasm_probe_mod = b.createModule(.{
        .root_source_file = b.path("src/wasm_probe.zig"),
        .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
        .optimize = .ReleaseSmall,
    });
    const wasm_probe = b.addExecutable(.{ .name = "zxtalori-probe", .root_module = wasm_probe_mod });
    wasm_probe.entry = .disabled;
    wasm_probe.rdynamic = true;
    const probe_install = b.addInstallArtifact(wasm_probe, .{ .dest_dir = .{ .override = .{ .custom = "web" } } });
    b.step("wasm-probe", "verify WebAssembly compiler target").dependOn(&probe_install.step);
}
