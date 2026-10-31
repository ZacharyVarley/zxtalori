// SPDX-License-Identifier: MIT
//! Product grids on the inverse Hopf fibration for every group, from SO3K_Grids so3k/product.py.
//! pi(q) = q^-1 z satisfies pi(q g) = g^-1 pi(q), so SO(3)/K fibres over the sphere
//! modulo K, with fibres {R_z(t) q}. Three root grids are built directly, each a K-invariant
//! sphere mesh times n evenly spaced fibre angles: O and I subdivide an octahedron or
//! icosahedron at frequency f and map the nodes to the sphere with an area-preserving
//! Knothe-Rosenblatt map; D6 uses equal-area latitude rings. Each K-orbit of nodes keeps its
//! node v in the sector at z, with stabilizer order k, and takes
//! R_z((2 pi j + phi) / n) s(v) for j < n / k. Alternate layers are staggered, phi = pi (l mod 2),
//! and every stabilizer order divides n, so N_FZ = n |V| / |K| with no orientation twice.
//! Every other group H inherits a root P: with R H R^-1 inside P, one q c R per H-orbit is kept
//! (c over the cosets), and the H-extended set is the P-extended set rotated by R.
//! The base mesh is small (|V| is about 4 f^2 to 10 f^2); all of it runs in f64 here.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const groups = @import("groups.zig");
const pi: f64 = std.math.pi;

pub const Result = struct { quats: gpu.Buf, count: u32 };
pub const max_frequency: u32 = 256;
pub const max_count: u64 = 1000000;
const Q = [4]f64;
const V = [3]f64;
const M = [3][3]f64;

/// The root grid of a group: icosahedral wherever the group embeds in I, the most uniform of the three.
pub fn rootOf(group: groups.Group) groups.Group {
    return switch (group) {
        .C2, .C3, .D2, .D3, .T, .I => .I,
        .C4, .D4, .O => .O,
        .C6, .D6 => .D6,
    };
}
/// R with R H R^-1 inside the root, from so3k.product.embedding (the identity where H already lies in it).
fn embedding(group: groups.Group) Q {
    return switch (group) {
        .C2, .D2, .T => .{ 0.9619383577839175, 0, 0.2732665289126717, 0 },
        .C3, .D3 => .{ 0.9459753748917731, 0.17451645846277514, 0.2687317799958305, -0.04957646865452023 },
        else => .{ 1, 0, 0, 0 },
    };
}
fn order(group: groups.Group) u32 {
    return @intCast(groups.operations(group).len);
}
/// Rotations per full fibre, n, of the root grid.
fn fibreCount(root: groups.Group, f: u32) u32 {
    return switch (root) {
        .O => if (f < 6) 4 * f else 4 * f - 12 * ((2 * f + 9) / 18),
        .I => 5 * f,
        else => 6 * ((4 * f + 5) / 6),
    };
}
/// Population of ring j of the D6 grid (j = f is the equator).
fn ringSize(f: u32, j: u32) u32 {
    const mult: u32 = if (j == f) 12 else 6;
    const x = 4 * @as(f64, @floatFromInt(f)) * @sin(pi * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(2 * f)));
    return @max(mult, mult * @as(u32, @intFromFloat(@floor(x / @as(f64, @floatFromInt(mult)) + 0.5))));
}
fn nodeCount(root: groups.Group, f: u32) u32 {
    switch (root) {
        .O => return 4 * f * f + 2,
        .I => return 10 * f * f + 2,
        else => {
            var total: u32 = 2 + ringSize(f, f);
            for (1..f) |j| total += 2 * ringSize(f, @intCast(j));
            return total;
        },
    }
}
fn rawCount(group: groups.Group, f: u32) u64 {
    const root = rootOf(group);
    return @as(u64, fibreCount(root, f)) * nodeCount(root, f) / order(group);
}
pub fn count(group: groups.Group, f: u32) !u32 {
    if (f < 1 or f > max_frequency) return error.InvalidGridResolution;
    const n = rawCount(group, f);
    if (n > max_count) return error.GridTooLarge;
    return @intCast(n);
}
/// The frequency whose count is nearest the target by ratio. Counts are not monotonic in f for O and its subgroups.
pub fn nearestFrequency(group: groups.Group, target: u32) u32 {
    var best: u32 = 1;
    var score = std.math.inf(f64);
    for (1..max_frequency + 1) |i| {
        const n = rawCount(group, @intCast(i));
        if (n > max_count) continue;
        const s = @abs(@log(@as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(@max(target, 1)))));
        if (s < score) {
            score = s;
            best = @intCast(i);
        }
    }
    return best;
}
/// The frequency with the largest count not above the budget, or null.
pub fn largestFrequency(group: groups.Group, budget: u32) ?u32 {
    var best: ?u32 = null;
    var most: u64 = 0;
    for (1..max_frequency + 1) |i| {
        const n = rawCount(group, @intCast(i));
        if (n <= budget and n > most) {
            most = n;
            best = @intCast(i);
        }
    }
    return best;
}

fn mul(a: Q, b: Q) Q {
    return .{
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    };
}
fn dot4(a: Q, b: Q) f64 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
fn normalize4(q: Q) Q {
    const s = 1 / @sqrt(dot4(q, q));
    return .{ q[0] * s, q[1] * s, q[2] * s, q[3] * s };
}
fn dot(a: V, b: V) f64 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
fn unit(v: V) V {
    const s = 1 / @sqrt(dot(v, v));
    return .{ v[0] * s, v[1] * s, v[2] * s };
}
fn apply(m: M, v: V) V {
    return .{ dot(m[0], v), dot(m[1], v), dot(m[2], v) };
}
fn applyT(m: M, v: V) V {
    return .{
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    };
}
fn matrix(q: Q) M {
    const w = q[0];
    const x = q[1];
    const y = q[2];
    const z = q[3];
    return .{
        .{ 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y) },
        .{ 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x) },
        .{ 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y) },
    };
}
/// A rotation of 2 pi / turns about axis.
fn turn(axis: V, turns: f64) Q {
    const a = unit(axis);
    const s = @sin(pi / turns);
    return .{ @cos(pi / turns), a[0] * s, a[1] * s, a[2] * s };
}

/// Group elements in f64, closed from the reference's generators (the frames of groups.zig)
/// and sorted as it sorts them: decreasing scalar part, identity first.
const Elements = struct {
    q: [60]Q = undefined,
    n: usize = 0,
    fn init(group: groups.Group) Elements {
        const z: V = .{ 0, 0, 1 };
        const x: V = .{ 1, 0, 0 };
        const d: V = .{ 1, 1, 1 };
        var gens: [3]Q = undefined;
        const ng: usize = switch (group) {
            .C2, .C3, .C4, .C6 => 1,
            .T => 3,
            else => 2,
        };
        switch (group) {
            .C2 => gens[0] = turn(z, 2),
            .C3 => gens[0] = turn(z, 3),
            .C4 => gens[0] = turn(z, 4),
            .C6 => gens[0] = turn(z, 6),
            .D2 => gens = .{ turn(z, 2), turn(x, 2), undefined },
            .D3 => gens = .{ turn(z, 3), turn(x, 2), undefined },
            .D4 => gens = .{ turn(z, 4), turn(x, 2), undefined },
            .D6 => gens = .{ turn(z, 6), turn(x, 2), undefined },
            .T => gens = .{ turn(z, 2), turn(x, 2), turn(d, 3) },
            .O => gens = .{ turn(z, 4), turn(d, 3), undefined },
            .I => gens = .{ turn(z, 5), turn(.{ 2, 0, 1 }, 5), undefined },
        }
        var e = Elements{};
        e.q[0] = .{ 1, 0, 0, 0 };
        e.n = 1;
        var i: usize = 0;
        while (i < e.n) : (i += 1) {
            for (gens[0..ng]) |g| {
                const p = mul(e.q[i], g);
                const known = for (e.q[0..e.n]) |k| {
                    if (@abs(dot4(k, p)) > 1 - 1e-9) break true;
                } else false;
                if (!known) {
                    e.q[e.n] = p;
                    e.n += 1;
                }
            }
        }
        std.debug.assert(e.n == order(group));
        for (e.q[0..e.n]) |*q| {
            q.* = normalize4(q.*);
            if (q[0] < 0) q.* = .{ -q[0], -q[1], -q[2], -q[3] };
        }
        std.sort.insertion(Q, e.q[0..e.n], {}, struct {
            fn less(_: void, a: Q, b: Q) bool {
                return a[0] > b[0];
            }
        }.less);
        return e;
    }
};

// Base polyhedra in the reference's vertex and face order (its unique-sorted vertices
// and Qhull simplices), so nodes and orbit order match.
const octa_faces = [8][3]u8{ .{ 5, 4, 3 }, .{ 5, 4, 0 }, .{ 2, 4, 3 }, .{ 2, 4, 0 }, .{ 1, 5, 0 }, .{ 1, 5, 3 }, .{ 1, 2, 0 }, .{ 1, 2, 3 } };
const ico_faces = [20][3]u8{ .{ 5, 3, 0 }, .{ 1, 3, 0 }, .{ 4, 5, 0 }, .{ 9, 5, 3 }, .{ 7, 1, 3 }, .{ 7, 9, 11 }, .{ 7, 9, 3 }, .{ 10, 4, 5 }, .{ 10, 9, 5 }, .{ 10, 8, 11 }, .{ 10, 9, 11 }, .{ 10, 8, 4 }, .{ 6, 7, 1 }, .{ 6, 7, 11 }, .{ 6, 8, 11 }, .{ 2, 8, 4 }, .{ 2, 1, 0 }, .{ 2, 4, 0 }, .{ 2, 6, 1 }, .{ 2, 6, 8 } };
// The icosahedron's non-polar vertices: ring (+-1/sqrt5) and azimuth in degrees.
const ico_ring = [12][2]f64{ .{ -1, 180 }, .{ 1, -144 }, .{ 1, 144 }, .{ -1, -108 }, .{ -1, 108 }, .{ 0, 0 }, .{ 0, 0 }, .{ 1, -72 }, .{ 1, 72 }, .{ -1, -36 }, .{ -1, 36 }, .{ 1, 0 } };

fn vertices(ico: bool) [12]V {
    var v: [12]V = undefined;
    if (!ico) {
        v[0..6].* = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 }, .{ -1, 0, 0 }, .{ 0, -1, 0 }, .{ 0, 0, -1 } };
        return v;
    }
    const r: f64 = 2 / @sqrt(@as(f64, 5));
    for (&v, ico_ring) |*p, ring| {
        const a = ring[1] * pi / 180;
        p.* = .{ r * @cos(a), r * @sin(a), ring[0] / @sqrt(@as(f64, 5)) };
    }
    v[5] = .{ 0, 0, -1 };
    v[6] = .{ 0, 0, 1 };
    return v;
}

/// One symmetry chamber of O_h (octahedron) or I_h (icosahedron): the spherical Voronoi
/// cell of a base point, a triangle with the pole z as a vertex, azimuths [lo, hi] and one
/// far edge, on one flat face of the polyhedron.
const Chamber = struct {
    ops: [120]M = undefined,
    rotated: [120]V = undefined,
    n_ops: usize = 0,
    lo: f64,
    hi: f64,
    /// The flat face holding the chamber.
    normal: V,
    /// The far-edge plane, h . x >= 0.
    edge: V = undefined,
    source_lo: f64 = 0,
    source_span: f64 = 1,
    target_lo: f64 = 0,
    target_span: f64 = 1,

    fn init(ico: bool, v: [12]V) Chamber {
        const p: V = if (ico) .{ @sin(@as(f64, 0.2)) * @cos(0.1 * pi), @sin(@as(f64, 0.2)) * @sin(0.1 * pi), @cos(@as(f64, 0.2)) } else unit(.{ 1, 2, 3 });
        var c = Chamber{
            .lo = if (ico) 0 else pi / 4,
            .hi = if (ico) pi / 5 else pi / 2,
            .normal = if (ico) unit(.{ v[6][0] + v[11][0] + v[8][0], v[6][1] + v[11][1] + v[8][1], v[6][2] + v[11][2] + v[8][2] }) else unit(.{ 1, 1, 1 }),
        };
        const e = Elements.init(if (ico) .I else .O);
        for (e.q[0..e.n]) |q| {
            var m = matrix(q);
            // Exact symmetries, but composition leaves 1 - 4e-16 for 1: snap near-integers,
            // as a residue would open a sliver in the sector planes.
            for (&m) |*row| for (row) |*x| {
                if (@abs(x.* - @round(x.*)) < 1e-10) x.* = @round(x.*);
            };
            var neg = m;
            for (&neg) |*row| for (row) |*x| {
                x.* = -x.*;
            };
            c.ops[c.n_ops] = m;
            c.ops[c.n_ops + e.n] = neg;
            c.n_ops += 1;
        }
        c.n_ops *= 2;
        const mid = 0.5 * (c.lo + c.hi);
        var best: f64 = -1;
        for (c.ops[0..c.n_ops], 0..) |m, i| {
            const mp = applyT(m, p);
            c.rotated[i] = mp;
            var h: V = .{ p[0] - mp[0], p[1] - mp[1], p[2] - mp[2] };
            if (@sqrt(dot(h, h)) <= 1e-9) continue;
            h = unit(h);
            // Planes through the pole bound the azimuth, not the polar angle.
            if (@abs(h[2]) < 1e-12) continue;
            const psi = std.math.atan2(h[2], h[0] * @cos(mid) + h[1] * @sin(mid));
            if (psi > best) {
                best = psi;
                c.edge = h;
            }
        }
        c.source_lo = c.source(c.lo);
        c.source_span = c.source(c.hi) - c.source_lo;
        c.target_lo = c.target(c.lo);
        c.target_span = c.target(c.hi) - c.target_lo;
        return c;
    }
    /// Polar extent of the sector at azimuth phi.
    fn thetaHigh(c: *const Chamber, phi: f64) f64 {
        return pi - std.math.atan2(c.edge[2], c.edge[0] * @cos(phi) + c.edge[1] * @sin(phi));
    }
    // The reference integrates both azimuthal marginals numerically; with the far edge on
    // one plane h they are closed forms. Flat-face area per unit azimuth is proportional to
    // 1 / (P cos + Q sin)^2, whose integral is tan(phi - phi1).
    fn source(c: *const Chamber, phi: f64) f64 {
        const h = c.edge;
        const n = c.normal;
        const p = h[2] * n[0] - n[2] * h[0];
        const q = h[2] * n[1] - n[2] * h[1];
        return (p * @sin(phi) - q * @cos(phi)) / (p * @cos(phi) + q * @sin(phi));
    }
    /// Solid angle 1 - cos(theta_high) per unit azimuth, integrated.
    fn target(c: *const Chamber, phi: f64) f64 {
        const h = c.edge;
        return phi - std.math.asin(std.math.clamp(-h[0] * @sin(phi) + h[1] * @cos(phi), -1, 1));
    }
    fn targetInverse(c: *const Chamber, fraction: f64) f64 {
        var lo = c.lo;
        var hi = c.hi;
        var x = lo + fraction * (hi - lo);
        for (0..64) |_| {
            const residual = (c.target(x) - c.target_lo) / c.target_span - fraction;
            if (residual < 0) lo = x else hi = x;
            const slope = (1 - @cos(c.thetaHigh(x))) / c.target_span;
            const step = x - residual / slope;
            const next = if (step > lo and step < hi) step else 0.5 * (lo + hi);
            if (@abs(next - x) < 1e-15) return next;
            x = next;
        }
        return x;
    }
    /// Knothe-Rosenblatt map from the flat faces to the sphere, applied chamber by chamber.
    fn map(c: *const Chamber, v: V) V {
        var best: usize = 0;
        var score = dot(v, c.rotated[0]);
        for (1..c.n_ops) |g| {
            const s = dot(v, c.rotated[g]);
            if (s > score) {
                score = s;
                best = g;
            }
        }
        const w = apply(c.ops[best], v);
        const phi = std.math.clamp(std.math.atan2(w[1], w[0]), c.lo, c.hi);
        const theta = std.math.atan2(@sqrt(w[0] * w[0] + w[1] * w[1]), w[2]);
        const n = c.normal;
        const a = n[0] * @cos(phi) + n[1] * @sin(phi);
        const t = @tan(theta);
        const tm = @tan(c.thetaHigh(phi));
        const ratio = (t / (n[2] + a * t)) / (tm / (n[2] + a * tm));
        const new_phi = c.targetInverse(std.math.clamp((c.source(phi) - c.source_lo) / c.source_span, 0, 1));
        const new_theta = 2 * std.math.asin(@sqrt(std.math.clamp(ratio * ratio, 0, 1)) * @sin(0.5 * c.thetaHigh(new_phi)));
        const out: V = .{ @sin(new_theta) * @cos(new_phi), @sin(new_theta) * @sin(new_phi), @cos(new_theta) };
        return unit(applyT(c.ops[best], out));
    }
};

const Base = struct {
    nodes: []V,
    /// Rows of the flat face lattice from the nearest vertex, or for D6 rings from the nearest pole.
    layers: []u32,
    fn deinit(b: Base, allocator: std.mem.Allocator) void {
        allocator.free(b.nodes);
        allocator.free(b.layers);
    }
};

/// The area-mapped frequency-f nodes of every face, each shared node once, in the
/// reference's first-occurrence order.
fn polyhedralBase(allocator: std.mem.Allocator, root: groups.Group, f: u32) !Base {
    const ico = root == .I;
    const v = vertices(ico);
    const faces: []const [3]u8 = if (ico) &ico_faces else &octa_faces;
    const chamber = Chamber.init(ico, v);
    const nodes = try allocator.alloc(V, nodeCount(root, f));
    errdefer allocator.free(nodes);
    const layers = try allocator.alloc(u32, nodes.len);
    errdefer allocator.free(layers);
    // A node is identified exactly by its nonzero barycentric weights on polyhedron vertices.
    var seen = std.AutoHashMap([6]u16, void).init(allocator);
    defer seen.deinit();
    var at: usize = 0;
    const scale = 1 / @as(f64, @floatFromInt(f));
    for (faces) |face| {
        for (0..f + 1) |i| {
            for (0..f + 1 - i) |j| {
                const weights = [3]usize{ i, j, f - i - j };
                var key = [6]u16{ 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff };
                var pairs: [3][2]u16 = undefined;
                var np: usize = 0;
                for (0..3) |m| {
                    if (weights[m] == 0) continue;
                    pairs[np] = .{ face[m], @intCast(weights[m]) };
                    np += 1;
                }
                std.mem.sort([2]u16, pairs[0..np], {}, struct {
                    fn less(_: void, x: [2]u16, y: [2]u16) bool {
                        return x[0] < y[0];
                    }
                }.less);
                for (pairs[0..np], 0..) |pair, m| key[2 * m ..][0..2].* = pair;
                if ((try seen.getOrPut(key)).found_existing) continue;
                if (at == nodes.len) return error.BaseMeshCount;
                var x: V = .{ 0, 0, 0 };
                for (0..3) |m| {
                    const b = @as(f64, @floatFromInt(weights[m])) * scale;
                    for (0..3) |e| x[e] += b * v[face[m]][e];
                }
                nodes[at] = chamber.map(unit(x));
                layers[at] = f - @as(u32, @intCast(@max(weights[0], @max(weights[1], weights[2]))));
                at += 1;
            }
        }
    }
    if (at != nodes.len) return error.BaseMeshCount;
    return .{ .nodes = nodes, .layers = layers };
}

/// D6-invariant equal-area rings: both poles, 2 f - 2 rings and the equator, every ring at
/// the middle height of a band of area 4 pi / |V| per node. Alternate rings are staggered
/// by half a step; the equator is not, and holds every two-fold axis.
fn ringBase(allocator: std.mem.Allocator, f: u32) !Base {
    const total = nodeCount(.D6, f);
    const nodes = try allocator.alloc(V, total);
    errdefer allocator.free(nodes);
    const layers = try allocator.alloc(u32, total);
    errdefer allocator.free(layers);
    nodes[0] = .{ 0, 0, 1 };
    nodes[1] = .{ 0, 0, -1 };
    layers[0] = 0;
    layers[1] = 0;
    var at: usize = 2;
    const size: f64 = @floatFromInt(total);
    var top = 1 - 2 / size;
    for (1..f) |j| {
        const m = ringSize(f, @intCast(j));
        const width = 2 * @as(f64, @floatFromInt(m)) / size;
        const z = top - 0.5 * width;
        const radius = @sqrt(1 - z * z);
        for ([_]f64{ z, -z }) |height| {
            for (0..m) |i| {
                const phi = 2 * pi * (@as(f64, @floatFromInt(i)) + 0.5 * @as(f64, @floatFromInt(j % 2))) / @as(f64, @floatFromInt(m));
                nodes[at] = .{ radius * @cos(phi), radius * @sin(phi), height };
                layers[at] = @intCast(j);
                at += 1;
            }
        }
        top -= width;
    }
    const equator = ringSize(f, f);
    for (0..equator) |i| {
        const phi = 2 * pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(equator));
        nodes[at] = .{ @cos(phi), @sin(phi), 0 };
        layers[at] = f;
        at += 1;
    }
    std.debug.assert(at == total);
    return .{ .nodes = nodes, .layers = layers };
}

fn cell(v: V, offset: [3]i32) [3]i32 {
    var k: [3]i32 = undefined;
    for (0..3) |e| k[e] = @as(i32, @intFromFloat(@round(v[e] * 1e7))) + offset[e];
    return k;
}

/// A rotation taking each unit vector v to z; a fixed 180 degree turn about x at -z.
fn section(v: V) Q {
    const q: Q = .{ 1 + v[2], v[1], -v[0], 0 };
    if (@sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) < 1e-12) return .{ 0, 1, 0, 0 };
    return normalize4(q);
}

/// Each image's place in the sector at z: the cell of the vertex z among the polyhedron's
/// vertices (for D6 the northern hemisphere) at azimuths 0 .. 2 pi / m, m the order of the z
/// axis. On its edges the orbit's node is the one with the largest x, then y.
const Sector = struct {
    poles: [12]V,
    n_poles: usize,
    wedge: f64,
    fn init(root: groups.Group) Sector {
        return switch (root) {
            .O => .{ .poles = vertices(false), .n_poles = 6, .wedge = 2 * pi / 4 },
            .I => .{ .poles = vertices(true), .n_poles = 12, .wedge = 2 * pi / 5 },
            else => blk: {
                var poles: [12]V = undefined;
                poles[0] = .{ 0, 0, 1 };
                poles[1] = .{ 0, 0, -1 };
                break :blk .{ .poles = poles, .n_poles = 2, .wedge = 2 * pi / 6 };
            },
        };
    }
    fn contains(s: *const Sector, w: V) bool {
        for (s.poles[0..s.n_poles]) |p| {
            if (dot(w, p) > w[2] + 1e-11) return false;
        }
        if (@sqrt(w[0] * w[0] + w[1] * w[1]) < 1e-11) return true;
        const az = std.math.atan2(w[1], w[0]);
        return az >= -1e-11 and az <= s.wedge + 1e-11;
    }
};
fn better(a: V, b: V) bool {
    const ax = @round(a[0] * 1e11);
    const bx = @round(b[0] * 1e11);
    if (ax != bx) return ax > bx;
    return @round(a[1] * 1e11) > @round(b[1] * 1e11);
}

/// Zone points of the frequency-f grid, scalar first, as f32 rows.
pub fn points(allocator: std.mem.Allocator, group: groups.Group, f: u32) ![][4]f32 {
    const total = try count(group, f);
    const root = rootOf(group);
    const base = if (root == .D6) try ringBase(allocator, f) else try polyhedralBase(allocator, root, f);
    defer base.deinit(allocator);
    const nodes = base.nodes;
    const root_elements = Elements.init(root);
    const parent = root_elements.q[0..root_elements.n];
    var mats: [60]M = undefined;
    for (parent, 0..) |q, i| mats[i] = matrix(q);
    const group_elements = Elements.init(group);
    const elements = group_elements.q[0..group_elements.n];

    // The cosets c H' of H' = R H R^-1 in the root; one q c R per H-orbit.
    const r = embedding(group);
    const r_inverse: Q = .{ r[0], -r[1], -r[2], -r[3] };
    var cosets: [60]Q = undefined;
    var n_cosets: usize = 0;
    var covered = [_]bool{false} ** 60;
    for (parent, 0..) |g, i| {
        if (covered[i]) continue;
        cosets[n_cosets] = g;
        n_cosets += 1;
        for (elements) |h| {
            const member = mul(g, mul(mul(r, h), r_inverse));
            for (parent, 0..) |p, k| {
                if (@abs(dot4(p, member)) > 1 - 1e-9) covered[k] = true;
            }
        }
    }
    if (n_cosets * elements.len != parent.len) return error.GroupNotInRoot;

    var index = std.AutoHashMap([3]i32, u32).init(allocator);
    defer index.deinit();
    for (nodes, 0..) |v, i| try index.put(cell(v, .{ 0, 0, 0 }), @intCast(i));
    const seen = try allocator.alloc(bool, nodes.len);
    defer allocator.free(seen);
    @memset(seen, false);
    const n = fibreCount(root, f);
    const sector = Sector.init(root);
    const out = try allocator.alloc([4]f32, total);
    errdefer allocator.free(out);
    var at: usize = 0;
    for (nodes, 0..) |v, i| {
        if (seen[i]) continue;
        // Mark the orbit; the stabilizer order is the number of images landing on v.
        var stabilizer: u32 = 0;
        var representative: ?V = null;
        for (mats[0..parent.len]) |m| {
            const image = applyT(m, v);
            const j = index.get(cell(image, .{ 0, 0, 0 })) orelse search: {
                for (0..27) |o| {
                    const offset = [3]i32{ @as(i32, @intCast(o % 3)) - 1, @as(i32, @intCast(o / 3 % 3)) - 1, @as(i32, @intCast(o / 9)) - 1 };
                    if (index.get(cell(image, offset))) |found| break :search found;
                }
                return error.BaseMeshNotInvariant;
            };
            const d: V = .{ nodes[j][0] - image[0], nodes[j][1] - image[1], nodes[j][2] - image[2] };
            if (dot(d, d) > 1e-18) return error.BaseMeshNotInvariant;
            seen[j] = true;
            if (j == i) stabilizer += 1;
            if (sector.contains(image) and (representative == null or better(image, representative.?))) representative = image;
        }
        if (n % stabilizer != 0) return error.FibreNotDivisible;
        const s = section(representative orelse return error.OrbitOutsideSector);
        // Odd layers: half a spin step.
        const phase = pi * @as(f64, @floatFromInt(base.layers[i] % 2));
        for (0..n / stabilizer) |j| {
            const t = (2 * pi * @as(f64, @floatFromInt(j)) + phase) / @as(f64, @floatFromInt(n));
            const q = normalize4(mul(.{ @cos(0.5 * t), 0, 0, @sin(0.5 * t) }, s));
            for (cosets[0..n_cosets]) |c| {
                const inherited = mul(mul(q, c), r);
                // The zone representative: of the q g, the one nearest the identity.
                var best = inherited;
                var best_w: f64 = -2;
                for (elements) |g| {
                    var candidate = mul(inherited, g);
                    if (candidate[0] < 0) candidate = .{ -candidate[0], -candidate[1], -candidate[2], -candidate[3] };
                    if (candidate[0] > best_w) {
                        best = candidate;
                        best_w = candidate[0];
                    }
                }
                if (at == total) return error.ProductCount;
                out[at] = .{ @floatCast(best[0]), @floatCast(best[1]), @floatCast(best[2]), @floatCast(best[3]) };
                at += 1;
            }
        }
    }
    if (at != total) return error.ProductCount;
    return out;
}

pub fn generate(allocator: std.mem.Allocator, g: *gpu.Gpu, group: groups.Group, f: u32) !Result {
    const host = try points(allocator, group, f);
    defer allocator.free(host);
    const n: u32 = @intCast(host.len);
    const output = g.storage(@as(u64, n) * 16);
    g.writeSlice(output, 0, [4]f32, host);
    return .{ .quats = output, .count = n };
}
