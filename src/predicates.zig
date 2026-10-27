//! Exact cold predicates for the represented finite binary32 coordinates.
//!
//! Every f32 times 2^149 is an integer of magnitude <2^277. Differences are
//! <2^278; direct orient3d and insphere expansions are <2^837 and <2^1397.
//! Their intermediates therefore fit i2048. No floating arithmetic, allocated
//! expansions, epsilon, or data-dependent precision bound enters the sign.
//!
//! Inputs whose nonzero exponents span at most 36 binades are scaled by 2^-e_min
//! instead: integers of magnitude <2^60, differences <2^61, orient3d terms <2^183
//! (6 of them: <2^186), lifts <2^124 and insphere terms <2^310 (4: <2^312), so i320 holds
//! every intermediate. Same integers up to one power of two, hence the same signs.
const std = @import("std");

pub const Point = [3]f32;
pub const Sign = enum(i8) { negative = -1, zero = 0, positive = 1 };
pub const Error = error{ NonFinite, DuplicateId };
const Wide = i2048;
const Narrow = i320;
/// Largest exponent span (binades) the narrow width holds.
const narrow_span = 36;

fn sign(comptime T: type, value: T) Sign {
    return if (value < 0) .negative else if (value > 0) .positive else .zero;
}

/// Biased exponent of a finite value, subnormals counting as 1; 0 for zero.
fn exponentOf(value: f32) Error!u32 {
    const bits: u32 = @bitCast(value);
    const exponent = (bits >> 23) & 255;
    if (exponent == 255) return error.NonFinite;
    if (bits & 0x7fffffff == 0) return 0;
    return @max(exponent, 1);
}

/// value * 2^(149 - lowest + 1) as an integer: lowest = 1 gives the f32 * 2^149 scale.
fn integer(comptime T: type, value: f32, lowest: u32) T {
    const bits: u32 = @bitCast(value);
    const exponent = (bits >> 23) & 255;
    const fraction = bits & 0x7fffff;
    if (bits & 0x7fffffff == 0) return 0;
    var magnitude: T = @intCast(if (exponent == 0) fraction else fraction | 0x800000);
    const e = @max(exponent, 1);
    magnitude <<= @intCast(e - lowest);
    return if (bits >> 31 == 0) magnitude else -magnitude;
}

/// The narrowest exact width for these points, by exponent span.
fn narrow(comptime n: usize, points: [n]Point) Error!?u32 {
    var lowest: u32 = 255;
    var highest: u32 = 0;
    for (points) |point| for (point) |value| {
        const e = try exponentOf(value);
        if (e == 0) continue;
        lowest = @min(lowest, e);
        highest = @max(highest, e);
    };
    if (highest == 0) return 1;
    return if (highest - lowest <= narrow_span) lowest else null;
}

fn convert(comptime T: type, comptime n: usize, points: [n]Point, lowest: u32) [n][3]T {
    var result: [n][3]T = undefined;
    for (points, 0..) |point, i| {
        for (point, 0..) |value, k| result[i][k] = integer(T, value, lowest);
    }
    return result;
}

fn subtract(comptime T: type, a: [3]T, b: [3]T) [3]T {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn det3(comptime T: type, rows: [3][3]T) T {
    const a = rows[0];
    const b = rows[1];
    const c = rows[2];
    return a[0] * (b[1] * c[2] - b[2] * c[1]) -
        a[1] * (b[0] * c[2] - b[2] * c[0]) +
        a[2] * (b[0] * c[1] - b[1] * c[0]);
}

fn orientInteger(comptime T: type, points: [4][3]T) T {
    return det3(T, .{
        subtract(T, points[1], points[0]),
        subtract(T, points[2], points[0]),
        subtract(T, points[3], points[0]),
    });
}

fn insphereInteger(comptime T: type, points: [5][3]T) T {
    var rows: [4][3]T = undefined;
    var lifts: [4]T = undefined;
    for (0..4) |i| {
        rows[i] = subtract(T, points[i], points[4]);
        const p = rows[i];
        lifts[i] = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    }
    var determinant: T = 0;
    for (0..4) |i| {
        var minor: [3][3]T = undefined;
        var j: usize = 0;
        for (0..4) |r| {
            if (r == i) continue;
            minor[j] = rows[r];
            j += 1;
        }
        const term = lifts[i] * det3(T, minor);
        determinant += if (i & 1 == 0) -term else term;
    }
    return determinant;
}

/// Sign of (b-a) dot ((c-a) cross (d-a)). Exact coplanarity returns zero.
pub fn orient3d(points: [4]Point) Error!Sign {
    if (try narrow(4, points)) |lowest| return sign(Narrow, orientInteger(Narrow, convert(Narrow, 4, points, lowest)));
    return sign(Wide, orientInteger(Wide, convert(Wide, 4, points, 1)));
}

/// Raw sign of det([x,y,z,x*x+y*y+z*z,1]) for the five ordered rows.
/// For orient3d(a,b,c,d)>0 it is negative when e is inside. A zero-orientation
/// defining tetrahedron has no geometric inside/outside classification;
/// callers must retain that degeneracy even if this raw polynomial is nonzero.
pub fn insphere(points: [5]Point) Error!Sign {
    if (try narrow(5, points)) |lowest| return sign(Narrow, insphereInteger(Narrow, convert(Narrow, 5, points, lowest)));
    return sign(Wide, insphereInteger(Wide, convert(Wide, 5, points, 1)));
}

/// The same raw determinant, resolving exact full-dimensional cospheres by
/// positive infinitesimal lift perturbations in increasing stable-ID order.
/// All IDs must be globally unique and follow points through permutations.
/// Fully coplanar configurations and duplicate represented coordinates retain
/// zero. A caller must still reject orientation-zero defining tetrahedra.
pub fn insphereSos(points: [5]Point, ids: [5]u64) Error!Sign {
    for (0..5) |i| {
        for (i + 1..5) |j| {
            if (ids[i] == ids[j]) return error.DuplicateId;
        }
    }
    if (try narrow(5, points)) |lowest| return sos(Narrow, convert(Narrow, 5, points, lowest), ids);
    return sos(Wide, convert(Wide, 5, points, 1), ids);
}

fn sos(comptime T: type, coordinates: [5][3]T, ids: [5]u64) Sign {
    const determinant = insphereInteger(T, coordinates);
    if (determinant != 0) return sign(T, determinant);
    for (0..5) |i| {
        for (i + 1..5) |j| {
            if (std.mem.eql(T, &coordinates[i], &coordinates[j])) return .zero;
        }
    }
    var consumed = [_]bool{false} ** 5;
    for (0..5) |_| {
        var row: usize = 5;
        for (0..5) |i| {
            if (!consumed[i] and (row == 5 or ids[i] < ids[row])) row = i;
        }
        consumed[row] = true;
        var minor: [4][3]T = undefined;
        var j: usize = 0;
        for (0..5) |i| {
            if (i == row) continue;
            minor[j] = coordinates[i];
            j += 1;
        }
        // det([x,y,z,1])=-orient3d; the lift column's cofactor is
        // (-1)^(row+3) times that minor, hence (-1)^row * orient3d.
        const orientation = orientInteger(T, minor);
        if (orientation != 0) return sign(T, if (row & 1 == 0) orientation else -orientation);
    }
    return .zero;
}

test "finite inputs and unique tie IDs are explicit requirements" {
    const origin: Point = .{ 0, 0, 0 };
    const bad: Point = .{ std.math.inf(f32), 0, 0 };
    const nan: Point = .{ std.math.nan(f32), 0, 0 };
    try std.testing.expectError(error.NonFinite, orient3d(.{ origin, origin, origin, bad }));
    try std.testing.expectError(error.NonFinite, insphere(.{ origin, origin, origin, origin, nan }));
    try std.testing.expectError(error.DuplicateId, insphereSos(.{ origin, origin, origin, origin, origin }, .{ 0, 1, 2, 3, 3 }));
    try std.testing.expectEqual(Sign.zero, try insphereSos(.{ origin, origin, origin, origin, origin }, .{ 0, 1, 2, 3, 4 }));
}
