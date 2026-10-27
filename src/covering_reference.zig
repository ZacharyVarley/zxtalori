//! Simplex-bound series from SO3K_Transports/so3kt/measures/reference.py.
const std = @import("std");
/// SO(3) angle in degrees; count includes symmetry images and both antipodes.
pub fn simplexDegrees(count: u64) ?f64 {
    if (count == 0) return null;
    const t = std.math.cbrt(1.0 / @as(f64, @floatFromInt(count)));
    const coefficients = [_]f64{ 1.889832101454757, 0.573930763512449, 0.581069217539205, 0.859789414116394, 1.490131881944242, 2.791922498626376, 5.481645765756282, 11.12163594341394 };
    var value: f64 = 0;
    var i: usize = coefficients.len;
    while (i > 0) {
        i -= 1;
        value = coefficients[i] + t * t * value;
    }
    return t * value * 360.0 / std.math.pi;
}
