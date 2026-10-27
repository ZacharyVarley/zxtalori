"""Exact Fraction oracle for orient/insphere signs and the predicate test cases
shared by the GPU predicate tests (independent of the Zig and WGSL code)."""
from __future__ import annotations

from fractions import Fraction
from itertools import permutations
import random
import struct


def from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def rounded(value):
    return from_bits(bits(value))


def determinant(matrix):
    """Fraction Gaussian elimination, independent of the Zig cofactor expansion."""
    a = [list(row) for row in matrix]
    result = Fraction(1)
    for i in range(len(a)):
        pivot = next((r for r in range(i, len(a)) if a[r][i]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            result = -result
        value = a[i][i]
        result *= value
        for r in range(i + 1, len(a)):
            factor = a[r][i] / value
            for c in range(i + 1, len(a)):
                a[r][c] -= factor * a[i][c]
    return result


def orientation(points):
    return determinant([[p[k] - points[0][k] for k in range(3)] for p in points[1:]])


def sign(value):
    return (value > 0) - (value < 0)


def reference(points, ids):
    q = [[Fraction(v) for v in p] for p in points]
    orient = orientation(q[:4])
    raw = determinant([p + [sum(v * v for v in p), Fraction(1)] for p in q])
    sos = sign(raw)
    if not raw and len(set(map(tuple, q))) == 5:
        for i in sorted(range(5), key=ids.__getitem__):
            # Compute the actual 4x4 cofactor independently, without using
            # the orient3d/cofactor sign identity implemented in Zig.
            minor = determinant([p + [Fraction(1)] for j, p in enumerate(q) if j != i])
            cofactor = (-1) ** (i + 3) * minor
            if cofactor:
                sos = sign(cofactor)
                break
    return [sign(orient), sign(raw), sos], orient, raw


def make_cases():
    rng = random.Random(20261022)
    cases = []

    def add(label, points, ids=None):
        points = [tuple(rounded(v) for v in point) for point in points]
        cases.append((label, points, ids or rng.sample(range(1 << 20), 5)))

    unit = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for e in [(0.25, 0.25, 0.25), (2, 2, 2), (1, 1, 1)]:
        add("known_unit_tetrahedron", unit + [e])
    sphere = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0)]
    stable_ids = [900, 12, 351, 4, 70]
    for order in permutations(range(5)):
        add("exact_sphere_permutations", [sphere[i] for i in order], [stable_ids[i] for i in order])
    for direction in [-1, 1]:
        near = sphere[:4] + [(0, from_bits(bits(-1.0) + direction), 0)]
        for order in permutations(range(4)):
            add("near_sphere_permutations", [near[i] for i in order] + [near[4]], [stable_ids[i] for i in order] + [stable_ids[4]])
    for _ in range(80):
        add("ordinary_random", [[rng.uniform(-2, 2) for _ in range(3)] for _ in range(5)])
    for _ in range(80):
        values = []
        while len(values) < 15:
            candidate = rng.getrandbits(32)
            if (candidate >> 23) & 255 != 255:
                values.append(from_bits(candidate))
        add("all_finite_bit_patterns", [values[i:i + 3] for i in range(0, 15, 3)])
    for _ in range(40):
        add("coplanar_dyadic", [[rng.randrange(-32, 33) / 16, rng.randrange(-32, 33) / 16, 0] for _ in range(5)])
        centre = [rng.randrange(-16, 17) / 16 for _ in range(3)]
        radius = 2.0 ** rng.randrange(-8, 4)
        add("exact_dyadic_sphere", [[centre[k] + radius * p[k] for k in range(3)] for p in sphere])
    tiny = from_bits(1)
    add("subnormal", [[tiny * p[k] for k in range(3)] for p in sphere])
    add("signed_zero_duplicate", [(0, -0.0, 0), (-0.0, 0, 0)] + sphere[:3])
    maximum = from_bits(0x7F7FFFFF)
    previous = from_bits(0x7F7FFFFE)
    extreme = [(-maximum, -maximum, -maximum), (maximum, -maximum, -maximum),
               (-maximum, maximum, -maximum), (-maximum, -maximum, maximum)]
    add("maximum_finite_cospherical", extreme + [(maximum, maximum, maximum)])
    add("maximum_finite_near_cospherical", extreme + [(maximum, maximum, previous)])
    add("maximum_and_subnormal", [(maximum, tiny, 0), (-maximum, 0, tiny),
                                  (tiny, maximum, 0), (0, tiny, -maximum), (tiny, -maximum, maximum)])
    return cases


