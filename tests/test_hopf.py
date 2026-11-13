"""Inverse Hopf product grids (src/hopf.zig) against the SO3K_Grids so3k.product reference.

tests/fixtures/hopf-reference.npz holds product_grid(group, f) in float64 and
product_covering_bound(group, f) for all eleven groups at small frequencies.
"""
from pathlib import Path

import numpy as np
import pytest

import zxtalori
from test_covering import full_orbit_covering
from test_generation import check_geometry, minimum_orbit_chord, multiply

FIXTURES = Path(__file__).resolve().parent / "fixtures"
with np.load(FIXTURES / "hopf-reference.npz") as _data:
    REFERENCE = {key: _data[key] for key in _data.files}
CASES = [(key.split("_")[0], int(key.split("_")[1])) for key in REFERENCE if not key.endswith("bound")]
ROOT = {"C2": "I", "C3": "I", "D2": "I", "D3": "I", "T": "I", "I": "I", "C4": "O", "D4": "O", "O": "O", "C6": "D6", "D6": "D6"}
ORDER = {"C2": 2, "C3": 3, "C4": 4, "C6": 6, "D2": 4, "D3": 6, "D4": 8, "D6": 12, "T": 12, "O": 24, "I": 60}


def ring(f, j):
    mult = 12 if j == f else 6
    return max(mult, mult * int(np.floor(4 * f * np.sin(np.pi * j / (2 * f)) / mult + 0.5)))


def count(group, f):
    root = ROOT[group]
    if root == "O":
        n, nodes = (4 * f if f < 6 else 4 * f - 12 * ((2 * f + 9) // 18)), 4 * f * f + 2
    elif root == "I":
        n, nodes = 5 * f, 10 * f * f + 2
    else:
        n, nodes = 6 * int(np.ceil(4 * f / 6)), 2 + ring(f, f) + 2 * sum(ring(f, j) for j in range(1, f))
    return n * nodes // ORDER[group]


@pytest.mark.parametrize("group,f", CASES)
def test_matches_reference_rows(group, f):
    expected = REFERENCE[f"{group}_{f}"]
    with zxtalori.generate(group, h=f) as grid:
        actual = grid.quaternions.astype(np.float64)
        assert grid.stats["grid"] == "hopf" and grid.stats["h"] == f
        operations = grid.operations.astype(np.float64)
    assert len(actual) == len(expected) == count(group, f)
    check_geometry(actual, operations)
    # Row order is the reference's. A point on the zone boundary may take another
    # zone image there, so each row is compared with the reference row's orbit.
    images = multiply(expected[None], operations[:, None])
    chord = np.minimum(np.linalg.norm(actual[None] - images, axis=-1), np.linalg.norm(actual[None] + images, axis=-1)).min(0)
    assert chord.max() < 1e-6, chord.max()


@pytest.mark.parametrize("group,f", [("T", 3), ("O", 4), ("O", 7), ("I", 3), ("D6", 4), ("C6", 3), ("C3", 2), ("D4", 3)])
def test_covering_matches_hull_and_product_bound(group, f):
    with zxtalori.generate(group, h=f) as grid:
        measured = grid.covering()["covering_radius_deg"]
        expected = full_orbit_covering(grid.quaternions, grid.operations)
        assert minimum_orbit_chord(grid.quaternions, grid.operations) > 1e-4
    assert abs(measured - expected) < 2e-4
    # The bound is an S3 angle; the covering radius is its SO(3) rotation angle.
    assert measured <= 2 * np.degrees(float(REFERENCE[f"{group}_{f}_bound"])) + 1e-4


# Covering excess and mesh ratio from the reference's toi_sweep and axial_sweep records.
@pytest.mark.parametrize("group,f,excess,mesh", [
    ("O", 13, 0.125316, 0.887439), ("I", 12, 0.096216, 0.71551), ("T", 6, 0.079974, 0.700277), ("D6", 7, 0.140646, 0.830677),
    ("C6", 4, 0.146348, 0.896823), ("C4", 6, 0.138155, 0.802456), ("D4", 6, 0.138155, 0.802456), ("D3", 4, 0.0666, 0.694876), ("C2", 4, 0.0666, 0.694876),
])
def test_measures_match_reference_records(group, f, excess, mesh):
    with zxtalori.generate(group, h=f) as grid:
        grid.covering()
        assert abs(grid.stats["covering_excess"] - excess) < 1e-5
        assert abs(grid.measure()["mesh_ratio"] - mesh) < 1e-5


def regenerate(grid, group, h=0, count=2048):
    """Generate again on the grid's own engine: creating an engine costs about half a second."""
    grid._check(grid._lib.zo_generate(grid._handle, zxtalori.api._group_id(group), h, count, 0, 0))


@pytest.mark.parametrize("group", zxtalori.GROUPS)
def test_counts_and_selection(group):
    sizes = {f: count(group, f) for f in range(1, 513) if count(group, f) <= 10_000_000}
    with zxtalori.generate(group, count=1000, exact=True, identity=True) as grid:
        assert len(grid) == 1000 and grid.stats["grid"] == "hopf"
        assert grid.stats["base_count"] == max(n for n in sizes.values() if n <= 999)
        # Every frequency up to about 1,000 orientations (O reaches h=14, where its ring
        # count takes its second correction).
        for f in [f for f, n in sizes.items() if n <= 1100]:
            regenerate(grid, group, h=f)
            assert len(grid) == count(group, f) and grid.stats["h"] == f
        # With no frequency, the count nearest the request by ratio.
        for target in (300, 900):
            regenerate(grid, group, count=target)
            assert grid.stats["grid"] == "hopf"
            assert abs(np.log(len(grid) / target)) == min(abs(np.log(n / target)) for n in sizes.values())


def test_frequency_beyond_the_count_limit():
    with pytest.raises(RuntimeError, match="GridTooLarge"):
        zxtalori.generate("C2", h=100)


def test_transport_is_polyhedral_only():
    with zxtalori.generate("O", h=6, kind="transport") as grid:
        assert len(grid) == 1043 and grid.stats["grid"] == "transport"
    with zxtalori.generate("O", count=1000, exact=True, kind="transport") as grid:
        assert len(grid) == 1000 and grid.stats["grid"] == "transport"
    with pytest.raises(ValueError):
        zxtalori.generate("D4", kind="transport")
    with pytest.raises(ValueError):
        zxtalori.generate("O", kind="lattice")


@pytest.mark.large
def test_ten_million():
    # Near the count limit, in the zone and nearest to the request.
    with zxtalori.generate("O", count=10_000_000) as grid:
        q = grid.quaternions
        assert grid.stats["h"] == 281 and len(grid) == count("O", 281)
        assert np.all(q[:, 0] >= -1e-6) and np.abs(np.linalg.norm(q, axis=1) - 1).max() < 1e-5
        assert grid.spacing()["separation_deg"] > 0.3
