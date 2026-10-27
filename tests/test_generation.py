"""Actual Python -> Zig -> WGSL generation against archived scientific references.

Run with PYTHONPATH=python after building the native engine. Symmetry expansion
below is confined to these small independent tests, never the product engine.
"""
from pathlib import Path
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

import zxtalori

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def multiply(q, g):
    """Independent scalar-first Hamilton product, broadcasting on leading axes."""
    w = q[..., :1] * g[..., :1] - np.sum(q[..., 1:] * g[..., 1:], axis=-1, keepdims=True)
    v = q[..., :1] * g[..., 1:] + g[..., :1] * q[..., 1:] + np.cross(q[..., 1:], g[..., 1:])
    return np.concatenate((w, v), axis=-1)


def orbit_cloud(q, operations):
    cloud = multiply(q[:, None, :], operations[None, :, :]).reshape(-1, 4)
    cloud /= np.linalg.norm(cloud, axis=1, keepdims=True)
    return np.concatenate((cloud, -cloud))


def check_geometry(q, operations):
    assert np.isfinite(q).all()
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=4e-7, rtol=0)
    assert (q[:, 0] >= -3e-7).all()
    # Closest-to-identity representative, independent of the engine's fold.
    assert np.max(np.abs(q.astype(np.float64) @ operations.astype(np.float64).T) - q[:, :1]) < 3e-6


def minimum_orbit_chord(q, operations):
    # The nearest image is the query's own identity image. The second is the
    # nearest distinct extended-set point, including seam orbit duplicates.
    return cKDTree(orbit_cloud(q.astype(np.float64), operations.astype(np.float64))).query(q, k=2)[0][:, 1].min()


@pytest.mark.parametrize("group", ["T", "O", "I"])
@pytest.mark.parametrize("h", [2, 8, 12])
def test_native_transport(group, h):
    fixture = "reference-small.npz" if h == 2 else "transports-h8-h12.npz"
    with np.load(FIXTURES / fixture) as data:
        expected = data[group if h == 2 else f"{group}_{h}"]
    with zxtalori.generate(group, h=h) as grid:
        actual = grid.quaternions
        assert len(grid) == len(expected)
        check_geometry(actual, grid.operations)
        # Dedicated transports preserve the source lattice order.
        errors = np.linalg.norm(actual.astype(np.float64) - expected, axis=1)
        assert errors.max() < 3e-6, (group, h, errors.max())
        assert minimum_orbit_chord(actual, grid.operations) > 1e-4
        assert grid.stats["group"] == group


@pytest.mark.parametrize("group", ["C2", "C3", "C4", "C6", "D2", "D3", "D4", "D6"])
@pytest.mark.parametrize("target", [257, 1024])
def test_native_mtex_product(group, target):
    metadata = json.loads((FIXTURES / "mtex-cd.json").read_text())
    assert metadata["skip_dihedral_equator"] is True
    with np.load(FIXTURES / "mtex-cd.npz") as data:
        expected = data[f"{group}_{target}"]
    with zxtalori.generate(group, count=target) as grid:
        actual = grid.quaternions
        assert len(grid) == len(expected), (group, target, len(grid), len(expected))
        operations = grid.operations.astype(np.float64)
        check_geometry(actual, operations)
        # MTEX product grids live in an Euler-box domain. The native engine
        # chooses closest-identity zone representatives, so compare orbits.
        forward = cKDTree(orbit_cloud(expected.astype(np.float64), operations)).query(actual, k=1)[0]
        backward = cKDTree(orbit_cloud(actual.astype(np.float64), operations)).query(expected, k=1)[0]
        assert forward.max() < 4e-6, (group, target, "native -> reference", forward.max())
        assert backward.max() < 4e-6, (group, target, "reference -> native", backward.max())
        assert minimum_orbit_chord(actual, operations) > 1e-5, (group, target, "near-duplicate orbit")
