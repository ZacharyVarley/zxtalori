"""Thin ctypes interface to the same GPU engine used by the browser."""
from __future__ import annotations

import ctypes as C
import json
import os
import sys
from pathlib import Path

import numpy as np

GROUPS = ("C2", "C3", "C4", "C6", "D2", "D3", "D4", "D6", "T", "O", "I")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_LIB = None


def _library():
    global _LIB
    if _LIB is not None:
        return _LIB
    filename = "zxtalori.dll" if sys.platform == "win32" else "libzxtalori.dylib" if sys.platform == "darwin" else "libzxtalori.so"
    paths = [HERE / "_lib" / filename, ROOT / "zig-out" / ("bin" if sys.platform == "win32" else "lib") / filename]
    selected = Path(os.environ["ZXTALORI_LIBRARY"]) if os.environ.get("ZXTALORI_LIBRARY") else next((p for p in paths if p.is_file()), None)
    if selected is None:
        raise RuntimeError("zxtalori engine missing. Build with Zig or install a platform wheel; see CONTRIBUTING.md.")
    lib = C.CDLL(str(selected))
    ptr = C.c_void_p
    signatures = {
        "zo_create": ([C.c_char_p], ptr), "zo_create_error": ([], C.c_char_p),
        "zo_destroy": ([ptr], None), "zo_generate": ([ptr, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32], C.c_int),
        "zo_random": ([ptr, C.c_uint32, C.c_uint32, C.c_uint32], C.c_int),
        "zo_set": ([ptr, C.c_uint32, C.POINTER(C.c_float), C.c_uint32], C.c_int),
        "zo_count": ([ptr], C.c_uint32), "zo_order": ([ptr], C.c_uint32),
        "zo_quaternions": ([ptr], C.POINTER(C.c_float)), "zo_operations": ([ptr], C.POINTER(C.c_float)),
        "zo_measure": ([ptr], C.c_int), "zo_relax": ([ptr, C.c_uint32], C.c_int),
        "zo_relax_neighbourhood": ([ptr, C.c_float], C.c_int),
        "zo_point_values": ([ptr, C.c_uint32], C.POINTER(C.c_float)),
        "zo_identity": ([ptr, C.c_uint32], C.c_int),
        "zo_cover": ([ptr], C.c_int),
        "zo_cover_optimize": ([ptr, C.c_uint32, C.c_uint32, C.c_uint32, C.c_float], C.c_int),
        "zo_mesh": ([ptr], C.c_int),
        "zo_mesh_vertices": ([ptr], C.POINTER(C.c_float)),
        "zo_mesh_tets": ([ptr], C.POINTER(C.c_uint32)),
        "zo_mesh_vertex_count": ([ptr], C.c_uint32),
        "zo_mesh_tet_count": ([ptr], C.c_uint32),
        "zo_generate_exact": ([ptr, C.c_uint32, C.c_uint32, C.c_uint32], C.c_int),
        "zo_stats": ([ptr], C.c_char_p), "zo_error": ([ptr], C.c_char_p), "zo_cancel": ([ptr], None),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    _LIB = lib
    return lib


def _group_id(group):
    try:
        return GROUPS.index(str(group).upper())
    except ValueError:
        raise ValueError(f"group must be one of {', '.join(GROUPS)}") from None


def _wgpu_directory():
    if os.environ.get("ZXTALORI_WGPU"):
        return Path(os.environ["ZXTALORI_WGPU"])
    if (HERE / "_lib").is_dir():
        return HERE / "_lib"
    machine = __import__("platform").machine().lower()
    arm = machine in ("arm64", "aarch64")
    tag = ("win_arm64" if arm else "win_amd64") if sys.platform == "win32" else ("macosx_11_0_arm64" if arm else "macosx_11_0_x86_64") if sys.platform == "darwin" else ("manylinux_2_28_aarch64" if arm else "manylinux_2_28_x86_64")
    return ROOT / ".deps" / "wgpu-native" / tag / "lib"


class Grid:
    """A GPU-resident sample of orientation orbits. Close it to release the device.

    Quaternions have shape (N,4), scalar first, and right action q*g.
    C/D grids use the MTEX product-grid port; T/O/I use FCC transports.
    """
    def __init__(self, group="O", *, h=6, count=2048, quaternions=None, exact=False, identity=False, fixed_identity=False):
        self._lib = _library()
        self._handle = self._lib.zo_create(str(_wgpu_directory()).encode())
        if not self._handle:
            raise RuntimeError(self._lib.zo_create_error().decode(errors="replace"))
        self.history = []
        try:
            if quaternions is None:
                if not 1 <= int(h) <= 64 or not 1 <= int(count) <= 1_000_000:
                    raise ValueError("h must be 1–64 and count must be 1–1,000,000")
                if exact:
                    self._check(self._lib.zo_generate_exact(self._handle, _group_id(group), int(count), int(identity or fixed_identity)))
                else:
                    self._check(self._lib.zo_generate(self._handle, _group_id(group), int(h), int(count), 0))
            else:
                self.set(quaternions, group)
            if identity or fixed_identity:
                self.identity(add=True, fixed=fixed_identity)
        except BaseException:
            self.close()
            raise

    def _check(self, code):
        if code != 0:
            raise RuntimeError(self._lib.zo_error(self._handle).decode(errors="replace"))

    def _open(self):
        if not self._handle:
            raise RuntimeError("Grid is closed")

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.zo_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    def __len__(self):
        self._open()
        return int(self._lib.zo_count(self._handle))

    @property
    def quaternions(self):
        self._open()
        return np.ctypeslib.as_array(self._lib.zo_quaternions(self._handle), shape=(len(self), 4)).copy()

    def point_values(self, kind):
        """Per-orientation display values, or None before the pass that makes them.

        "separation": angle to the nearest other orbit point (SO(3) degrees), from
        every energy pass. "covering": radius of the largest Delaunay cap the point
        touches (degrees) and "faces": faces of its Voronoi cell, from every covering
        pass. stats["point_separation_iteration"] / ["point_covering_iteration"]
        give the iteration they describe.
        """
        self._open()
        kinds = {"separation": 0, "covering": 1, "faces": 2}
        if kind not in kinds:
            raise ValueError(f"kind must be one of {', '.join(kinds)}")
        pointer = self._lib.zo_point_values(self._handle, kinds[kind])
        if not pointer:
            return None
        return np.ctypeslib.as_array(pointer, shape=(len(self),)).copy()

    @property
    def operations(self):
        self._open()
        count = self._lib.zo_order(self._handle)
        return np.ctypeslib.as_array(self._lib.zo_operations(self._handle), shape=(count, 4)).copy()

    @property
    def stats(self):
        self._open()
        return json.loads(self._lib.zo_stats(self._handle))

    def set(self, quaternions, group=None):
        self._open()
        q = np.asarray(quaternions, dtype=np.float32, order="C")
        if q.ndim != 2 or q.shape[1] != 4:
            raise ValueError("quaternions must have shape (N,4), scalar first")
        group = group or self.stats["group"]
        self._check(self._lib.zo_set(self._handle, _group_id(group), q.ctypes.data_as(C.POINTER(C.c_float)), len(q)))
        self.history.clear()
        return self

    def measure(self):
        """Fused Riesz1/2/3, Green, log, separation and Quey energies."""
        self._open()
        self._check(self._lib.zo_measure(self._handle))
        return self.stats

    def identity(self, *, add=False, fixed=False):
        """Optionally add the identity and hold it exactly fixed during relaxation."""
        self._open()
        self._check(self._lib.zo_identity(self._handle, int(add) | (int(fixed) << 1)))
        return self

    def covering(self):
        """Compute the current covering radius with the GPU shell triangulator.

        The radius is an SO(3) rotation angle in degrees. Subsequent relaxation
        retains the last value with covering_current=False until checked again.
        """
        self._open()
        self._check(self._lib.zo_cover(self._handle))
        return self.stats

    def triangulation(self):
        """Current FZ-plus-shell DT: stereographic xyz and tetrahedron indices.

        Explicitly computes covering and transfers the mesh from the GPU.
        The returned arrays are independent copies, valid after later motion.
        """
        self._open()
        self._check(self._lib.zo_mesh(self._handle))
        n=self._lib.zo_mesh_vertex_count(self._handle)
        t=self._lib.zo_mesh_tet_count(self._handle)
        vertices=np.ctypeslib.as_array(self._lib.zo_mesh_vertices(self._handle),shape=(n,4))[:,:3].copy()
        tetrahedra=np.ctypeslib.as_array(self._lib.zo_mesh_tets(self._handle),shape=(t,4)).copy()
        return {"stereographic":vertices,"tetrahedra":tetrahedra,"stats":self.stats}

    def relax(self, steps=50, *, callback=None, neighbourhood=None):
        """Quey geodesic force relaxation with BB steps; updates this grid.

        callback(grid, stats) runs after each accepted engine step. Return False
        to stop. Metrics describe the current quaternions; no expanded set is stored.
        Without a callback the steps run on the GPU in batches of up to 1,000, with
        one readback and one history entry per batch.

        neighbourhood: forces only from orbit points within this many mean
        orientation radii (Quey et al. 2018's delta; 20 holds about 7.5k
        neighbours and relaxes about as all pairs do, at a cost linear in the
        count), or 0 for all pairs. It stays set for later calls; None keeps the
        current setting. Energies reported meanwhile are near-field sums
        (stats["energy_neighbourhood"]).
        """
        self._open()
        if int(steps) < 0:
            raise ValueError("steps must be nonnegative")
        if neighbourhood is not None:
            self._check(self._lib.zo_relax_neighbourhood(self._handle, float(neighbourhood)))
        remaining = int(steps)
        while remaining > 0:
            count = 1 if callback is not None else min(remaining, 1000)
            self._check(self._lib.zo_relax(self._handle, count))
            remaining -= count
            stats = self.stats
            self.history.append(stats)
            if callback is not None and callback(self, stats) is False:
                break
        return self

    def optimize_covering(self, steps=2000, *, mode="face", batch=100, callback=None, local=True, check=False, local_rounds=None, min_separation_deg=None, live=True):
        """Lower the covering radius with GPU local steps; keeps the best sample.

        Each step moves disjoint groups of points around the largest empty caps
        and keeps a group's move only if its exact local maximum drops. Plateaus
        are left by small kicks from the best sample. mode is "tet" (free the
        offending cap's four sources) or "face" (also the apex of each cap sharing
        a face with it). callback(grid, stats) runs after each batch of
        steps; return False to stop. The best sample found is kept on return.

        Each job re-tessellates its neighbourhood on the GPU to judge its moves.
        local=False instead re-measures the whole sample after every step (the
        slower earlier method). check=True also re-measures after every step and
        counts jobs whose local maximum disagrees with the exact one, in
        stats["covering_optimizer_mismatches"]; it is for validation. local_rounds
        sets the local LP rounds each job runs per step (default 6).

        Local steps keep one Delaunay triangulation and update only the stars
        their moves change (images entering and leaving the shell included);
        it is built again only after kicks or when the covering radius outgrows
        its margin (stats["covering_optimizer_live_builds"]).
        live=False measures the whole sample before every step instead.
        Between batches the covering radius comes from that triangulation, and
        energy and separation are re-measured on a time budget (about a tenth
        of the run); stats["measures_current"] says whether they describe the
        current iteration. The final sample is always measured in full.

        min_separation_deg keeps every pair of orientations at least that SO(3)
        angle apart in local steps; pairs already nearer are never brought
        nearer. Full-pass steps (local=False, or the automatic fallback for
        lattice-like samples) do not enforce it.
        """
        self._open()
        modes = {"tet": 0, "face": 1}
        floor = float(min_separation_deg or 0)
        if not 0 <= floor < 180:
            raise ValueError("min_separation_deg must be in [0, 180)")
        if local_rounds is not None and not 1 <= int(local_rounds) <= 255:
            raise ValueError("local_rounds must be 1-255")
        variant = (0 if local else 4) | (8 if check and local else 0) | (0 if live else 16) | (int(local_rounds or 0) << 16)
        if mode not in modes:
            raise ValueError(f"mode must be one of {', '.join(modes)}")
        if int(steps) < 0 or int(batch) < 1:
            raise ValueError("steps must be nonnegative and batch positive")
        done = 0
        while done < int(steps):
            count = min(int(batch), int(steps) - done)
            done += count
            last = done >= int(steps)
            self._check(self._lib.zo_cover_optimize(self._handle, count, modes[mode] | variant, int(last), floor))
            stats = self.stats
            self.history.append(stats)
            if not last and callback is not None and callback(self, stats) is False:
                self._check(self._lib.zo_cover_optimize(self._handle, 0, modes[mode] | variant, 1, floor))
                break
        return self

    def save(self, path):
        """Save JSON shared with the browser, or NPZ for Python."""
        path = Path(path)
        stats = self.stats
        if path.suffix.lower() == ".npz":
            np.savez_compressed(path, quaternions=self.quaternions, metadata=json.dumps(stats))
        else:
            path.write_text(json.dumps({"format": "zxtalori-1", "group": stats["group"], "metadata": stats, "quaternions": self.quaternions.tolist()}, indent=2) + "\n", encoding="utf-8")
        return path


def generate(group="O", *, h=6, count=2048, exact=False, identity=False, fixed_identity=False):
    return Grid(group, h=h, count=count, exact=exact, identity=identity, fixed_identity=fixed_identity)


def from_quaternions(quaternions, group="O"):
    return Grid(group, quaternions=quaternions)


def load(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            return from_quaternions(data["quaternions"], metadata["group"])
    data = json.loads(path.read_text(encoding="utf-8"))
    return from_quaternions(data["quaternions"], data["group"])

def haar(group="O", count=1000, seed=193):
    """Seeded GPU Haar-uniform random orientations folded into SO(3)/K."""
    grid=Grid(group,h=1,count=8)
    try:
        grid._check(grid._lib.zo_random(grid._handle,_group_id(group),count,seed))
        return grid
    except BaseException:
        grid.close()
        raise
