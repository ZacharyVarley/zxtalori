# Initial reference fixtures

These fixtures were generated once, in a separate Python reference environment,
from an archived copy of the MIT SO3K_Transports sources. The archive and the
generation scripts are kept outside this repository; the fixtures are the
published record.

`reference-small.npz` holds the initial porting references:

- C2, C3, C4, C6, D2, D3, D4, D6, T, O, I: h=2 zone grids, CPU float64,
  scalar-first quaternions, reference default lattice/aspect settings.
  The C/D transport entries are historical preparation fixtures only; current
  C/D generation tests use the repaired MTEX fixtures described below.
- `dt_random_points`: NumPy default_rng(193) normal (64,3), float64.
- `dt_random_tets`: canonical sorted tetrahedra, matched by CGAL and Qhull.
- `dt_fcc_points`: 13 integer FCC points in [-1,1]^3.
- `dt_fcc_cgal_tets`: one legal CGAL subdivision. Other valid cospherical
  subdivisions are allowed; do not require identical tet lists here.

These establish reproducible small inputs for porting. They are neither an
independent proof of transport correctness nor sufficient large-grid tests.
The DT entries support the current GPU triangulation regression tests.

## Current generation fixtures

`transports-h8-h12.npz` contains CPU-f64 T/O/I grids at h=8 and h=12 from the
same source, made by the same exporter that wrote `shaders/grid_tables.wgsl`.

`mtex-cd.npz` contains all eight C/D groups at requested counts 257 and 1024,
using the archived MTEX port with **`skip_dihedral_equator=True`**. The polar
parity repair is the application's default: it avoids equatorial orbit twins
present in the unrepaired MTEX sequence. Its precise generation choices and
actual counts are in `mtex-cd.json`.

`tests/test_generation.py` checks the actual native Python/Zig/WGSL path:
source-order agreement for the dedicated T/O/I transports, count and
orbit-equivalent agreement for C/D, unit norms, FZ membership, and absence of
near-duplicate orbits. Full symmetry clouds are created transiently only in
these small reference tests.

`browser-reference.json` holds independent CPU-f64 T/O/I at h=2, 6 and 12
for `tests/browser-parity.html`. It catches browser shader compiler differences
that are invisible to native tests. It comes from the same table exporter and
is served only from the development QA page.

`covering-reference.json` holds small full-orbit Qhull covering radii for the
browser covering QA page; `pair-operators.npz` holds the dense f64 pair-kernel
references for `tests/test_pair_kernels.py`.

`gao-reflection-predicates.json` captures binary32 coordinates from real exact
zero predicates in the O grid. Independent rational determinants verify the
zeros. These cases exercise two disjoint reflection pairs with an arbitrary
fifth point, which need not lie in the same plane as the other four.
