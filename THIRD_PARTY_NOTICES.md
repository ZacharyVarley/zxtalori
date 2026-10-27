# Licenses and source provenance

zxtalori is distributed under the MIT license (`LICENSE`), copyright 2026
Zachary T. Varley. Third-party components keep their own notices below.

## C/D equispaced grids

`src/mtex.zig` and `shaders/mtex.wgsl` implement the C/D branch of the
equispaced SO(3) grid construction of Bachmann, Hielscher & Schaeben (2010), as
used by MTEX's equispacedSO3Grid and equispacedS2Grid. They keep MATLAB rounding,
polar staggering, fibre holonomy and the ZYZ convention so that counts and points
match MTEX (checked against the fixtures in `tests/fixtures/mtex-cd.*`). The
default uses the `skip_dihedral_equator=True` repair. Orbit-aware deduplication
accounts for f32 seam representation differences. Please cite Bachmann,
Hielscher & Schaeben (2010).

## Code from the author's other projects

The GPU abstraction in `src/gpu/`, browser GPU adapter, dependency fetcher,
wheel/build backend and Python local launcher come from zcmir; T/O/I group
geometry, transport coefficients and formulas come from SO3K_Transports. Both
are by the same author under the same MIT license. Registration-specific
front-end/engine logic is not included. The browser import module remains named
`zcgpu` for ABI reuse.

## Gao and Chen parallel Delaunay reference

`shaders/dt_star.wgsl` adapts the independent one-ring / incremental halfspace
construction in Wuheng Gao's MIT implementation of Gao and Chen (2025), pinned
at commit `6e20ee49d3bf13914f1ce87b9194bdde6f2b3273`. Copyright (c) 2025
Wuheng Gao; the full notice is retained in `licenses/MIT-Gao.txt`.
The WebGPU scheduler, outward-rounded search enclosures, integer modular exact
predicates, stable-ID lift ties, covering certificate and interface are new
implementations. CUDA/CUB and the reference predicate implementation are not
linked into the application.

## Runtime and development references

wgpu-native has MIT / Apache-2.0 terms. See `scripts/licenses/` and the
downloaded distribution. The packaged wheel includes its MIT notice.

The preserved research archive includes separately licensed code, papers and
data. It is excluded from application builds and wheels. CGAL, KeOps, Torch
and SciPy are development references only, not bundled production engines.
Their own terms remain applicable when used separately.
