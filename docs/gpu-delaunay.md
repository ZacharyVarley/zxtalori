# GPU Delaunay stars and covering radius

Covering uses the source-guided Gao constructor in `src/gao_delaunay.zig` and
`shaders/gao_*.wgsl`. It follows the
CUDA source's independent-star main/post algorithm, with GPU exact predicates
and checked capacity retries. Both native Python and the browser use this path.

The retained general full-hull triangulator is
`src/gpu_delaunay.zig` with `shaders/dt_*.wgsl`. Its full-hull construction and
validation are described below. CPU code schedules dispatches, reads compact
completion metadata and grows buffers. CPU constructors remain development
oracles; there is no automatic CPU substitution on failure.

## General full-hull construction

The reference is Wuheng Gao's implementation of Gao & Chen's *Parallel 3D
Delaunay Triangulation on the GPU* (2025),
[paper](https://doi.org/10.1016/j.cad.2025.103933),
[CUDA source](https://github.com/WuhengGao/Parallel-3D-Delaunay-GPU), pinned at
`6e20ee49d3bf13914f1ce87b9194bdde6f2b3273`. Its MIT notice is retained in
`licenses/MIT-Gao.txt`.

Each site owns a private, oriented triangulated link. Insertion clips that
link by the new site's Voronoi halfspace: remove conflicting faces, extract
and validate one directed horizon cycle, then append the new faces. Sites
proceed independently; there is no cross-workgroup shared topology mutation.
A linked uniform grid supplies initial neighbors and circumsphere candidate
queries. Circumsphere search bounds use outward-rounded interval arithmetic;
the same intervals reject sites provably outside the sphere before evaluating
a determinant. An uncertain bound widens the query rather than certifying an
empty region.
This first implementation uses conservative sphere/AABB queries. It does not
yet implement all of Gao's localization, sorting and CUDA-specific scheduling
optimizations, so performance parity with that CUDA code is not claimed.

A state machine bounds each dispatch and resumes after cold predicates.
Thirty-two step/exact rounds precede a compact completion readback. Storage is
batched, with 128 link faces initially and checked capacity retries up to
1,024. Output capacity comes from measured tetrahedron counts. Minimum global
site ID owns each emitted tetrahedron. Final links must be closed with reversed
edge incidences; exposed real hull faces are tested against all sites.
Finite enclosing vertices grow on a detected incomplete hull and the same
GPU algorithm is retried. Capacity, topology, hull and iteration failures are
explicit errors; no partial triangulation is reported as complete.

## Predicates and degeneracy

The geometric contract is the represented finite binary32 stereographic
coordinates. Raw bits are stored in integer buffers so subnormal information
survives. Common power-of-two normalization makes filtered floating-point
calculations safer; a coordinate that cannot be represented normally after
scaling bypasses the filter. Conservative binary32 filters precede
double-single filters with runtime integer rounding barriers. Uncertain signs
queue an exact modular GPU determinant.

The exact kernel evaluates residues modulo up to 96 distinct 16-bit primes and
recovers the signed result with Garner mixed-radix reconstruction. Scaling
any finite binary32 coordinate by 2^149 gives an integer of magnitude below
2^277. A conservative bound on the scaled insphere determinant is 2^1397;
the prime product exceeds 2^1440, so the signed result is unique. Residue
products fit unsigned 32-bit arithmetic. A floating quotient estimate is
corrected by integer remainder checks; determinant signs do not depend on
an uncorrected floating result. A common exact power of two is removed
first; the resulting bit-range bound selects fewer primes for ordinary inputs.
The bound is recomputed for each determinant, including finite enclosing
vertices. This also handles exact zeros and
subnormals without GPU f64 or host predicate evaluation.

Exact co-spheres use the first nonzero orientation cofactor of a symbolic
height perturbation ordered by stable global IDs. Every star sees the same
tie order. Coordinates are never jittered. Duplicate represented sites and
lower-dimensional inputs are errors, rather than fictitious positive-volume
tetrahedra. A degenerate cell may have a different valid tetrahedral subdivision
from Qhull; positive volume, hull coverage and empty spheres remain required.

## Covering shell and completeness

`shaders/shell.wgsl` streams every right-action operator and both antipodes.
It retains core representatives plus images within S3 angular margin T of
the identity fundamental zone, using all signed group halfspaces, including
the hemisphere boundary. A source farther than T from every zone boundary
needs no halo images. There is no persistent full symmetry-expanded set.
The initial T scales with point density; failed completeness checks enlarge
it. Capacity grows from the measured retained-image count.

Retained images are compacted stably, in image order: operator, then sign,
then source. A per-workgroup count, a one-workgroup scan and the write give
every site the same id in every pass, and the DT's symbolic tie-breaking
depends on these ids. With atomic compaction they depended on scheduling,
and lattice passes needed several times as many exact predicates.

The completeness check is independent of the triangulator. Four normalized
cube charts `(1,x,y,z)`, with the major coordinate permuted, cover RP3.
For m subdivisions per coordinate, every chart point lies within S3 angular
distance delta <= sqrt(3)/m of a grid vertex: the parameterization has
derivative norm at most one and the nearest vertex is within sqrt(3)/m.
Probes are generated on the fly. A chart vertex can be omitted if both it
and its antipode violate some unit FZ halfspace by more than sin(delta), with
numerical padding. A nearest chart vertex to any FZ point is within delta, so
it cannot violate that point's halfspaces by more than sin(delta). Thus this
pruning preserves the chart cover, including equatorial seams. Remaining
probes are folded into the FZ. Each must find an
orbit site within T-delta, with additional numerical margins. By the triangle
inequality every domain point then has a site within T. Therefore every
maximal empty cap has radius <= T, and all sites that can influence an FZ cap
lie in the admitted shell. Failure enlarges T and repeats the check.

The same compaction appends a fence after the shell: the images beyond margin
T but within 1.5T (below pi/2), of sources within 1.5T of a zone boundary. Fence sites are real orbit sites but not seeds. The Gao main pass
builds no star for them, so only tetrahedra owned by a shell site are output.
A cap through a fence site that is centred in the FZ has radius above T, so
the circumcap filter drops it. Every cap that counts (FZ-centred, radius at
most T) contains no fence site, so its tetrahedron is in both triangulations,
with shell vertices only, and is output by a shell seed. The radius and witness
are therefore unchanged; on Haar samples of eight groups they are bit-identical. On
co-spherical input a tied maximum can come from another tetrahedron of the
same empty sphere: the witness can then be a symmetric equivalent, and the
radius can differ by rounding (1e-6 degrees on T h=12). The fence bounds the empty spheres of the outermost
shell sites, which otherwise reach the regional guards: their searches crossed
up to 200 cells and were then deferred to the cooperative post pass.

On lattices the fence's co-spherical images multiply exact predicates (about
tenfold on T h=8). The fence is therefore used only after a Gao pass with at most n/256 exact predicates, and a
new sample starts without it. `Gpu.delaunay_mode` bit `1 << 20` omits it for
comparison.

The shell is stereographically projected from the south pole, triangulated,
and mapped back to S3. Double-single circumcap normals retain the pole-side
choice and the FZ filter. The largest cap's witness is independently checked
against every original orbit; a mismatch is an error. `sample.covering()`
returns its SO(3) rotation radius in degrees, twice the S3 angular radius,
along with witness, shell bound/count, tetrahedron/probe counts, cold-predicate
count and scheduling rounds.

Exact predicates apply to the rounded projected sites. Quaternion generation,
group multiplication, projection, shell tests and cap evaluation still have
floating-point error. Numerical guards and independent oracle comparisons
support the reported precision; this is not an exact-arithmetic certificate
for arbitrary real-valued input rotations. Poor, highly clustered imports may
need a broad shell or fail the checked bounds. A failure retains the last
successful radius and marks it stale after motion.

## Validation

`tests/test_gpu_predicates.py` checks 832 exact signs and 1,248 adaptive
orientation/insphere/symbolic-tie decisions against rational determinants,
including extreme finite values, subnormals, permutations and degeneracies.
The source-guided path additionally has independent topology, cooperative
construction, exact-reflection and capacity-growth tests in `test_gao_*.py`.
`tests/test_gpu_delaunay.py` checks random 64/1,000-site clouds against Qhull
and independent empty spheres, plus valid 13/171-site FCC subdivisions.
`tests/test_workbench.py` covers all-group arbitrary counts, fixed identity,
FZ folding and thin hulls through the production GPU path.

`tests/test_covering.py` compares all eleven groups before and after relaxation
against a full signed-orbit 4D Qhull oracle. Checked native small-case differences
were below 0.00001 degrees. Browser differences were below 0.00002 degrees. The browser QA page uses independent Qhull fixtures and
also checks current/stale transitions.

Million-point triangulation and covering have not been validated.
Exact-count construction now uses regional Gao DT, with host-side cap sorting
and streamed greedy emptiness checks. The general full-hull constructor remains
a separately tested development reference.

Auto uses a density/group-based initial estimate and then scales measured warm
latency with count and group order. It requires enough relaxation work between
checks to amortize their cost. Explicit cadence modes remain available;
`tests/check_covering_policy.mjs` records those scheduling decisions.
