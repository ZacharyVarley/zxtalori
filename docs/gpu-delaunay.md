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

Only images of the rim are classified: the sources within the widest band
(the fence's, below) of a zone halfspace, listed in source order by the same
count, scan and write, with the core sites written on the way. An image of a
source in the zone lies in another copy of the zone, at least the source's
distance to the zone boundary away, so no other source has an image in a band.
The image passes are dispatched indirectly over the rim's images, in the same
order, so the shell is unchanged. Classifying all 2|K|N images took 55 ms of
the GPU time of a first check of an I grid at 494k (I has 119 halfspaces), the
rim passes 6 ms.

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
probes are folded into the FZ. The zones of D2, D4, D6, T, O and I lie in
chart 0 (q0 >= |q_k|), within a box of half-width `groups.chartReach` (the zone's
largest Rodrigues component: 1, 1, 1, 1, sqrt(2)-1 and 0.4016, from linear
programs over the halfspaces; tests/test_covering_certificate.py recomputes
them). The nearest chart-0 vertex of a zone point lies within half a step of
that box, so only the box's vertices (plus a step) are probed: about 1% of the
four charts' vertices for I, 4% for O. C_n and D3 zones leave chart 0 and keep
all four. Each must find an
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

Without the fence the outermost shell stars' searches reach the regional guards
and cross much of the grid: a first check of an I grid of 494k orientations took
2.4 s unfenced and 0.19 s fenced, a T grid of 490k 3.2 s and 0.35 s. On small
lattices the fence's co-spherical images multiply exact predicates (about
tenfold on T h=8, 3,281 points: 78 ms unfenced against 428 ms), but from about
20k orientations it is faster there too (T h=16: 1.9 s against 0.23 s). So every
sample is fenced except a lattice of fewer than 10,000 orientations: a transport
grid before its first pass, or a sample whose last Gao pass had more than n/256
exact predicates.

The grid holds about five sites per occupied cell. A new sample's first pass
counts the occupied cells of a box-sized trial grid (one small readback) and
sizes its grid from that; later passes reuse the last grid's occupancy, and
lattices keep the box-sized grid, which suits their co-spherical cells (T
transport at 354k: 4.5 s against 13.5 s sized to occupancy).

The main and post passes run in slices of 32,768 stars, a submit apart (the
slice's first star in `lower.w`): one dispatch over a million stars can hold the
GPU for seconds, and the display waits for it. A shell that fails the
certificate is rejected before it is triangulated, so a margin too small for a
random sample costs the shell and the probes, not a triangulation. `Gpu.delaunay_mode` bit `1 << 20` omits it for
comparison.

A post-pass search over more than 512 cells grows boxes from the range's start
cell (the sphere centre's, or the seed's when the centre is outside the grid)
and stops at the first box holding a point inside the sphere; smaller ranges
are searched whole, as before. Any point inside the sphere is a valid
insertion, so the stars are unchanged. Large spheres through a star still
contain points near it, while the range's far side is often empty box: on a
first check of an I grid at 494k, 22 searches started from the range's low
corner crossed 1.9M cells, and the post pass took 38 ms of GPU time; it now
takes 4 ms, with the same 3,260,558 tetrahedra.

The shell is stereographically projected from the south pole, triangulated,
and mapped back to S3. Double-single circumcap normals retain the pole-side
choice and the FZ filter. The largest cap's witness is independently checked
against every original orbit; a mismatch is an error. A cap is accepted when its
four vertices are equidistant from the double-single centre, rounded to f32,
within 2e-6 chord + 1e-4 of its squared chord: the rounded centre alone moves a
squared chord by about 5e-7 chord. An FZ-centred cap that fails this, or has no
centre, is counted (`covering_unresolved_caps`) with an upper estimate of its
radius, and the check fails (UnresolvedCircumcap) unless that estimate is below
the reported radius, so an unresolved cap can never hide the largest one. Caps
whose vertices all lie beyond T are dropped as before. A repeated request for an
unchanged sample (the same group and orientations, by hash) returns the last
result (`covering_cached`). `sample.covering()`
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
