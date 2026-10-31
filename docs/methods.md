# Current numerical methods

## Generation

Every group defaults to a product grid on the inverse Hopf fibration, ported
from SO3K_Grids `so3k/product.py` to `src/hopf.zig`. The projection
pi(q) = q^-1 z satisfies pi(q g) = g^-1 pi(q), so SO(3)/K fibres over the
sphere modulo K, with fibres R_z(t) q. Three root grids are built directly:

| Root | Serves | Sphere grid | Spins n |
|---|---|---|---|
| I | C2 C3 D2 D3 T I | area-preserving icosahedral, 10f^2 + 2 nodes | 5f |
| O | C4 D4 O | area-preserving octahedral, 4f^2 + 2 nodes | 4f below f=6, then 4f - 12 floor(f/9 + 1/2) |
| D6 | C6 D6 | equal-area latitude rings | 6 ceil(2f/3) |

The flat faces of the polyhedron are subdivided at frequency f, and a
Knothe–Rosenblatt map in the O_h or I_h chambers carries the nodes to the sphere
with constant area. With the chamber's far edge on one plane, both azimuthal
marginals have closed forms, which the reference integrates numerically.

Each K-orbit of nodes contributes its node v in the sector at z, the cell of the
vertex z (for D6 the northern hemisphere) at azimuths 0 to 2 pi / m, m the order
of the z axis, so that neighbouring sections are nearly aligned. With stabilizer
order k, the grid takes R_z((2 pi j + phi) / n) s(v) for j < n / k, with s(v) the
shortest rotation taking v to z. Alternate layers are staggered by half a spin
step, phi = pi (l mod 2), where l is f - max(i, j, k) for barycentric indices, or
for D6 the ring number from the nearest pole. Every stabilizer order divides n,
so each orbit appears once: N_FZ = n |V| / |K|.

Any other group H inherits the root P that contains it: with R H R^-1 inside P,
the grid is one q c R per H-orbit, c over the cosets of R H R^-1 in P. The
H-extended set is the P-extended set rotated by R, so every measure of it is the
parent's. N_FZ grows by |P| / |H|. The rotations R come from
`so3k.product.embedding`, tabulated in `src/hopf.zig`.

The whole construction is f64 on the host, O(N |K|) with no iteration, and
nothing is cut or deduplicated. Counts jump with f and are not monotonic for the
octahedral root (f=13 gives 1,130 points for O, f=14 gives 1,048). A requested
count takes the frequency nearest it by ratio. All eleven groups match the
reference row for row, up to the zone image of boundary points, to 5e-8 in
quaternion chord. Their covering excess and mesh ratio match its records.
`kind="transport"` keeps the T/O/I transports below.

The transports: T uses an FCC octahedron, O an FCC truncated cube, and I an FCC cube through
the cubochoric/homochoric map. WGSL evaluates the current transport fits and
quadrature/Newton inversions. The I source-CDF polynomial was regenerated and
checked when the tables were exported. Tests include source ordering,
fundamental-zone membership, unit norms, orbit uniqueness and native glue.
The h2/h8/h12 grids agree with archived CPU f64 quaternions to less than 8e-7
in quaternion Euclidean norm on the tested GPU.

Browser generation is tested separately through the actual WebAssembly/WebGPU
engine. T/O/I at h=2, 6 and 12 agree with the independent CPU-f64 grids to
less than 7e-7 on this device. The Chebyshev fits use explicit recurrence
stages: a dynamically indexed constant-array recurrence gave incorrect O
coordinates under the browser shader compiler despite passing native tests.
The browser QA page (`tests/browser-parity.html`) checks this and a
measure/relax/refresh cycle. Native checks alone do not establish browser parity.

## Pair measures

For representatives q_i and operators g in K, each row visits q_j*g in
workgroup tiles. Antipodal pairs enter analytically from the plus and minus
chords. No 2|K|N cloud or N² matrix is allocated. The identity self pair is
excluded, its true antipodal contribution is included; own-orbit interactions
remain in the Riesz/Green/log measures. Exact duplicate orientations are an
error, rather than silently regularized into a plausible score.

Riesz s=1/2 normalize by N² times 8/(3π) and 1; s=3 normalizes by
2N²log(N)/(3π), where N=2|K|N_FZ. Green uses the zero-mean S3 Green kernel and
the Beltran–Lizarte N^(4/3) bound convention from the archive. Log is centered
before accumulation and uses the archive's 5/12 normalization of its linear
remainder. Compare energy ratios at matched N and with the same convention.
These ratios are not bounded quality percentages.

Distances use quaternion differences, rather than subtracting nearly equal
dot products. Reductions use compensated sums, with host f64 reduction of
per-source rows. Log requires particular care because its normalized remainder
is small relative to the all-pairs sum. Consult the recorded numerical
validation before interpreting tiny log differences; emulated arithmetic must
be tested under the actual compiler and is not assumed equivalent to f64.
As validated, the sensitive log calculation uses double-single
arithmetic with runtime integer-rounding barriers that preserve its low terms.
The tested 200-value logarithm diagnostic has max absolute error 1.14e-13;
the end-to-end normalized log index agrees with dense f64 to better than 1e-9
on the checked roughly 1k-point generated/random cases. These are measured
results, not a portable f64 guarantee.

## Quey relaxation

The mean potential is 1/d + 1/(π-d), where d is the S3 geodesic distance to
q_j*g. Own-source orbit pairs are excluded, matching the archived force.
Forces, curvature and energies share a fused tiled kernel; geodesic updates
run in WGSL. Zig computes alternating BB1/BB2 scales, the median-curvature
preconditioner and step clipping from O(N) row results. A fast post-update
Quey/separation check keeps diagnostics aligned with displayed quaternions.
The costly energy ratios are cleared after moves and computed on demand.
Full measurements split operators into bounded dispatches while retaining
the emulated log high/low words across batches. The Python API
and UI can stop between steps. BB steps are not a monotonic line search.

The fast path splits operators across workgroups and reduces partial sums on
the GPU; scratch is bounded by increasing the operator batch. This improves
occupancy on small sets while keeping symmetry images implicit.

This implements the archived all-pairs version of Quey, Villani & Maurice's
orientation repulsion, rather than the paper's large-neighborhood truncation.
See *Nearly uniform sampling of crystal orientations* (2018),
https://doi.org/10.1107/S1600576718009019.

## Covering optimization

`optimize_covering` lowers the covering radius, the largest empty cap centred
in the zone, by sequential linear programming on disjoint local jobs, in the
spirit of Torquato and Jiao's packing algorithm (PRE 82, 061302, 2010). All
vectorizable work is in `shaders/local.wgsl`; Zig schedules it.

Each step starts from the covering pipeline: the symmetry shell, its GPU
triangulation and the empty zone-centred caps. The shell records each site's
source, operator and sign, so `cap_rows_kernel` can linearize every cap in the
moves of its *sources*. Moves are left multiplications `exp(d) q`, which commute
with the right action, so every symmetry image follows its source. With cap
centre `c`, vertices `v_j` and angular radius `rho`, write `c = sum_j lambda_j v_j`.
Then `d rho = -cot(rho) sum_j lambda_j c.(d v_j)`.

Caps within 30% of the largest are candidates. A job frees one candidate's
sources: the tet (mode `tet`), or also the apex of each cap sharing a face with
it (`face`, the default; at most 8 sources). Its region is every source on a cap
touching a free source. Jobs are chosen in four parallel independent-set rounds.
Priorities are quantized radius and cap id packed into one atomic, and
propagating only each vertex's top claim gives a correct independent set in one
pass. Later rounds block candidates near regions already chosen. Disjoint
regions mean that no cap couples two jobs.

One 256-thread workgroup runs each job, re-tessellating its own neighbourhood.
Let R be the job's largest cap and D = 0.1 R its displacement budget for the
step. The job gathers every orbit point within 2R + 2D of its free sources from
the covering shell's grid; the shell is built wide enough for that. It then
computes each free source's Delaunay star by brute force over
neighbour triples. Every cap of radius at most R through a free point lies within
2R of it, so the stars contain every such cap and no false one. A star is
complete when its tets' solid angles at the free point sum to 4π. A cap larger
than R leaves a deficit, and the move is rejected.

Each star is computed in the stereographic chart of its centre P. Points are
defined as the exact covering pass defines them: each shell site's f32
stereographic coordinate, inverted and rotated into P's chart in double-single.
P sits at the origin, so a cap through P is a sphere through the origin of radius
r, with angular radius atan(2r). New cores are predicted with the same f32
expression the apply kernel uses. The job's local maximum then agrees with a
later full pass to about 1e-6 relative; `check=True` measures this on every step.

Inside the workgroup the job repeats six rounds. It linearizes its caps, solves
the minimax LP, re-tessellates at the trial positions, and keeps the move only
if every star is complete and the largest cap dropped. The LP minimizes the
largest linearized cap, each free source moving within its own trust radius. The
solver is Chambolle–Pock with an entropic (softmax) dual and the best primal
iterate, 120 iterations per round. The step size comes from a power-iteration
estimate of the job's gradient norm. Sums over caps use fixed-point workgroup
atomics, scaled from the job's largest gradient. Each free source ends with one
accumulated rotation. Its trust radius grows by 1.25 if the job kept a move and
halves otherwise. A held identity has trust radius 0.

With `min_separation_deg`, each job also lists every pair between its free
points and the other local points closer than the floor plus twice its move
budget: up to 256 pairs, skipping images of the same source, whose distance a
left move keeps. The LP gains the linearized constraints theta + h.d >= theta_min
with clamped multipliers, and picks its best iterate by exact penalty. A move is
kept only if no listed pair falls below the floor, and no pair already below it
gets nearer. On O at N = 1043 for 30 s, no floor reached 6.09% excess, with
separation falling from 7.97 to 6.32 degrees. A floor at 90% of the start also
reached 6.09%, holding separation at 7.17 degrees; a floor at 100% reached 6.28%
with separation unchanged. Full-pass steps do not enforce the floor.

A job holds at most 192 gathered points and 256 caps. Lattice-like or
co-spherical samples can exceed that; when most of a step's jobs are skipped,
the next 50 steps use full passes. `local=False` keeps the earlier variant for
comparison: it solves each job once and judges it by a second full covering
pass after the move.

Plateaus are left by basin hopping. A descent has plateaued after 150 steps
without a relative gain of 5e-4. The best sample is then restored, every free
source receives an isotropic left kick of RMS angle 0.04 rho, and trust radii
restart at 0.03 rho. The engine keeps the best sample on the GPU and leaves it
in place when a run finishes or stops. The constants were measured on O/T/I at
roughly 1,000 FZ orientations.

Covering optimization lowers minimum separation unless a separation floor
is set.

The app runs optimization in batches of about 300 ms of GPU wait time and
queues the next batch before drawing the finished one, so the engine is never
idle while the page draws. A local step waits on the GPU about 4.5 times: three
readbacks for the covering measure and one for the step's tallies; job sizes
never come back mid-step, since the job kernels are dispatched indirectly.
An unrelaxed Haar sample of 10,000 overflows the local jobs (its holes are
large) and runs full-pass steps until relaxed.
`tests/test_covering_optimizer.py` checks results against a full-orbit Qhull
oracle and checks the local maxima against full passes.

## Scope

The present work targets 1k–10k representatives. Exact-count augmentation and
covering checks use the parallel WGSL triangulator described in
[gpu-delaunay.md](gpu-delaunay.md). CPU implementations are development
oracles. Larger-scale covering optimization remains future work. KeOps is a
development oracle, not a runtime dependency.
