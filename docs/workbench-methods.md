# Fundamental zones, display and exact counts

The quaternion convention is scalar first with right symmetry action `q*g`.

## Zone membership and relaxation

The fast tests come from the archived Grids_SO3K `laue_ops.py`,
`ori_in_fz_laue`: homogeneous quaternion inequalities for C/D/T/O; for I,
conservative scalar-component gates followed by the twelve face-defining
five-fold rotations. The D3/D4/D6 planar constraints retain the archived group
frame. `shaders/state.wgsl` implements these tests without Rodrigues division.

The final `SO3K_Transports/so3kt/zone.py` gives the group-independent equivalent:
after hemisphere standardization, membership is `w >= max_g |dot(q,g)|`.
An outside quaternion is replaced by the hemisphere-standardized `q*g` with
maximum scalar component. Operators are streamed; no N-by-order intermediate
is allocated. Both import and every relaxation step use this folding kernel.

Updates are left multiplications, `exp(delta)*q`. Right-action folding commutes
with a left update. Therefore the left-trivialized force and BB displacement
history do not need to rotate when a representative changes zones.

An optional identity constraint zeros that site's displacement and writes
`(1,0,0,0)` exactly. Its force is excluded from the BB secant calculation.
Adding the identity is idempotent and retains existing source points. In exact
count mode it consumes one of the requested points, rather than adding N+1.

## Projection and camera

Cyclic zones retain the directional boundary mesh. For bounded D/T/O/I zones,
the Rodrigues halfspaces are intersected to form vertices; vertices sharing
two supporting planes define the edge net. Each straight Rodrigues edge is
sampled and mapped through `theta = 2 atan(|r|)` to either axis–angle radius
`theta` or homochoric radius `[3(theta-sin(theta))/4]^(1/3)`.
Thus D2 has eight vertices/twelve curved cube edges, T six vertices/twelve
octahedron edges, O twenty-four vertices/thirty-six edges, and I twenty
vertices/thirty dodecahedron edges.

Camera extent comes only from this geometry. Sample motion, sample count,
original comparison and drawing limits never change the camera scale.
Full SO(3) uses the entire angle-pi ball. Rodrigues space is unbounded for
cyclic zones and full SO(3); the adjustable RF radius explicitly clips that
view. Orbit rotations are applied around screen axes, with Shift/right-drag
for screen-space pan. The scientific projection remains orthographic.

The display expands only the selected source/operator indices. A coprime
stride prevents bounded display subsets from aliasing to just one operator.
SO(3) has `|K|N` images; the `2|K|N` antipodal S3 count remains appropriate for
the energy-ratio convention but is not the SO(3) display count.

## Exact-count construction

`zo_generate_exact` / `generate(..., exact=True)` share the same Zig path in
Python and WebAssembly. For T/O/I, select the highest lattice semi-edge whose
count fits the budget. For C/D, `mtex.rawCount` evaluates the actual MTEX ring
sum `gamma_count * sum_ring azimuthal_steps` using the same rounding/parity
rules as generation. Enumerate angular-rounding boundaries to choose the largest fitting
grid; subsequent seam compaction may reduce that count further.

Only representatives and nearby symmetry images are kept. The shell streams
all operators and both antipodes of boundary-near sources; it does not assume
that only face-adjacent operators suffice. The hemisphere halfspace is included
for cyclic equatorial seams. Stereographic projection from the south pole
turns relevant spherical Delaunay caps into Euclidean balls.

`src/gpu_delaunay.zig` supplies regional Gao triangulation for insertion.
Spatial queries, local topology and exact predicate signs run on the GPU.
The finite-guard regional argument protects bounded caps up to the shell
margin; candidate caps are then checked against all original symmetry orbits.
The retained general full-hull constructor is a separate development reference.

Circumcaps are computed in quaternion space from the represented stereographic
sites. Retain upper-hemisphere centres in the FZ; reject singular/ill-conditioned
caps, and stream every original orbit to verify emptiness. If the largest cap
approaches the shell margin, enlarge and repeat the shell. This is augmentation
of the generated grids, not a public certificate for arbitrary-set covering.

Process cap centres in descending angular-radius order. Nearly co-spherical
subdivisions can yield identical or very close centres, so a cap occupied by an
earlier insertion is skipped, including symmetry-equivalent insertions. Continue
down the priority list and retriangulate in another batch if needed. This avoids
filling the same hole repeatedly while still achieving the requested count.
The construction is followed by the ordinary optional energy relaxation.

`tests/test_gpu_delaunay.py` compares GPU random 64/1000-site results with Qhull
and checks positive volume, hull volume and independent empty spheres for
those and 13/171-site degenerate FCC fixtures. The CPU Zig constructor and
exhaustive star implementation remain independent development oracles.
`tests/test_workbench.py` exercises exact counts, identity and post-relaxation
FZ membership for all eleven groups. `tests/check_plot_geometry.mjs` checks
polyhedron topology, D2 edge curvature and symmetry-subset coverage.

## Covering checks

`sample.covering()` and the app's Check button use the fully GPU shell path,
including a domain coverage test before triangulation. This is separate from
the greedy exact-count insertion schedule. The measured radius is the largest
retained empty circumcap, checked again against every original symmetry orbit
at its selected witness. The result records the shell bound, probe count,
tetrahedron count and exact-predicate work. See the GPU methods document for
the completeness argument and numerical scope.

Auto initially checks samples up to 10,000 representatives; a measured fast
warm check can enable larger samples. It uses measured
warm latency to choose whether to update during a long relaxation, at its end,
or only on request. A check taking over 1.2 seconds switches Auto to manual
checks for that sample; checks over 0.4 seconds run only at generation/run end.
During-run checks also require five steps and enough elapsed relaxation work
to amortize their cost. First-use compilation is tracked separately. Explicit
Every 10 steps, At end of run and Manual modes override this policy. A retained
radius always carries its iteration; it is marked stale after a move.

Mesh ratio follows the final research package: covering radius divided by minimum
separation. Covering excess is the dimensionless fraction R/R_simplex - 1,
using the archived eight-term simplex series at N = 2|K|N_FZ. Both radii in the
API use SO(3) degrees. Mesh ratio is unavailable until separation and covering
are current together; retained excess is marked stale with its covering result.


## Initialization and inspection controls

Haar initialization uses the seeded GPU Shoemake construction for a uniform
unit quaternion, then the same right-action folding kernel as imports and
relaxation. Python exposes `haar(group, count, seed)`; browser seeds can be
entered or randomized. It creates the requested number, including identity
when that option is enabled in the workbench.

Exact-count generation now accepts up to 1,000,000 representatives. C/D seed
selection enumerates the angular subdivision rounding boundaries instead of
every integer request. Insertion uses Gao regional triangulation: finite
guards cannot affect caps below the shell margin. Candidate emptiness and
newly occupied caps are still checked against streamed original orbits.
The numerical count limit is not a runtime or memory guarantee for large jobs.
The greedy cap validation runs on the CPU. Each candidate's
emptiness is tested through a spatial hash of the orientations of both signs,
which also holds the caps already chosen, so a check touches the few cells
within the cap's chord instead of every orientation. Seam deduplication indexes
the grid without checking it and tests only the inserted points, in cells twice
the tolerance wide. Insertion order can differ between
runs on co-spherical grids: ties in cap radius are broken by GPU scheduling, and
the resulting sets agree except for those ties.

Pair measurements are split into dispatches of about 0.1 s of work: the fused
measure covers a few operators over a range of sites (2.5e8 pair evaluations),
the force pass a range of sites for all operators (3e9), and the rows carry the
sums from one dispatch to the next, which keeps large samples within GPU
watchdog limits.
`tests/test_pair_chunks.py` compares chunked and single-dispatch sums. Refresh
computes the Riesz, Green and log ratios while n^2 |K| is at most 5e10 pair
evaluations (`ratio_limit` in the stats: T up to about 64,000 orientations, O
45,000, I 28,800). Above that it refreshes Quey energy and separation only, and
the ratios read as not computed.

The DT button exports actual vertices and tetrahedra from the covering shell,
only when requested. The display streams a bounded selection of projected
edges, optionally including halo images or transformed copies over SO(3).
These are projected endpoint segments; seam-spanning segments are suppressed,
not an exported curved-edge finite-element mesh. Drawing budgets do not alter
the triangulation. A move invalidates the displayed DT until refreshed.
Point radius, wire width, automatic rotation about sample/world Z, symmetry
operator colors, and six origin halfspace cuts are independent plot controls.
Cuts clip edge segments at the origin plane before camera rotation. History
plots any number of covering, mesh ratio, excess, separation, energy and ratio
measures at once. Chips and the measure tiles toggle series, and the choice is
remembered. Each series is scaled to its own range, so measures with different
units share the plot. A lone series keeps real axis values, and the hover
readout lists every series at the nearest step. Stale covering values are not
added as new measurements after motion. A Log tab in the same pane, hidden by
default and badged with the number of entries added since it was last open (red
after a failure), records each finished operation with its time and result: generation, measures, covering
checks (shell, fence and exact-predicate counts), relaxation and optimization
runs, stops and failures. The central pane spans the full height; the GPU chip
sits at the bottom right.
