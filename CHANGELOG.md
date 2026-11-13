# Changelog

Notable changes to zxtalori, newest first. Versions follow `__version__` in
`python/zxtalori/__init__.py`; a `v` tag of that version releases it.

## 0.3.0 (2026-11-12)

### Scale
- Generation, nearest-neighbour spacing and drawing run up to 10,000,000 orientations
  (`count` and `h` limits raised; product-grid frequencies up to 512). Ten million grid
  points take about a second.
- `Sample.spacing()`: each orientation's nearest orbit neighbour and the separation, in
  time linear in the count, so samples of any size can be coloured by spacing.
- Kernels whose threads grow with N use a two-dimensional workgroup grid, and a dispatch
  that would still exceed WebGPU's limits fails instead of being skipped.

### Covering checks
- Only the rim of the zone (sources near a zone plane) is classified for the shell.
- The certificate probes only the chart box that holds the zone (`groups.chartReach`), about
  1% of the probes for I and 4% for O.
- The Gao post pass searches growing boxes and stops at the first point inside the sphere;
  main and post passes run in slices so a large check does not hold the GPU for seconds.
- The Delaunay grid is sized from occupied cells; shells are fenced except on small lattices.
- Caps whose centre cannot be resolved are counted (`covering_unresolved_caps`) and can no
  longer hide the largest cap. A repeated check of an unchanged sample returns the cached result.
- First checks of grids of about 500k orientations take 0.15–0.3 s.

### Covering optimization
- Runs measure only the covering radius; energy and separation, which grow as N², are left
  stale (`measures_current` is False) until `measure()`. Short runs at 500k no longer take minutes.
- The plateau kicks are settable: `optimize_covering(patience=, plateau_gain=, kick=, kick_trust=)`,
  and a Plateau kicks panel in the app. `covering_kick_settings` reports them.

### Workbench
- Large samples draw as GPU discs and very large ones (or their SO(3) images) as a per-pixel
  density, chosen automatically.
- Measures use a scale relative to the median by default. The colour key shows the values'
  histogram and their coefficient of variation (CV); Voronoi face counts get one bar per value.
- The plot's controls sit over its sides: zone or SO(3), projection, a movable cut plane and the
  boundary on the left, the colour key on the right.
- The objectives read Auto, Repulsion, Covering, with Auto the default. Generate opens the
  Optimize tab. Covering updates show Manual, locked, for samples too large to check
  automatically.
- Removed: plotting the original points, and the zone and image counts in the right panel.
- Dark-theme dropdown lists are readable.

### Tests
- GPU tests stay near 1,000 orientations and short runs. Size-dependent paths are reached
  through settings (pair budget, near-field forces, render mode), and the three tests that
  need millions of orientations are marked `large`: `pytest -m large`.
- When Auto ends repulsion and covering is checked without the GPU
  (`tests/check_stopping.mjs`).

## 0.2.2 (2026-11-02)
- The workbench sidebar has two tabs, Instantiate and Optimize, with explanations in hover text.
- The Auto objective: repulsion, then covering, each stopped at a plateau, keeping 99% of the
  separation by default.
- After a kick, a descent's lowest sample becomes the best only if it holds the separation floor.

## 0.2.1 (2026-10-31)
- The macOS CI and release jobs run on macos-15.

## 0.2.0 (2026-10-31)
- Inverse Hopf product grids are the default for all eleven groups, with frequencies up to 256
  and up to 1,000,000 points; the FCC transports remain as `kind="transport"` for T, O and I.
- The MTEX grid port is removed.
- Covering-optimizer workgroup state fits Metal's 31-variable limit.

## 0.1.0 (2026-10-26)
- First release.
