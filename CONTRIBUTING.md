# Build and test

The engine is in `src/` and `shaders/`; Python and JavaScript are thin adapters.
The browser and native schedules share the same source. Development has been on
Windows x64; CI also runs the test suite on Apple M1 (Metal).

```powershell
python scripts/fetch_deps.py
uv --cache-dir .deps/uv-cache venv --python python .venv
uv --cache-dir .deps/uv-cache pip install --python .venv/Scripts/python.exe -r requirements-dev.lock
$env:ZIG_GLOBAL_CACHE_DIR = Join-Path (Get-Location) '.deps/zig-global-cache'
.\.deps\zig-0.16.0\zig.exe build -Doptimize=ReleaseFast
.\.deps\zig-0.16.0\zig.exe build wasm
node scripts/serve.mjs
```

The server prints the browser URL. `node scripts/build_web.mjs` packages a
static copy at `dist/web/`. After building, the Python launcher can also serve
the checkout:

```powershell
$env:PYTHONPATH = 'python'
.\.venv\Scripts\python.exe -m zxtalori --no-browser
.\.venv\Scripts\python.exe -m pytest -q
```

Use `PYTHONPATH=python` rather than `python -I`: isolated mode ignores it and
imports any installed copy of the package instead of this checkout.

## Tests

`tests/test_generation.py` exercises actual Zig lattice construction, GPU
transport/ZYZ conversion, seam deduplication and Python transfer against f64
fixtures. Direct shader tests compare pair/force kernels against dense f64
NumPy formulas, and native measure tests check public API invariants and Quey
descent. Full expanded orbits are permitted in small independent test oracles;
production code must keep them implicit.

- `test_gpu_predicates.py`, `test_gpu_delaunay.py`: WGSL filters, symbolic
  ties and integer exact signs against rational arithmetic.
- `test_covering.py`: every group before/after relaxation against full
  signed-orbit Qhull. `test_gao_*.py` cover the Gao triangulation's topology,
  exact predicates and capacity retries; `test_covering_certificate.py` checks
  that certificate pruning keeps the required FZ chart vertices.
- `test_covering_optimizer.py`: covering optimization, including live updates.
- `test_near_field.py`: near-field relaxation sums against a dense f64
  reference, and its outcome against all pairs.
- `test_point_values.py`: per-point separation, local covering radius and
  Voronoi faces against Qhull and a k-d tree.
- `node tests/check_covering_policy.mjs` checks the automatic covering schedule;
  `node tests/check_plot_geometry.mjs` checks polyhedron topology, homochoric
  edge curvature and display subsampling.
- `node tests/test_web.mjs` drives the app in headless Chrome (WebGPU) through
  generation, repulsion, covering optimization and a near-field run, failing on
  any page error; `--dir dist/web` tests a built site. `tests/test_app.py`
  checks that the launcher serves every file of the app.

Some tests also write validation receipts to `research/validation/`, which Git
ignores.

Run `python scripts/serve_browser_tests.py 4183` and open
`http://127.0.0.1:4183/qa/` for the browser/WASM regression: generation against
CPU-f64 T/O/I at h=2, 6 and 12, measure/relax/refresh, all-group exact counts,
fixed identity, FZ folding, thin-shell hull checks, and GPU covering against
full-orbit Qhull fixtures (`/qa/covering` checks the Gao path alone). Results are
displayed as JSON. The QA pages and their reference data are omitted from the
packaged app.

The reference fixtures in `tests/fixtures/` were generated once from the
archived SO3K_Transports sources; that archive and its generation tooling are
not part of this repository (see `tests/fixtures/README.md`).

## Packaging

```powershell
python scripts/build_wheels.py win_amd64
python -m pip install dist/zxtalori-0.1.0-py3-none-win_amd64.whl
python -m zxtalori
```

The custom PEP517 backend makes `pip install .` perform the same build.
Wheels contain the Python interface, native library, wgpu-native and static
browser app. Toolchains and test fixtures are excluded. Source releases use an
explicit source manifest and work before the first Git commit. No publication
is performed by a build.

After making an sdist, `python scripts/check_sdist.py` extracts it into a fresh
directory and builds a wheel using only the cached toolchains. It checks source
and wheel license metadata paths and catches files missing from the source
manifest. The resulting isolated wheel remains under `.deps/source-check-*/`.

`scripts/check_installed.py`, run with the installed environment's interpreter
and `-I`, checks native generation/relaxation, JSON/NPZ round-trips, and the
packaged launcher's HTML/JS/WASM/CSS responses. The launcher sends headers and
content together to avoid a reproduced Windows loopback binary-transfer stall.
It uses HTTP/1.1 and preserves the browser's persistent-connection choice;
native response checks alone do not replace the actual browser regression.

## CI and releases

`.github/workflows/ci.yml` builds on Linux (native library, wasm, site, Linux and
macOS wheels, an installed-wheel launcher check) and runs the test suite and the
headless-Chrome app test on an Apple M1 GPU. Pushing a tag `vX.Y.Z` runs
`release.yml`: all six wheels are built and tested on Linux, Windows and macOS,
then published to PyPI by trusted publishing, deployed to GitHub Pages and
attached to a GitHub Release. The tag must equal `__version__` in
`python/zxtalori/__init__.py` and `version` in `src/engine.zig`. The Pages site
ships the pipelines the M1 browser run compiled (`engine/pipelines.json`), so a
first visit starts compiling every kernel at load.

## Conventions

- Scalar-first quaternions and right action `q*g`; I uses the final research
  frame with five-fold axes along z and (2,0,1).
- C/D construction follows the MTEX port with its dihedral polar-parity
  repair. Actual counts are reported rather than truncating a product lattice.
- Preserve orbit/antipodal invariance and distinguish S3 from SO3 angles.
- Keep all vectorizable work in WGSL; Zig schedules and performs irregular
  bookkeeping and f64 final reductions. Keep scientific logic out of adapters.
- Validate precision tricks on the actual shader compiler. An algebraic
  double-single formula is not enough if optimization erases its low terms.
- Browsers inline every WGSL function call, so a large function called from
  several sites multiplies Direct3D compile time; route expensive paths through
  one call site.
- Interactive use is validated to about 100k orientations (near-field
  relaxation, live covering). One million remains a design goal.
- GPU triangulation errors must remain explicit. Never silently substitute a
  CPU triangulator or publish a radius from an unchecked partial shell.
