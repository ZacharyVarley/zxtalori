# zxtalori

Orientation grids on SO(3)/K: generate, measure, relax and lower the covering radius on the GPU,
in the browser or from Python.

**[Open the web app](https://zacharyvarley.github.io/zxtalori/)**: it runs on your GPU inside the
page (Chrome or Edge 137+, which have WebGPU and WebAssembly promise integration). It opens on an
octahedral grid; pick a group and a count, then relax or optimize the covering.

## What it does

- **Generates** samples for all eleven proper point groups: the MTEX equispaced product grids for
  C2–C6 and D2–D6 (with a dihedral polar-parity repair and orbit-aware seam deduplication), FCC
  lattice transports for T, O and I, seeded Haar-random samples, and exact counts from 32 to
  1,000,000 (a smaller grid filled at its largest empty Delaunay circumcentres).
- **Measures** minimum separation, mean geodesic repulsion energy, Riesz s = 1, 2, 3, Green and
  logarithmic energy ratios, and the **covering radius**: the largest empty cap, from a GPU
  Delaunay triangulation of the fundamental zone and a symmetry shell (a port of Gao and Chen's
  parallel 3D Delaunay method) with exact predicates and a shell-completeness certificate.
- **Relaxes** by Quey–Villani–Maurice geodesic repulsion with Barzilai–Borwein steps. Forces sum
  over all pairs, or over each point's near field (about 7,500 neighbours within 20 mean
  orientation radii, faded smoothly), which costs time linear in the count.
- **Lowers the covering radius** by disjoint local moves around the largest empty caps, judged
  against an exact triangulation that is updated in place as points move. An optional
  separation floor keeps every pair apart, and the best sample is kept.
- **Shows** the sample in homochoric, Rodrigues or axis–angle space, in the fundamental zone or
  all of SO(3), with points colored by symmetry zone, nearest-neighbour distance, largest
  adjacent cap or Voronoi face count, and the Delaunay edges on request. Every measure has a
  history plot.
- **Reads and writes** JSON and CSV quaternions (scalar first, right action `q*g`).

One engine (Zig and WGSL) runs natively through wgpu-native and as WebAssembly on WebGPU, so the
app and the Python package do the same thing. Symmetry images are evaluated implicitly; the
expanded orbit is never stored.

## Python

```bash
pip install zxtalori
python -m zxtalori          # the same app, served locally
```

```python
import zxtalori as zo

with zo.generate("O", h=6) as sample:          # 1,043 orientations
    sample.relax(30)
    print(sample.measure()["separation_deg"], sample.covering()["covering_radius_deg"])
    sample.optimize_covering(2000)             # lowers it; the best sample is kept
    q = sample.quaternions                     # NumPy (N, 4), scalar first
    sample.save("octahedral.json")             # opens in the web app

with zo.haar("O", count=100_000, seed=193) as sample:
    sample.relax(100, neighbourhood=20)        # near-field forces; 0 for all pairs
    sample.covering()
    sizes = sample.point_values("covering")    # per point: largest adjacent cap (degrees)

with zo.generate("D2", count=1000, exact=True, fixed_identity=True) as sample:
    sample.relax(30)                           # identity included and held fixed
```

`h` sets the T/O/I lattice resolution and `count` the approximate C/D budget; `exact=True`
reaches `count` exactly for any group. Separation and covering radius are SO(3) rotation angles
in degrees; the repulsion objective uses S³ geodesic distances in radians. Lower repulsion energy
does not by itself certify a lower covering radius.

## More

- [Methods](docs/methods.md): generation, measures, relaxation and covering optimization, with
  conventions and numerical limits; [workbench methods](docs/workbench-methods.md) for zone
  membership and exact counts.
- [GPU Delaunay triangulation](docs/gpu-delaunay.md): the parallel construction after Gao and Chen,
  exact predicates and the covering certificate.
- [CONTRIBUTING.md](CONTRIBUTING.md): building, testing, conventions.

Validated natively and in the browser through about 100,000 orientations; CI runs the test suite
on Apple M1 (Metal). One million orientations remains a
design goal.

MIT ([LICENSE](LICENSE)). Third-party components keep their notices
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). The wheels bundle
[wgpu-native](https://github.com/gfx-rs/wgpu-native) (MIT / Apache-2.0).
