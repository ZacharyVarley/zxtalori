"""Run inside an isolated environment after installing the platform wheel.

Exercises installed native code and packaged web resources, without source imports.
"""
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import zxtalori as zo


def receipt(root, name):
    """Validation receipts stay local, in the ignored research/validation folder."""
    folder = root / "research/validation"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def main():
    assert "site-packages" in str(Path(zo.__file__)).lower(), zo.__file__
    with zo.generate("O", h=6) as transported:
        assert len(transported) == 1043
        assert abs(transported.measure()["log_ratio"] - 0.2775034) < 2e-5
        covering=transported.covering()
        assert covering['triangulation_backend']=='wgsl' and covering['covering_current']
        assert abs(covering['covering_radius_deg']-7.29456)<2e-4
        transported.relax(1)
        assert not transported.stats['covering_current']
    with zo.generate("D4", count=1024) as sample:
        initial = sample.measure()
        assert len(sample) == 938 and initial["separation_deg"] > 0.1
        sample.relax(steps=3)
        after = sample.stats
        assert after["quey_energy"] <= initial["quey_energy"] + 1e-6
        with tempfile.TemporaryDirectory() as temp:
            for extension in ("json", "npz"):
                path = sample.save(Path(temp) / f"sample.{extension}")
                with zo.load(path) as loaded:
                    assert len(loaded) == len(sample)
                    np.testing.assert_allclose(loaded.measure()["quey_energy"], after["quey_energy"], rtol=2e-6)
    with zo.generate("D2", count=1000, exact=True, fixed_identity=True) as sample:
        assert len(sample)==1000
        index=sample.stats["identity_index"]
        sample.relax(3)
        np.testing.assert_array_equal(sample.quaternions[index],[1,0,0,0])
        q=sample.quaternions.astype(np.float64)
        assert np.max(np.abs(q@sample.operations.astype(np.float64).T)-q[:,:1])<1e-6
    process = subprocess.Popen([sys.executable, "-I", "-m", "zxtalori", "--no-browser", "--port", "4181"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(40):
            try:
                with urllib.request.urlopen("http://127.0.0.1:4181/", timeout=1) as response:
                    assert response.version == 11, "Browser fetch compatibility requires the tested HTTP/1.1 response"
                    html = response.read()
                break
            except (urllib.error.URLError, TimeoutError):
                if process.poll() is not None:
                    raise RuntimeError(process.stderr.read().decode())
                time.sleep(0.1)
        else:
            raise RuntimeError("Packaged launcher did not start")
        assert b'Full SO(3)' in html and b'id="fix-identity"' in html
        for name in ("engine/zxtalori.js", "engine/zxtalori.wasm", "app.js", "style.css", "geometry.js", "covering-policy.js"):
            started=time.perf_counter()
            with urllib.request.urlopen("http://127.0.0.1:4181/" + name, timeout=15) as response:
                size=len(response.read())
                assert response.status == 200 and size > 100
            print(f"{name}: {size} bytes in {time.perf_counter()-started:.3f}s",flush=True)
    finally:
        process.terminate()
        process.communicate(timeout=10)
    root=Path(__file__).resolve().parents[1]
    package=Path(zo.__file__).resolve()
    report={"package": package.relative_to(root).as_posix() if package.is_relative_to(root) else package.name, "native_generate_measure_relax": True, "native_transport_measures": True,
            "exact_count_identity_zone": True, "gpu_covering_current_and_stale": True, "json_npz_roundtrip": True, "packaged_app_resources": True, "passed": True}
    print(json.dumps(report,indent=2))
    output=receipt(Path(__file__).resolve().parents[1], "package-validation.json")
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
