"""Build the source archive in a fresh folder, reusing only cached toolchains.

This catches source-manifest omissions that an in-checkout wheel build misses.
It does not need the research archive, tests, Git metadata or existing outputs.
The extracted build and wheel remain in .deps/ for inspection/installation.
"""
import json
from email.parser import Parser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def receipt(root, name):
    """Validation receipts stay local, in the ignored research/validation folder."""
    folder = root / "research/validation"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "dist/zxtalori-0.1.0.tar.gz"
    cache = ROOT / ".deps"
    staging = Path(tempfile.mkdtemp(prefix="source-check-", dir=cache)).resolve()
    assert staging.is_relative_to(cache.resolve())
    with tarfile.open(archive) as source:
        names = source.getnames()
        prefix = names[0].split("/")[0]
        assert f"{prefix}/build.zig" in names, "Source archive is missing build.zig"
        metadata = Parser().parsestr(source.extractfile(f"{prefix}/PKG-INFO").read().decode())
        for license_file in metadata.get_all("License-File", []):
            assert f"{prefix}/{license_file}" in names, f"Missing declared license: {license_file}"
        assert not any("/research/" in name or "/.deps/" in name for name in names)
        source.extractall(staging, filter="data")
    checkout = staging / prefix
    native_cache = cache / "wgpu-native/win_amd64"
    if sys.platform != "win32" or not native_cache.is_dir():
        raise RuntimeError("This validation currently uses the Windows x64 cached toolchain")
    shutil.copytree(native_cache, checkout / ".deps/wgpu-native/win_amd64")
    output = checkout / "built-wheel"
    output.mkdir()
    env = os.environ.copy()
    env["ZIG"] = str(cache / "zig-0.16.0/zig.exe")
    env["ZIG_GLOBAL_CACHE_DIR"] = str(cache / "zig-global-cache")
    subprocess.run([sys.executable, "-c", "import sys;sys.path.insert(0,'scripts');import build_backend;print(build_backend.build_wheel('built-wheel'))"],
                   cwd=checkout, env=env, check=True)
    wheel, = output.glob("*.whl")
    with zipfile.ZipFile(wheel) as built:
        for name in ("zxtalori/_lib/zxtalori.dll", "zxtalori/web/engine/zxtalori.wasm", "zxtalori/web/app.js"):
            assert len(built.read(name)) > 100
        info, = [name for name in built.namelist() if name.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(built.read(info).decode())
        for license_file in metadata.get_all("License-File", []):
            assert len(built.read(info.removesuffix("METADATA") + "licenses/" + license_file)) > 100
    report = {"archive": str(archive.relative_to(ROOT)), "source_entries": len(names),
              "fresh_source_build": True, "declared_licenses_present": True, "research_archive_required": False,
              "wheel": str(wheel.relative_to(ROOT)), "passed": True}
    print(json.dumps(report, indent=2))
    (receipt(ROOT, "source-package-validation.json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
