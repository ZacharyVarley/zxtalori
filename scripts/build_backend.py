"""PEP 517 build backend, so `pip install .` works from a checkout (stdlib only).

It builds the wheel for this machine with scripts/build_wheels.py: Zig (fetched into .deps/ on
first use, or $ZIG) compiles the library and zxtalori.wasm, and wgpu-native for this platform is
fetched and checksummed. Release wheels for every platform come from build_wheels.py directly.
"""
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import build_wheels as bw
from fetch_deps import fetch_zig, host_wheel_tag


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    import os
    if not os.environ.get("ZIG"):
        fetch_zig()
    whl = bw.build_wheel(host_wheel_tag())
    shutil.copy2(whl, Path(wheel_directory) / whl.name)
    return whl.name


def build_sdist(sdist_directory, config_settings=None):
    """A self-contained source release, including checkouts with no Git commit yet."""
    base = f"zxtalori-{bw.VERSION}"
    files = [Path(name) for name in ("README.md", "CONTRIBUTING.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "build.zig", "requirements-dev.in", "requirements-dev.lock")]
    for folder in ("src", "shaders", "python", "web", "scripts", "licenses", "docs", "tests"):
        files.extend(p.relative_to(bw.ROOT) for p in (bw.ROOT / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    out = Path(sdist_directory) / f"{base}.tar.gz"
    with tarfile.open(out, "w:gz", format=tarfile.PAX_FORMAT) as t:
        for f in files:
            p = bw.ROOT / f
            if p.is_file():
                t.add(p, f"{base}/{f}")
        info = bw.METADATA.encode()
        ti = tarfile.TarInfo(f"{base}/PKG-INFO")
        ti.size = len(info)
        t.addfile(ti, io.BytesIO(info))
    return out.name
