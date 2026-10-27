#!/usr/bin/env python3
"""The `zxtalori` app launcher serves every file the web app loads, with the right types.

    python tests/test_app.py        (installed wheel, or a checkout after `zig build wasm`)

No GPU and no browser: this only checks the local server. Under pytest it runs as one test.
"""
import sys
import urllib.request
from pathlib import Path

try:
    import zxtalori.app as app
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
    import zxtalori.app as app

TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".wasm": "application/wasm"}
REQUIRED = ["/engine/zxtalori.js", "/engine/zxtalori.wasm", "/app.js", "/style.css"]


def check_app():
    """GETs every file of the app; returns the paths that failed."""
    url = app.serve(port=0, open_browser=False, block=False)
    failures = []

    def check(path, ctype, min_size, prefix=None):
        try:
            with urllib.request.urlopen(url + path.lstrip("/"), timeout=20) as r:
                status, ct, body = r.status, r.headers.get("Content-Type", ""), r.read()
            ok = status == 200 and ct.startswith(ctype) and len(body) >= min_size and (prefix is None or body.startswith(prefix))
            info = f"{status} {ct} {len(body)} B"
        except Exception as e:
            ok, info = False, repr(e)
        print(f"{'ok  ' if ok else 'FAIL'} {path}  {info}")
        if not ok:
            failures.append(path)

    check("/", "text/html", 1000)
    root, extra = app.site()
    paths = {"/" + p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} | set(extra)
    for p in REQUIRED:
        if p not in paths:
            print(f"FAIL {p}  missing from the app")
            failures.append(p)
    for p in sorted(paths):
        ext = p[p.rfind("."):]
        check(p, TYPES.get(ext, ""), 1, prefix=b"\0asm" if ext == ".wasm" else None)
    return failures


def test_app_serves_every_file():
    assert not check_app()


if __name__ == "__main__":
    failures = check_app()
    print("app", "OK" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)
