"""Loopback-only browser tests against source assets and the current WASM build."""
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from zxtalori.app import _Handler

FILES={
    '/engine/zxtalori.js':ROOT/'web/zxtalori.js',
    '/engine/zxtalori.wasm':ROOT/'zig-out/web/zxtalori.wasm',
    '/qa/':ROOT/'tests/browser-parity.html',
    '/qa/covering':ROOT/'tests/browser-covering.html',
    '/qa/compile':ROOT/'tests/browser-gao-compile.html',
    '/qa/reference.json':ROOT/'tests/fixtures/browser-reference.json',
    '/qa/covering-reference.json':ROOT/'tests/fixtures/covering-reference.json',
}
for name in ['gao_storage','gao_topology','gao_geometry','dt_math','dt_double','dt_integer','gao_predicates','gao_double','gao_post']:
    FILES[f'/qa/shaders/{name}.wgsl']=ROOT/f'shaders/{name}.wgsl'
if __name__=='__main__':
    port=int(sys.argv[1]) if len(sys.argv)>1 else 4183
    server=ThreadingHTTPServer(('127.0.0.1',port),partial(_Handler,directory=str(ROOT/'web/app'),files=FILES))
    print(f'Browser validation: http://127.0.0.1:{port}/qa/covering',flush=True)
    server.serve_forever()
