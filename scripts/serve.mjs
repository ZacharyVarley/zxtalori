#!/usr/bin/env node
// Loopback dev server. Research archives and development caches are not served.
import {createServer} from 'node:http';
import {createReadStream,existsSync,statSync} from 'node:fs';
import {resolve,join,extname,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
const ROOT=resolve(fileURLToPath(new URL('..',import.meta.url)));
const args=process.argv.slice(2),arg=name=>args.includes(name)?args[args.indexOf(name)+1]:null;
const port=Number(arg('--port')||4180),built=arg('--dir'),base=built?resolve(ROOT,built):join(ROOT,'web/app');
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.wasm':'application/wasm','.json':'application/json','.svg':'image/svg+xml','.png':'image/png'};
createServer((req,res)=>{let path;try{path=decodeURIComponent(new URL(req.url,'http://localhost').pathname);}catch{res.writeHead(400).end('bad path');return;}let file;
  if(!built&&path==='/engine/zxtalori.js')file=join(ROOT,'web/zxtalori.js');
  else if(!built&&path==='/engine/zxtalori.wasm')file=join(ROOT,'zig-out/web/zxtalori.wasm');
  else if(!built&&path==='/qa/')file=join(ROOT,'tests/browser-parity.html');
  else if(!built&&path==='/qa/reference.json')file=join(ROOT,'tests/fixtures/browser-reference.json');
  else if(!built&&path==='/qa/covering-reference.json')file=join(ROOT,'tests/fixtures/covering-reference.json');
  else{if(path.endsWith('/'))path+='index.html';file=resolve(base,'.'+path);if(!file.startsWith(base+sep)){res.writeHead(403).end('forbidden');return;}}
  if(!existsSync(file)||!statSync(file).isFile()){res.writeHead(404).end('not found');return;}
  res.writeHead(200,{'Content-Type':types[extname(file)]||'application/octet-stream','Cache-Control':'no-store'});createReadStream(file).pipe(res);
}).listen(port,'127.0.0.1',()=>console.log(`zxtalori: http://127.0.0.1:${port}/`));
