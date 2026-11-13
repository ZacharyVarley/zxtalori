#!/usr/bin/env node
// Buildless static web package. No application logic is duplicated in JavaScript.
import {cpSync,existsSync,mkdirSync,readdirSync} from 'node:fs';
import {resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
const ROOT=resolve(fileURLToPath(new URL('..',import.meta.url)));
const OUT=join(ROOT,'dist/web'),wasm=join(ROOT,'zig-out/web/zxtalori.wasm');
if(!existsSync(wasm))throw new Error('Build the Zig wasm target before packaging the web app.');
mkdirSync(join(OUT,'engine'),{recursive:true});
// Every file of web/app, as the wheel carries it (scripts/build_wheels.py), so a new module cannot be left out.
for(const path of readdirSync(join(ROOT,'web/app')))if(!path.startsWith('.'))cpSync(join(ROOT,'web/app',path),join(OUT,path),{recursive:true});
cpSync(join(ROOT,'web/zxtalori.js'),join(OUT,'engine/zxtalori.js'));
cpSync(wasm,join(OUT,'engine/zxtalori.wasm'));
console.log(`Built ${OUT}`);
