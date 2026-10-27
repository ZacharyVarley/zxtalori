#!/usr/bin/env node
// Buildless static web package. No application logic is duplicated in JavaScript.
import {cpSync,existsSync,mkdirSync} from 'node:fs';
import {resolve,join} from 'node:path';
import {fileURLToPath} from 'node:url';
const ROOT=resolve(fileURLToPath(new URL('..',import.meta.url)));
const OUT=join(ROOT,'dist/web'),wasm=join(ROOT,'zig-out/web/zxtalori.wasm');
if(!existsSync(wasm))throw new Error('Build the Zig wasm target before packaging the web app.');
mkdirSync(join(OUT,'engine'),{recursive:true});
for(const path of ['index.html','style.css','app.js','geometry.js','covering-policy.js'])cpSync(join(ROOT,'web/app',path),join(OUT,path));
cpSync(join(ROOT,'web/zxtalori.js'),join(OUT,'engine/zxtalori.js'));
cpSync(wasm,join(OUT,'engine/zxtalori.wasm'));
console.log(`Built ${OUT}`);
