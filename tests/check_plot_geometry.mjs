import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {polyhedron,boundary,displayPoints} from '../web/app/geometry.js';
const table=JSON.parse(readFileSync('tests/fixtures/plot-ops.json','utf8'));
const groups=['C2','C3','C4','C6','D2','D3','D4','D6','T','O','I'];
for(const [index,group] of groups.entries()){
 const ops=Float32Array.from(table[group].flat());
 if(index>=4){const p=polyhedron(ops,index);const expected=index===4?[8,12]:index===8?[6,12]:index===9?[24,36]:index===10?[20,30]:null;if(expected)assert.deepEqual([p.vertices.length,p.edges.length],expected,group);const g=boundary(ops,index,'homochoric');assert.equal(g.wire.length,p.edges.length);assert(g.extent>0);if(group==='D2'){const edge=g.wire[0],mid=edge[20],straight=edge[0].map((x,i)=>(x+edge[40][i])/2);assert(Math.hypot(...mid)>Math.hypot(...straight)+.01,'bulged cube edge');}console.log(group,p.vertices.length,'vertices',p.edges.length,'edges');}
 const q=new Float32Array(1000*4);for(let i=0;i<1000;i++)q[i*4]=1;const small=displayPoints(q,ops,'homochoric',true,1000);assert.equal(new Set(small.operation).size,ops.length/4,group+' subset covers all operators');assert.equal(small.xyz.length,3*small.count);
}
console.log('Boundary topology, homochoric curvature and symmetry subsampling passed.');

