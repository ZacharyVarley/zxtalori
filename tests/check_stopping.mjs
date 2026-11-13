import assert from 'node:assert/strict';
import {repulsionPlateau,coveringPlateau,RELAX_WINDOW,COVER_QUIET_KICKS} from '../web/app/stopping.js';

// Repulsion: steps in batches of 10, ending at the first step the rule says it has plateaued.
const relax=(energy,steps=5000)=>{const plateaued=repulsionPlateau();for(let done=10;done<=steps;done+=10)if(plateaued(done,energy(done)))return done;return null;};
assert.equal(relax(()=>1),3*RELAX_WINDOW);                       // flat: the first window only sets the reference
assert.equal(relax((s)=>1+1e-4*Math.exp(-s/10)),3*RELAX_WINDOW); // gains that fade out within the first window
assert.equal(relax((s)=>1+1e-4*Math.exp(-s/50)),6*RELAX_WINDOW); // gains that fall under 1e-9 per step by step 500
assert.equal(relax((s)=>2-1e-6*s),null);                          // a steady gain of about 5e-7 per step never stops
assert.equal(relax((s)=>(s%200===0?5:1)),3*RELAX_WINDOW);        // spikes in the energy do not hold it up
assert.equal(relax(()=>null),null);                               // no energy measured, no judgement
const late=repulsionPlateau();let at=null;
for(let done=10;done<=2000;done+=10)if(late(done,done<1000?2-1e-6*done:1)&&at===null)at=done;
assert(at>1000&&at<=1000+3*RELAX_WINDOW);                         // stops soon after the gains end

// Covering: the best radius and the kick count at each batch.
const cover=(rows)=>{const plateaued=coveringPlateau();return rows.findIndex(([best,kicks])=>plateaued(best,kicks));};
assert.equal(cover([[10,0],[9,0],[9,1],[9,2],[9,3],[9,4],[9,5]]),5);              // four kicks without a gain
assert.equal(cover([[10,0],[9,1],[8,2],[7,3],[6,4],[5,5],[4,6],[3,7]]),-1);       // a gain at every kick
assert.equal(cover([[10,0],[9.999,1],[9.998,2],[9.997,3],[9.996,4]]),4);          // gains under 5e-4 do not count
assert.equal(cover([[10,0],[10,3],[9,3],[9,6],[9,7]]),4);                         // the count starts at the last gain
assert.equal(COVER_QUIET_KICKS,4);
console.log('Auto stopping: repulsion windows, spikes, covering kicks without a gain passed.');
