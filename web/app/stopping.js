// When Auto ends each stage. They are kept apart from the GPU work so they can be tested on their own.

// Repulsion ends when the lowest energy gains less than RELAX_GAIN per step, relative, over two
// windows of RELAX_WINDOW steps in a row: its Barzilai-Borwein steps spike now and then, and the
// energy recovers.
export const RELAX_WINDOW=100,RELAX_GAIN=1e-9;
// Returns a function of (steps done, energy) that is true once repulsion has plateaued.
export function repulsionPlateau(){let lowest=Infinity,reference=Infinity,referenceStep=0,calm=0;
  return (done,energy)=>{if(Number.isFinite(energy)){lowest=Math.min(lowest,energy);
      if(done-referenceStep>=RELAX_WINDOW){calm=(reference-lowest)/Math.abs(lowest)/(done-referenceStep)<RELAX_GAIN?calm+1:0;reference=lowest;referenceStep=done;}}
    return calm>=2;};}

// Covering is judged only at the engine's kicks. A kick only comes once a descent has plateaued, so a
// sample just kicked is never judged: it ends once COVER_QUIET_KICKS kicks have passed since the best
// radius last gained COVER_GAIN, relative.
export const COVER_QUIET_KICKS=4,COVER_GAIN=5e-4;
// Returns a function of (best radius, kicks so far) that is true once covering has plateaued.
export function coveringPlateau(){let bestSeen=Infinity,kicksAtBest=0;
  return (best,kicks)=>{if(best<bestSeen*(1-COVER_GAIN)){bestSeen=best;kicksAtBest=kicks;}return kicks-kicksAtBest>=COVER_QUIET_KICKS;};}
