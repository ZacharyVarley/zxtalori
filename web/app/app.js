import {Zxtalori} from './engine/zxtalori.js';
import {multiply,boundary,displayPoints,project} from './geometry.js';
import {shouldCheckCovering} from './covering-policy.js';

const $=id=>document.getElementById(id);
const choice=name=>document.querySelector(`input[name="${name}"]:checked`)?.value;
const setChoice=(name,value)=>{const input=document.querySelector(`input[name="${name}"][value="${value}"]`);if(input)input.checked=true;};
const GROUPS=['C2','C3','C4','C6','D2','D3','D4','D6','T','O','I'];
const SYMBOLS=GROUPS.map(g=>g.replace(/\d/,d=>'₀₁₂₃₄₅₆₇₈₉'[d]));
const ORDER=[2,3,4,6,4,6,8,12,12,24,60];
const NAMES=['Monoclinic','Trigonal','Tetragonal','Hexagonal','Orthorhombic','Trigonal','Tetragonal','Hexagonal','Tetrahedral','Octahedral','Icosahedral'];
const GENERATORS=['MTEX equispaced grid','MTEX equispaced grid','MTEX equispaced grid','MTEX equispaced grid','MTEX equispaced grid','MTEX equispaced grid','MTEX equispaced grid','MTEX equispaced grid','Octahedral FCC transport','Truncated-cube FCC transport','FCC cube → ball → zone'];
const state={engine:null,points:null,q:new Float32Array(),original:null,group:9,operations:new Float32Array(),stats:{},history:[],busy:false,stop:false,selected:-1,selectedOperation:0,provenance:null,coverModel:null,coverEnded:0,stepMilliseconds:5,coverFailed:false,mesh:null,meshIteration:-1,series:[]};
const METRICS=[{key:'energy',label:'Repulsion'},{key:'covering_radius_deg',label:'Covering radius'},{key:'covering_excess',label:'Covering excess'},{key:'mesh_ratio',label:'Mesh ratio'},{key:'separation_deg',label:'Separation'},{key:'riesz_1',label:'Riesz 1'},{key:'riesz_2',label:'Riesz 2'},{key:'riesz_3',label:'Riesz 3'},{key:'green_ratio',label:'Green'},{key:'log_ratio',label:'Log'}];
try{state.series=JSON.parse(localStorage.getItem('zxtalori-series')||'[]').filter(k=>METRICS.some(m=>m.key===k));}catch{}
if(!state.series.length)state.series=['energy'];
const countFmt=new Intl.NumberFormat('en-US');
function number(value,digits=5){if(value===null||value===undefined||typeof value==='boolean'||!Number.isFinite(+value))return '—';const n=+value;return n===0?'0':Math.abs(n)>=1e6||Math.abs(n)<1e-4?n.toExponential(3):n.toLocaleString('en-US',{maximumFractionDigits:digits});}
function value(...keys){for(const key of keys)if(state.stats[key]!==undefined&&state.stats[key]!==null)return state.stats[key];return null;}
function groupId(){return GROUPS.indexOf(choice('group'));}
// Log pane: one line per finished operation, with its timing and result.
const logEntries=[];
function log(label,text='',{level='info'}={}){
  const list=$('log-list'),stuck=list.scrollHeight-list.scrollTop-list.clientHeight<24,item=document.createElement('li');item.dataset.level=level;
  const time=document.createElement('time'),now=new Date();time.textContent=now.toLocaleTimeString('en-GB')+'.'+String(now.getMilliseconds()).padStart(3,'0');
  const message=document.createElement('span');message.className='msg';const bold=document.createElement('b');bold.textContent=label;message.append(bold,document.createTextNode(text?` ${text}`:''));
  item.append(time,message);list.append(item);logEntries.push(item);while(logEntries.length>400)logEntries.shift().remove();
  $('log-count').textContent=`${logEntries.length} ${logEntries.length===1?'entry':'entries'}`;if(stuck)list.scrollTop=list.scrollHeight;
  if(!logShown()){unread++;unreadError||=level==='error';showBadge();}
}
$('log-clear').onclick=()=>{for(const item of logEntries.splice(0))item.remove();$('log-count').textContent='';};
// The bottom pane shows History by default; the Log tab counts entries added while hidden.
let unread=0,unreadError=false;
const logShown=()=>$('tab-log').getAttribute('aria-selected')==='true';
function showBadge(){$('log-badge').hidden=!unread;$('log-badge').textContent=unread>99?'99+':String(unread);$('log-badge').classList.toggle('error',unreadError);$('tab-log').title=unread?`${unread} new log ${unread===1?'entry':'entries'}${unreadError?', including a failure':''}`:'';}
function selectPane(log){
  $('tab-history').setAttribute('aria-selected',!log);$('tab-log').setAttribute('aria-selected',log);$('tab-history').tabIndex=log?-1:0;$('tab-log').tabIndex=log?0:-1;
  $('history-panel').hidden=log;$('log-panel').hidden=!log;$('history-series').hidden=log;$('log-count').hidden=!log;$('log-clear').hidden=!log;
  if(log){unread=0;unreadError=false;showBadge();$('log-list').scrollTop=$('log-list').scrollHeight;}else drawHistory();
}
$('tab-history').onclick=()=>selectPane(false);$('tab-log').onclick=()=>selectPane(true);
for(const id of ['tab-history','tab-log'])$(id).onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){const log=!logShown();selectPane(log);$(log?'tab-log':'tab-history').focus();e.preventDefault();}};
const seconds=ms=>`${(ms/1000).toFixed(ms<10000?2:1)} s`;
function status(text,detail=''){$('status').classList.remove('error');$('status').replaceChildren(Object.assign(document.createElement('span'),{className:'status-dot'}),document.createTextNode(text));$('status-detail').textContent=detail||'Zig + WGSL · WebGPU';}
function showError(error,label='Error'){log(label+' failed',error.message||String(error),{level:'error'});$('error-message').textContent=error.message||String(error);$('error-banner').hidden=false;status('Operation failed','See message');$('status').classList.add('error');}
$('dismiss-error').onclick=()=>$('error-banner').hidden=true;
function progress(fraction){$('run-progress').hidden=fraction===null;if(fraction!==null)$('run-progress-bar').style.width=`${100*fraction}%`;}
function busy(on){state.busy=on;document.body.classList.toggle('busy',on);const ready=!!state.engine,has=state.q.length>0;for(const id of ['generate-button','group','count-mode','semi-edge','target-count','random-seed','shuffle-seed','import-button','add-identity','fix-identity','iterations','objective','forces','free-set','separation-floor','floor-from-current'])$(id).disabled=on||!ready;for(const id of ['relax-button','measure-button','cover-button','export-button','load-dt'])$(id).disabled=on||!ready||!has;$('restore-button').disabled=on||!state.original;$('stop-button').disabled=!on;}
function updateGroup(){const id=groupId(),mode=choice('count-mode'),exact=mode==='exact',random=mode==='random',product=id<8;$('semi-edge').max='64';if(+$('semi-edge').value>+$('semi-edge').max)$('semi-edge').value=$('semi-edge').max;$('group-order').textContent=`${NAMES[id]} · ${ORDER[id]} rotations`;$('h-field').hidden=product||exact||random;$('count-field').hidden=!product&&!exact&&!random;$('seed-field').hidden=!random;$('target-label').textContent=exact||random?'Orientations':'Requested count';$('count-note').textContent=random?'Independent Haar samples, folded into the selected zone.':exact?'A smaller grid is filled at the largest Delaunay circumcentres.':product?'MTEX equispaced grid; ring spacing determines the achieved count.':`${GENERATORS[id]}; h sets the lattice density.`;}
$('count-mode').onchange=updateGroup;
$('shuffle-seed').onclick=()=>$('random-seed').value=crypto.getRandomValues(new Uint32Array(1))[0];
$('group').onchange=updateGroup;
function resetHistory(){state.history=[];state.coverFailed=false;state.coverEnded=0;drawHistory();}
function updateSnapshot(snapshot,{original=false}={}){if(state.mesh&&state.meshIteration!==snapshot.stats?.iteration)$('dt-note').textContent='DT is stale; compute again for the current sample.';state.q=snapshot.quaternions;state.operations=snapshot.operations;state.points=snapshot.points||null;state.stats=snapshot.stats||{};state.group=GROUPS.indexOf(state.stats.group)>=0?GROUPS.indexOf(state.stats.group):state.group;if(original){state.mesh=null;state.original=state.q.slice();state.selected=-1;resetHistory();}$('view-empty').hidden=state.q.length>0;updateMetrics();view.rebuild();view.draw();busy(state.busy);}
function updateMetrics(){const n=state.q.length/4,id=state.group,order=+value('order','group_order')||ORDER[id];$('point-count').textContent=n?countFmt.format(n):'—';$('extended-count').textContent=n?countFmt.format(order*n):'—';$('sample-badge').textContent=n?`${countFmt.format(n)} orientations`:'No sample';$('sample-heading').textContent=`${SYMBOLS[id]} · ${NAMES[id]}`;$('iteration-count').textContent=value('iteration','iterations')||0;const sep=value('separation_deg','separation');$('separation').textContent=sep===null?'—':`${number(sep,4)}°`;$('energy').textContent=number(value('quey_energy','energy'),7);for(const [id,keys]of Object.entries({'riesz-1':['riesz_1','riesz1'],'riesz-2':['riesz_2','riesz2'],'riesz-3':['riesz_3','riesz3'],'green':['green','green_ratio'],'log':['log','logarithmic','log_ratio']}))$(id).textContent=number(value(...keys),5);$('precision-note').textContent=value('precision')||'';$('construction-note').textContent=value('inserted_count')?`${countFmt.format(value('base_count'))} grid + ${countFmt.format(value('inserted_count'))} inserted`:'';updateCoveringNote();updateSelection();}
function addHistory(){const iteration=+value('iteration')||0;const extra={};for(const k of ['covering_radius_deg','covering_excess','mesh_ratio','separation_deg','riesz_1','riesz_2','riesz_3','green_ratio','log_ratio']){if(['riesz_1','riesz_2','riesz_3','green_ratio','log_ratio'].includes(k)&&!value('energy_ratios_current'))continue;if(k==='separation_deg'&&value('measures_current')===false)continue;const v=value(k);if(v!==null&&(!k.startsWith('covering')&&k!=='mesh_ratio'||value('covering_current')))extra[k]=v;}const energy=value('measures_current')===false?null:value('quey_energy','energy');if(Object.keys(extra).length||energy!==null){const entry={iteration,...(energy!==null?{energy:+energy}:{}),...extra};if(state.history.at(-1)?.iteration===entry.iteration)state.history[state.history.length-1]={...state.history.at(-1),...entry};else state.history.push(entry);drawHistory();}}
async function withBusy(label,operation){if(state.busy||!state.engine)return;busy(true);state.stop=false;$('error-banner').hidden=true;status(label,'GPU working');const started=performance.now();try{await operation();const elapsed=performance.now()-started;status(state.stop?'Stopped; current sample retained':'Ready',`${(elapsed/1000).toFixed(2)} s · ${countFmt.format(state.q.length/4)} orientations`);log(state.stop?`${label} stopped`:label,`${seconds(elapsed)} · ${countFmt.format(state.q.length/4)} orientations`,{level:state.stop?'warn':'info'});}catch(e){if(state.stop){status('Stopped','Current sample retained');log(`${label} stopped`,'current sample retained',{level:'warn'});}else showError(e,label);}finally{progress(null);busy(false);}}
async function measure(){status('Measuring pair energies and separation','GPU working');const started=performance.now();updateSnapshot(await state.engine.measure());addHistory();const n=state.q.length/4;log('Measured',`${seconds(performance.now()-started)} · separation ${number(value('separation_deg'),4)}° · Quey energy ${number(value('quey_energy','energy'),7)} · ${value('energy_ratios_current')?'Riesz, Green and log ratios current':`ratios skipped: ${countFmt.format(n)} orientations exceed the ratio limit of ${countFmt.format(value('ratio_limit')||0)}`}`);}
$('generate-button').onclick=()=>withBusy('Generating sample',async()=>{const id=groupId(),h=+$('semi-edge').value,target=+$('target-count').value,seed=193,mode=choice('count-mode'),exact=mode==='exact';if(!Number.isInteger(h)||h<1||h>+$('semi-edge').max)throw new Error(`Choose an integer semi-edge between 1 and ${$('semi-edge').max}.`);if(!Number.isInteger(target)||target<32||target>1000000)throw new Error('Choose a count between 32 and 1,000,000.');if(mode==='random'&&(!Number.isInteger(Number($('random-seed').value))||Number($('random-seed').value)<0||Number($('random-seed').value)>4294967295))throw new Error('Choose an integer seed from 0 to 4,294,967,295.');let snapshot=mode==='random'?await state.engine.random(id,target-+$('add-identity').checked,Number($('random-seed').value)>>>0):exact?await state.engine.generateExact(id,target,{identity:$('add-identity').checked}):await state.engine.generate(id,h,target,seed);if(!exact&&$('add-identity').checked)snapshot=await state.engine.identity({add:true,fixed:$('fix-identity').checked});else snapshot=await state.engine.identity({fixed:$('fix-identity').checked});updateSnapshot(snapshot,{original:true});state.provenance={group:GROUPS[id],seed:value('seed'),generator:mode==='random'?'Haar random':GENERATORS[id],h:id>=8&&!exact&&mode!=='random'?h:null,target_count:exact||id<8?target:null};log('Generated',`${GROUPS[id]} · ${mode==='random'?'Haar random':exact?'exact count':id<8?'MTEX grid':`FCC grid h=${h}`} · ${countFmt.format(state.q.length/4)} orientations${value('inserted_count')?` (${countFmt.format(value('base_count'))} grid + ${countFmt.format(value('inserted_count'))} inserted)`:''}${value('removed_duplicates')?` · ${countFmt.format(value('removed_duplicates'))} seam duplicates removed`:''}`);if(state.q.length/4<=10000)await measure();await maybeCovering('generation');});
function updateCoveringNote(){
  const r=value('covering_radius_deg'),current=!!value('covering_current'),n=state.q.length/4,order=ORDER[state.group];$('covering-radius').textContent=r===null?'—':`${number(r,4)}°`;$('covering-radius').classList.toggle('stale',r!==null&&!current);
  $('mesh-ratio').textContent=number(value('mesh_ratio'),5);
  $('covering-excess').textContent=number(value('covering_excess'),5);
  $('covering-excess').classList.toggle('stale',r!==null&&!current);
  const mode=$('covering-mode').value,onDemand=mode==='auto'&&!shouldCheckCovering({mode:'auto',phase:'generation',count:n,order,model:state.coverModel});
  $('covering-note').textContent=state.coverFailed?'Check failed; last value retained':r!==null?`${current?'Current':'Last checked'} · step ${value('covering_iteration')}${state.coverModel?` · ${((state.coverModel.elapsedMilliseconds??state.coverModel.milliseconds)/1000).toFixed(2)} s${state.coverModel.cold?' (first check)':''}`:''}${onDemand?' · not automatic for this size':''}`:onDemand?'Not automatic for this size; use Check now':'Not checked';
  $('covering-note').title=mode==='auto'?'Auto uses measured check times. Small, fast samples update during long runs; slower samples update at the end or on demand.':'';
}
async function checkCovering(){
  status('Computing covering radius','GPU triangulation');const started=performance.now(),compile=state.engine.state.stats.compileWaitMs,pipelines=state.engine.state.compiling.total;
  try{const snapshot=await state.engine.covering();const elapsed=performance.now()-started,compileWait=state.engine.state.stats.compileWaitMs-compile;
    state.coverModel={milliseconds:Math.max(10,elapsed-compileWait),elapsedMilliseconds:elapsed,cold:state.engine.state.compiling.total>pipelines,count:snapshot.quaternions.length/4,order:ORDER[state.group]};state.coverEnded=performance.now();state.coverFailed=false;updateSnapshot(snapshot);addHistory();log('Covering',`${seconds(elapsed)}${state.coverModel.cold?' (first check, includes shader compilation)':''} · radius ${number(value('covering_radius_deg'),5)}° · excess ${number(value('covering_excess'),5)} · shell ${countFmt.format(value('covering_shell_points')||0)}+${countFmt.format(value('covering_fence_points')||0)} fence · ${countFmt.format(value('covering_tetrahedra')||0)} tetrahedra · ${countFmt.format(value('covering_exact_predicates')||0)} exact predicates`);
  }catch(error){state.coverFailed=!state.stop;updateCoveringNote();throw error;}
}
async function maybeCovering(phase){
  const iteration=+value('iteration')||0,last=value('covering_iteration');
  if(shouldCheckCovering({mode:$('covering-mode').value,phase,count:state.q.length/4,order:ORDER[state.group],age:last===null?iteration+1:iteration-last,elapsed:performance.now()-state.coverEnded,stepMilliseconds:state.stepMilliseconds,model:state.coverModel,failed:state.coverFailed})){
    try{await checkCovering();}catch(error){if(state.stop)return;log('Covering check failed',error.message||String(error),{level:'error'});$('error-message').textContent=error.message||String(error);$('error-banner').hidden=false;}
  }
}
$('cover-button').onclick=()=>withBusy('Computing covering radius',checkCovering);
$('covering-mode').onchange=updateCoveringNote;
$('measure-button').onclick=()=>withBusy('Measuring sample',measure);
$('restore-button').onclick=()=>withBusy('Restoring generated sample',async()=>{updateSnapshot(await state.engine.set(state.group,state.original));resetHistory();if(state.q.length/4<=10000)await measure();await maybeCovering('generation');});
// Covering optimization runs in engine batches sized to about 300 ms so the plot and
// measures stay live; the engine keeps the best sample and restores it at the end.
const runLength={repulsion:30,covering:1000};
function updateObjective(){const covering=choice('objective')==='covering';$('free-set').hidden=!covering;$('forces').hidden=covering;$('floor-field').hidden=!covering;$('iterations-label').textContent=covering?'Steps':'Iterations';$('iterations').max=covering?'100000':'1000';$('iterations').value=runLength[choice('objective')];$('relax-label').textContent=covering?'Optimize':'Relax';for(const note of document.querySelectorAll('[data-objective]'))note.hidden=note.dataset.objective!==choice('objective');view.draw();}
$('objective').onchange=updateObjective;
$('floor-from-current').onclick=()=>{const separation=value('separation_deg');if(separation!==null)$('separation-floor').value=(0.95*separation).toFixed(3);};
$('iterations').addEventListener('change',()=>{runLength[choice('objective')]=+$('iterations').value;});
async function optimizeCovering(){const steps=+$('iterations').value,mode=+choice('free-set'),floor=+$('separation-floor').value||0;if(!Number.isInteger(steps)||steps<1||steps>100000)throw new Error('Choose 1–100,000 optimization steps.');if(!(floor>=0&&floor<180))throw new Error('Choose a minimum separation from 0 to 180 degrees, or leave it empty.');updateSnapshot(await state.engine.identity({add:$('add-identity').checked,fixed:$('fix-identity').checked}));state.stop=false;progress(0);
  // As in relaxation: batches of about 300 ms of GPU wait time, the next one queued and
  // submitted before the finished one is drawn, so the GPU works while the page draws.
  let done=0,issued=0,batch=10,pending=null;
  const launch=count=>{issued+=count;const waited=state.engine.state.stats.waitMs;return state.engine.optimizeCovering(count,mode,false,floor).then(snapshot=>({snapshot,count,elapsed:state.engine.state.stats.waitMs-waited}));};
  pending=launch(Math.min(batch,steps));
  while(pending){const {snapshot,count,elapsed}=await pending;pending=null;done+=count;batch=Math.max(1,Math.min(500,Math.round(count*300/Math.max(1,elapsed))));
    if(issued<steps&&!state.stop){pending=launch(Math.min(batch,steps-issued));await new Promise(resolve=>setTimeout(resolve));}
    updateSnapshot(snapshot);status(`Optimizing covering · ${countFmt.format(done)} of ${countFmt.format(steps)} steps`,`${GROUPS[state.group]} · best ${number(value('covering_optimizer_best_deg'),4)}°`);progress(done/steps);addHistory();await new Promise(resolve=>requestAnimationFrame(resolve));}
  const stopped=state.stop;updateSnapshot(await state.engine.optimizeCovering(0,mode,true,floor));addHistory();if(stopped)status('Optimization stopped','Best sample retained');log(stopped?'Covering optimization stopped':'Covering optimization',`${countFmt.format(value('covering_optimizer_steps')||0)} steps · best ${number(value('covering_optimizer_best_deg'),5)}° · excess ${number(value('covering_excess'),5)} · ${value('covering_optimizer_kicks')||0} kicks · ${countFmt.format(value('covering_optimizer_improved')||0)} of ${countFmt.format(value('covering_optimizer_jobs')||0)} jobs improved in the last step${floor?` · separation floor ${number(floor,3)}°`:''}`);if(state.q.length/4<=10000)await measure();}
// Near-field forces (Quey's delta = 20) above this many orientations under Auto.
const NEAR_FIELD_FROM=20000;
const nearField=()=>choice('forces')==='near'||choice('forces')==='auto'&&state.q.length/4>NEAR_FIELD_FROM;
$('relax-button').onclick=()=>withBusy(choice('objective')==='covering'?'Optimizing covering':'Relaxing sample',async()=>{if(choice('objective')==='covering')return optimizeCovering();const iterations=+$('iterations').value;if(!Number.isInteger(iterations)||iterations<1||iterations>1000)throw new Error('Choose 1–1,000 iterations.');await state.engine.relaxNeighbourhood(nearField()?20:0);updateSnapshot(await state.engine.identity({add:$('add-identity').checked,fixed:$('fix-identity').checked}));state.stop=false;progress(0);
  // Steps run in engine batches of about 100 ms of GPU time, one readback each. The next batch
  // is queued before this one is drawn, so the GPU computes while the page draws. Batches are
  // sized by the engine's GPU wait time: a call's wall time also counts the page's drawing.
  let done=0,issued=0,batch=1,relaxing=0,pending=null;
  const launch=count=>{issued+=count;const waited=state.engine.state.stats.waitMs;return state.engine.relax(count).then(snapshot=>({snapshot,count,elapsed:state.engine.state.stats.waitMs-waited}));};
  pending=launch(Math.min(batch,iterations));
  while(pending){const {snapshot,count,elapsed}=await pending;pending=null;relaxing+=elapsed;done+=count;state.stepMilliseconds=.8*state.stepMilliseconds+.2*elapsed/count;batch=Math.max(1,Math.min(100,Math.round(count*100/Math.max(1,elapsed))));
    // The engine queue starts the call in a later task: yield once so it is submitted before drawing.
    if(issued<iterations&&!state.stop){pending=launch(Math.min(batch,iterations-issued));await new Promise(resolve=>setTimeout(resolve));}
    status(`Relaxing · ${countFmt.format(done)} of ${countFmt.format(iterations)}`,`${GROUPS[state.group]} · ${value('relax_neighbourhood')?'near-field':'all-pairs'} repulsion`);updateSnapshot(snapshot);progress(done/iterations);addHistory();await maybeCovering('step');await new Promise(resolve=>requestAnimationFrame(resolve));}
  if(state.stop)status('Relaxation stopped','Current sample retained');log(state.stop?'Relaxation stopped':'Relaxation',`${countFmt.format(done)} steps · ${value('relax_neighbourhood')?'near field':'all pairs'} · ${(relaxing/Math.max(1,done)).toFixed(2)} ms of GPU time per step · ${value('energy_neighbourhood')?'near-field ':''}energy ${number(value('quey_energy','energy'),7)}`);if(state.q.length/4<=10000)await measure();if(!state.stop)await maybeCovering('end');});
$('stop-button').onclick=()=>{state.stop=true;state.engine?.cancel();status('Stopping after the current GPU step','Current sample will be retained');};

function quaternionRows(q){return Array.from({length:q.length/4},(_,i)=>Array.from(q.subarray(i*4,i*4+4)));}
function download(filename,body,type){const url=URL.createObjectURL(new Blob([body],{type}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('export-button').onclick=()=>$('export-dialog').showModal();
$('export-json').onclick=()=>{const sample={format:'zxtalori-orientations',version:1,convention:'scalar-first, right group action',group:GROUPS[state.group],quaternions:quaternionRows(state.q),provenance:state.provenance,measures:state.stats,history:state.history};download(`zxtalori-${sample.group}-${state.q.length/4}.json`,JSON.stringify(sample,null,2),'application/json');$('export-dialog').close();};
$('export-csv').onclick=()=>{download(`zxtalori-${GROUPS[state.group]}-${state.q.length/4}.csv`,'w,x,y,z\n'+quaternionRows(state.q).map(row=>row.join(',')).join('\n'),'text/csv');$('export-dialog').close();};
$('import-button').onclick=()=>$('import-file').click();
$('import-file').onchange=async event=>{const file=event.target.files[0];if(!file)return;await withBusy('Importing sample',async()=>{if(file.size>128*1024*1024)throw new Error('Choose an orientation file smaller than 128 MB.');const text=await file.text();let rows,meta=null;if(file.name.toLowerCase().endsWith('.json')){meta=JSON.parse(text);rows=Array.isArray(meta)?meta:meta.quaternions;if(meta.group&&GROUPS.includes(meta.group)){setChoice('group',meta.group);updateGroup();}}else{rows=text.trim().split(/\r?\n/).filter(line=>line.trim()&&!/^\s*#/.test(line)).map(line=>line.trim().split(/[,;\s]+/).map(Number));if(rows[0]?.some(x=>!Number.isFinite(x)))rows.shift();}if(!Array.isArray(rows)||!rows.length||rows.some(row=>!Array.isArray(row)||row.length!==4||row.some(v=>typeof v!=='number'||!Number.isFinite(v))))throw new Error('Import rows must contain four finite numbers: w, x, y, z.');if(rows.length>1000000)throw new Error('Imports support up to 1,000,000 orientations.');if(rows.some(row=>Math.hypot(...row)<1e-12))throw new Error('A zero quaternion does not represent an orientation.');const q=Float32Array.from(rows.flat());updateSnapshot(await state.engine.set(groupId(),q),{original:true});state.provenance={...(meta?.provenance||{}),source_file:file.name,group:GROUPS[groupId()],generator:'Imported sample'};if(state.q.length/4<=10000)await measure();await maybeCovering('generation');});event.target.value='';};

// Points are splatted into one pixel buffer through a depth buffer: opaque discs of three
// sizes, shaded by depth in 16 bands mixed toward the canvas color (no sort, no paths), so
// a 100k-point frame takes milliseconds.
const DEPTH_BANDS=16,splat={width:0,height:0,image:null,pixels:null,depth:null,layer:document.createElement('canvas'),stamps:new Map(),colors:null,colorKey:''};
function stamp(radius){const key=Math.max(1,Math.round(radius*4));let offsets=splat.stamps.get(key);if(!offsets){const r=key/4,reach=Math.ceil(r),list=[];for(let dy=-reach;dy<=reach;dy++)for(let dx=-reach;dx<=reach;dx++)if(dx*dx+dy*dy<=r*r)list.push(dx,dy);if(!list.length)list.push(0,0);offsets=Int16Array.from(list);splat.stamps.set(key,offsets);}return offsets;}
// Per-point measures: one-hue sequential ramps validated against each theme's canvas
// (light end first); the strong end marks the worse value.
const RAMPS={light:['#86b6ef','#5598e7','#2a78d6','#1c5cab','#0d366b'],dark:['#184f95','#256abf','#3987e5','#6da7ec','#9ec5f4','#cde2fb']},VALUE_BINS=32;
const MEASURES={separation:{label:'Nearest neighbour',iteration:'point_separation_iteration',low:true,digits:3,unit:'°'},covering:{label:'Largest adjacent cap',iteration:'point_covering_iteration',digits:3,unit:'°'},faces:{label:'Voronoi faces',iteration:'point_covering_iteration',integer:true,unit:''}};
function rampColor(t){const ramp=RAMPS[palette.dark?'dark':'light'],x=Math.max(0,Math.min(1,t))*(ramp.length-1),i=Math.min(ramp.length-2,Math.floor(x)),f=x-i,hex=h=>[1,3,5].map(k=>parseInt(h.slice(k,k+2),16)),a=hex(ramp[i]),b=hex(ramp[i+1]);return a.map((v,k)=>v+(b[k]-v)*f);}
const measureCurrent=key=>!!state.points?.[key]&&value(MEASURES[key].iteration)===(+value('iteration')||0);
// What the points show: Objective picks the measure the current run optimizes (falling back to
// the other current one, then to symmetry zones), never one that needs a pass of its own.
function colorMode(){const pick=$('color-by').value,extended=choice('view-domain')==='so3';
  if(pick==='auto'){for(const key of choice('objective')==='covering'?['covering','separation']:['separation','covering'])if(measureCurrent(key))return key;return extended?'zone':'single';}
  if(pick in MEASURES)return state.points?.[pick]?pick:'single';
  return pick==='zone'&&extended?'zone':'single';}
// The shown range: 1st to 99th percentile (whole range for counts), cached per values array.
function valueScale(key){const values=state.points[key];if(splat.scale?.values!==values){const sorted=values.slice().sort(),at=p=>sorted[Math.min(sorted.length-1,Math.floor(p*(sorted.length-1)))],integer=MEASURES[key].integer,lo=integer?sorted[0]:at(.01),hi=integer?sorted.at(-1):at(.99);splat.scale={values,lo,hi:hi>lo?hi:lo+(integer?1:1e-6)};}return splat.scale;}
// Packed RGBA (little-endian ABGR) per color and depth band; the last band is the generated
// sample's color. Measures keep most of their lightness at depth, so the ramp stays readable.
function splatColors(mode,order){const key=`${mode}/${order}/${palette.pointRGB}/${palette.canvasRGB}/${palette.original}/${palette.dark}`;if(splat.colorKey===key)return splat.colors;const back=palette.canvasRGB.split(',').map(Number),hsl=(h,s,l)=>{const a=s*Math.min(l,1-l),f=n=>{const k=(n+h/30)%12;return 255*(l-a*Math.max(-1,Math.min(k-3,9-k,1)));};return [f(0),f(8),f(4)];},measure=mode in MEASURES,bases=mode==='zone'?Array.from({length:order},(_,op)=>hsl(op*137.508%360,.65,palette.dark?.65:.42)):measure?Array.from({length:VALUE_BINS},(_,b)=>rampColor(b/(VALUE_BINS-1))):[palette.pointRGB.split(',').map(Number)],original=palette.rgb(palette.original).split(',').map(Number),pack=(c,t)=>{const v=c.map((x,i)=>Math.round(back[i]+(x-back[i])*t));return (255<<24|v[2]<<16|v[1]<<8|v[0])>>>0;},near=measure?.75:.3;
  const colors=new Uint32Array((bases.length+1)*DEPTH_BANDS);bases.forEach((c,b)=>{for(let k=0;k<DEPTH_BANDS;k++)colors[b*DEPTH_BANDS+k]=pack(c,near+(1-near)*k/(DEPTH_BANDS-1));});colors.fill(pack(original,.5),bases.length*DEPTH_BANDS);splat.colors=colors;splat.colorKey=key;return colors;}
// The plot legend's first item names what the colors mean.
function updateColorKey(mode){const measure=MEASURES[mode];$('color-swatch').hidden=!!measure;$('color-scale').hidden=!measure;$('color-swatch').classList.toggle('zones',mode==='zone');
  if(!measure){$('color-name').textContent=mode==='zone'?'Current · by zone':'Current';return;}
  const {lo,hi}=valueScale(mode),format=v=>measure.integer?String(Math.round(v)):number(v,measure.digits),stale=!measureCurrent(mode);
  $('color-name').textContent=`${measure.label}${stale?` · step ${value(measure.iteration)}`:''}`;$('color-low').textContent=format(lo)+measure.unit;$('color-high').textContent=format(hi)+measure.unit;
  const stops=Array.from({length:6},(_,k)=>{const c=rampColor(measure.low?1-k/5:k/5);return `rgb(${c.map(Math.round).join(',')}) ${k*20}%`;});$('color-ramp').style.background=`linear-gradient(90deg,${stops.join(',')})`;
  $('color-key').title=`${measure.label}: ${measure.low?'shorter distances':'larger values'} are darker${palette.dark?' (lighter on the dark theme)':''}${measure.integer?'':'; 1st to 99th percentile'}${stale?'; from an earlier step':''}`;}
// Canvas colors come from the --plot-* CSS tokens, so each theme is defined once.
const palette={},probe=document.body.appendChild(Object.assign(document.createElement('i'),{hidden:true}));
function refreshPalette(){const token=name=>{probe.style.color=`var(${name})`;return getComputedStyle(probe).color;},rgb=color=>color.match(/[\d.]+/g).slice(0,3).join(',');for(const key of ['point','original','wire','axis','label','dt','shell','ring','grid','line'])palette[key]=token('--plot-'+key);palette.panel=token('--panel');palette.rgb=rgb;palette.series=METRICS.map((m,i)=>token('--series-'+(i+1)));palette.pointRGB=rgb(palette.point);palette.lineRGB=rgb(palette.line);palette.canvasRGB=rgb(token('--canvas'));palette.dark=palette.canvasRGB.split(',').reduce((a,b)=>a+ +b,0)<384;palette.font=getComputedStyle(document.body).fontFamily;}
refreshPalette();
function retheme(){refreshPalette();view.draw();drawHistory();}
$('theme-button').onclick=()=>{const theme=palette.dark?'light':'dark';document.documentElement.dataset.theme=theme;try{localStorage.setItem('zxtalori-theme',theme);}catch{}retheme();};
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',retheme);

const view={canvas:$('view'),points:null,original:null,originalKey:'',originalSource:null,wire:[],sx:new Float32Array(),sy:new Float32Array(),slot:new Int32Array(),shown:0,rotation:multiply([Math.cos(.18),Math.sin(.18),0,0],[Math.cos(-.275),0,Math.sin(-.275),0]),zoom:1,pan:[0,0],extent:1,geometryKey:'',rebuild(){
  const projection=choice('projection'),extended=choice('view-domain')==='so3',unbounded=projection==='rodrigues'&&(extended||state.group<4),clip=Math.max(1,Math.min(100,+$('rf-limit').value||4));$('clip-field').hidden=!unbounded;
  const key=`${state.group}/${projection}/${extended}/${clip}`;if(this.geometryKey!==key&&state.operations.length){const geometry=boundary(state.operations,state.group,projection,extended,clip);this.wire=geometry.wire;this.extent=geometry.extent;this.geometryKey=key;}
  const limit=+$('display-limit').value,current=displayPoints(state.q,state.operations,projection,extended,limit,unbounded?clip:Infinity);this.total=current.total;this.omitted=current.omitted;this.points=current;if(this.originalKey!==`${key}/${limit}`||this.originalSource!==state.original){this.originalKey=`${key}/${limit}`;this.originalSource=state.original;this.original=state.original?displayPoints(state.original,state.operations,projection,extended,limit,unbounded?clip:Infinity):null;}if(this.sx.length<current.count){this.sx=new Float32Array(current.count);this.sy=new Float32Array(current.count);this.slot=new Int32Array(current.count);}$('display-count').textContent=`${countFmt.format(current.count)} / ${countFmt.format(current.total)}${current.omitted?' · '+countFmt.format(current.omitted)+' clipped':''}`;
},transform(p){const q=this.rotation,v=[q[1],q[2],q[3]],t=[2*(v[1]*p[2]-v[2]*p[1]),2*(v[2]*p[0]-v[0]*p[2]),2*(v[0]*p[1]-v[1]*p[0])];return [p[0]+q[0]*t[0]+v[1]*t[2]-v[2]*t[1],p[1]+q[0]*t[1]+v[2]*t[0]-v[0]*t[2],p[2]+q[0]*t[2]+v[0]*t[1]-v[1]*t[0]];},orbit(dx,dy){const n=Math.hypot(dx,dy),a=n*.004,s=n?Math.sin(a)/n:0;this.rotation=multiply([Math.cos(a),dy*s,dx*s,0],this.rotation);const length=Math.hypot(...this.rotation);this.rotation=this.rotation.map(x=>x/length);},draw(){
  const canvas=this.canvas,rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio,2),W=Math.max(1,Math.round(rect.width*dpr)),H=Math.max(1,Math.round(rect.height*dpr));if(canvas.width!==W||canvas.height!==H){canvas.width=W;canvas.height=H;}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height,scale=Math.min(w,h)*.4*this.zoom/this.extent,cx=w/2+this.pan[0],cy=h/2+this.pan[1];const screen=p=>{const t=this.transform(p);return [cx+t[0]*scale,cy-t[1]*scale,t[2]];};ctx.clearRect(0,0,w,h);
  const cut=$('cut-plane').value,axis='xyz'.indexOf(cut.at(-1)),sign=cut.startsWith('-')?-1:1,visible=p=>axis<0||sign*p[axis]>=0;const line=(points,color,width=.7)=>{ctx.beginPath();for(let i=1;i<points.length;i++){let a=points[i-1],b=points[i];if(!visible(a)&&!visible(b))continue;if(visible(a)!==visible(b)){const t=a[axis]/(a[axis]-b[axis]),v=a.map((x,k)=>x+t*(b[k]-x));if(!visible(a))a=v;else b=v;}const x=screen(a),y=screen(b);ctx.moveTo(x[0],x[1]);ctx.lineTo(y[0],y[1]);}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();};const axes=[[1,0,0],[0,1,0],[0,0,1]];for(let i=0;i<3;i++){const p=axes[i].map(v=>v*this.extent*1.1);line([p.map(v=>-v),p],palette.axis,.75);const [x,y]=screen(p);ctx.fillStyle=palette.label;ctx.font='italic 13px Cambria, Georgia, serif';ctx.fillText(['x','y','z'][i],x+5,y-4);}if($('show-zone').checked)for(const wire of this.wire)line(wire,palette.wire,+$('wire-width').value);
  if($('show-dt').checked&&state.mesh&&state.meshIteration===(+value('iteration')||0)){
    const {vertices,tets}=state.mesh,extended=choice('view-domain')==='so3',order=extended?state.operations.length/4:1,budget=Math.max(100,+$('dt-budget').value||6000),step=Math.max(1,Math.ceil(tets.length/4*6*order/budget));
    const projected=(id,op)=>{const x=vertices.subarray(id*4,id*4+3),r=x.reduce((a,b)=>a+b*b,0);let q=[(1-r)/(1+r),...Array.from(x,v=>2*v/(1+r))];if(op)q=multiply(q,state.operations.subarray(op*4,op*4+4));return project(q,choice('projection'));};
    if($('show-shell').checked&&!extended){ctx.fillStyle=palette.shell;const stride=Math.max(1,Math.ceil((vertices.length/4-state.q.length/4)/Math.max(100,+$('display-limit').value||10000)));for(let id=state.q.length/4;id<vertices.length/4;id+=stride){const p=projected(id,0);if(p&&visible(p)){const s=screen(p);ctx.beginPath();ctx.arc(s[0],s[1],+$('point-size').value,0,Math.PI*2);ctx.fill();}}}
    for(let at=0;at<tets.length/4*order;at+=step){const tet=at%(tets.length/4),op=Math.floor(at/(tets.length/4)),v=Array.from(tets.subarray(tet*4,tet*4+4),id=>projected(id,op));for(let a=0;a<4;a++)for(let b=a+1;b<4;b++)if(v[a]&&v[b]&&Math.hypot(...v[a].map((x,k)=>x-v[b][k]))<this.extent)line([v[a],v[b]],palette.dt,+$('wire-width').value*.7);}
  }
  this.splat(ctx,W,H,dpr,cx,cy,scale,axis,sign);
  if(state.selected>=0&&this.points)for(let k=0;k<this.shown;k++)if(this.points.index[this.slot[k]]===state.selected){ctx.strokeStyle=palette.ring;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(this.sx[k],this.sy[k],6.5,0,2*Math.PI);ctx.stroke();}
},splat(ctx,W,H,dpr,cx,cy,scale,axis,sign){
  // The view rotation as a matrix: nine multiplies per point and no allocation.
  const [w,x,y,z]=this.rotation,m=[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)];
  if(splat.width!==W||splat.height!==H){splat.width=splat.layer.width=W;splat.height=splat.layer.height=H;splat.image=new ImageData(W,H);splat.pixels=new Uint32Array(splat.image.data.buffer);splat.depth=new Float32Array(W*H);}
  const pixels=splat.pixels,zbuffer=splat.depth;pixels.fill(0);zbuffer.fill(-Infinity);
  const mode=colorMode(),zones=mode==='zone',measure=MEASURES[mode],order=zones?Math.max(1,state.operations.length/4):1,colors=splatColors(mode,order),values=measure?state.points[mode]:null,{lo,hi}=measure?valueScale(mode):{lo:0,hi:1},bin=(VALUE_BINS-1)/(hi-lo),size=+$('point-size').value*dpr,stamps=[stamp(size*.85),stamp(size),stamp(size*1.15)],extent=this.extent,s=scale*dpr,ox=cx*dpr,oy=cy*dpr;
  const draw=(set,fixed,picking)=>{const {xyz,operation,index,count}=set;let shown=0;
    for(let i=0;i<count;i++){if(axis>=0&&sign*xyz[3*i+axis]<0)continue;const px=xyz[3*i],py=xyz[3*i+1],pz=xyz[3*i+2];
      const tx=m[0]*px+m[1]*py+m[2]*pz,ty=m[3]*px+m[4]*py+m[5]*pz,tz=m[6]*px+m[7]*py+m[8]*pz,sx=ox+tx*s,sy=oy-ty*s;
      if(picking){this.sx[shown]=sx/dpr;this.sy[shown]=sy/dpr;this.slot[shown]=i;shown++;}
      const depth=Math.max(0,Math.min(1,(.35+tz/(2.6*extent))/.7)),group=values?Math.max(0,Math.min(VALUE_BINS-1,Math.round((values[index[i]]-lo)*bin))):zones?operation[i]:0,color=fixed>=0?colors[fixed]:colors[(values&&measure.low?VALUE_BINS-1-group:group)*DEPTH_BANDS+Math.round(depth*(DEPTH_BANDS-1))],offsets=fixed>=0?stamps[0]:stamps[depth<.33?0:depth<.67?1:2],u0=Math.round(sx),v0=Math.round(sy),key=fixed>=0?-1e30:tz;
      for(let k=0;k<offsets.length;k+=2){const u=u0+offsets[k],v=v0+offsets[k+1];if(u<0||v<0||u>=W||v>=H)continue;const at=v*W+u;if(key>zbuffer[at]){zbuffer[at]=key;pixels[at]=color;}}
    }
    return shown;};
  // The generated sample stays behind every current point.
  if(this.original&&$('show-original').checked)draw(this.original,colors.length-1,false);
  this.shown=this.points?draw(this.points,-1,true):0;
  updateColorKey(mode);
  splat.layer.getContext('2d').putImageData(splat.image,0,0);ctx.save();ctx.setTransform(1,0,0,1,0,0);ctx.drawImage(splat.layer,0,0);ctx.restore();
  $('display-count').textContent=`${countFmt.format(this.shown)} visible / ${countFmt.format(this.total||0)}${this.omitted?' · RF clipped':''}`;
}};
// Measures with a history double as toggles: any number of series can be plotted at once.
for(const [i,metric] of METRICS.entries()){const chip=document.createElement('button');chip.type='button';chip.dataset.metric=metric.key;chip.style.setProperty('--swatch',`var(--series-${i+1})`);chip.append(document.createElement('i'),document.createTextNode(metric.label));$('history-series').append(chip);}
const metricButtons=document.querySelectorAll('[data-metric]');
function syncSeries(){for(const button of metricButtons){const on=state.series.includes(button.dataset.metric);button.classList.toggle('active',on);button.setAttribute('aria-pressed',on);}try{localStorage.setItem('zxtalori-series',JSON.stringify(state.series));}catch{}drawHistory();}
for(const button of metricButtons)button.onclick=()=>{const key=button.dataset.metric,on=state.series.includes(key);if(on&&state.series.length===1)return;state.series=METRICS.map(m=>m.key).filter(k=>k===key?!on:state.series.includes(k));syncSeries();};
for(const id of ['point-size','wire-width','color-by','cut-plane','show-dt','show-shell','dt-budget'])$(id).oninput=()=>view.draw();
for(const id of ['point-size','wire-width'])$(id).addEventListener('input',()=>$(id+'-value').textContent=$(id).value);
$('display-panel').addEventListener('toggle',e=>$('display-button').classList.toggle('on',e.newState==='open'));
addEventListener('keydown',e=>{if(e.key==='Escape'&&$('display-panel').matches(':popover-open'))$('display-panel').hidePopover();});
$('load-dt').onclick=()=>withBusy('Computing Delaunay geometry',async()=>{state.mesh=await state.engine.mesh();state.meshIteration=+value('iteration')||0;$('show-dt').checked=true;$('dt-note').textContent=`${countFmt.format(state.mesh.tets.length/4)} tetrahedra · projected edges · bounded drawing`;updateSnapshot(await state.engine.snapshot());addHistory();});
let rotationTime=0;
function animate(time){const dt=Math.min(.05,(time-rotationTime)/1000);rotationTime=time;if($('auto-rotate').checked&&!document.hidden){const a=dt*.12;view.rotation=multiply(view.rotation,[Math.cos(a/2),0,0,Math.sin(a/2)]);view.draw();}requestAnimationFrame(animate);}requestAnimationFrame(animate);
let drag=null;
view.canvas.oncontextmenu=e=>e.preventDefault();
view.canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,pan:e.shiftKey||e.button===2};view.canvas.setPointerCapture(e.pointerId);};
view.canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(drag.pan){view.pan[0]+=dx;view.pan[1]+=dy;}else view.orbit(dx,dy);drag.x=e.clientX;drag.y=e.clientY;view.draw();};
view.canvas.onpointerup=e=>{if(drag&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<5){const rect=view.canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let best=-1,distance=12;for(let k=0;k<view.shown;k++){const d=Math.hypot(view.sx[k]-x,view.sy[k]-y);if(d<distance){best=view.slot[k];distance=d;}}state.selected=best>=0?view.points.index[best]:-1;state.selectedOperation=best>=0?view.points.operation[best]:0;updateSelection();view.draw();}drag=null;};
view.canvas.onpointercancel=()=>drag=null;
view.canvas.onwheel=e=>{e.preventDefault();view.zoom=Math.max(.1,Math.min(15,view.zoom*Math.exp(-e.deltaY*.001)));view.draw();};
view.canvas.onkeydown=e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','0'].includes(e.key)){e.preventDefault();view.orbit(e.key==='ArrowLeft'?-12:e.key==='ArrowRight'?12:0,e.key==='ArrowUp'?-12:e.key==='ArrowDown'?12:0);if(e.key==='+')view.zoom*=1.1;if(e.key==='-')view.zoom/=1.1;if(e.key==='0')resetView();view.draw();}};
function resetView(){view.rotation=multiply([Math.cos(.18),Math.sin(.18),0,0],[Math.cos(-.275),0,Math.sin(-.275),0]);view.zoom=1;view.pan=[0,0];view.draw();}
$('reset-view').onclick=resetView;for(const id of ['projection','view-domain','rf-limit','display-limit'])$(id).onchange=()=>{view.rebuild();view.draw();updateSelection();};$('show-zone').onchange=()=>view.draw();$('show-original').onchange=()=>view.draw();
$('fix-identity').onchange=()=>{if($('fix-identity').checked)$('add-identity').checked=true;};
function updateSelection(){const selected=state.selected>=0&&state.selected<state.q.length/4;$('selection-values').hidden=!selected;$('selection-hint').hidden=selected;$('selected-badge').hidden=!selected;$('selection-index').textContent=selected?`#${state.selected}`:'none';if(!selected)return;$('selected-badge').textContent=`Selected #${state.selected}`;let q=state.q.subarray(state.selected*4,state.selected*4+4);if(choice('view-domain')==='so3'){q=multiply(q,state.operations.subarray(state.selectedOperation*4,state.selectedOperation*4+4));if(q[0]<0)q=q.map(x=>-x);$('selection-index').textContent=`#${state.selected} / g${state.selectedOperation}`;}['w','x','y','z'].forEach((key,i)=>$('quat-'+key).textContent=q[i].toFixed(7));$('rotation-angle').textContent=`${(2*Math.atan2(Math.hypot(q[1],q[2],q[3]),Math.abs(q[0]))*180/Math.PI).toFixed(3)}°`;}
let historyHover=null;
// Every plotted measure is scaled to its own range, so measures with different units share one
// plot: a lone series keeps real axis values, several read as trends, with values in the header.
function seriesScale(values){const lo0=Math.min(...values),hi0=Math.max(...values),flat=hi0-lo0<1e-10,range=Math.max(1e-6,Math.abs(hi0)*1e-6,hi0-lo0),margin=flat?range/2:range*.06;return {lo:lo0-margin,hi:hi0+margin,span:hi0-lo0+2*margin};}
function drawHistory(){
  const canvas=$('history'),rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio,2);canvas.width=Math.max(1,rect.width*dpr);canvas.height=Math.max(1,rect.height*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,rect.width,rect.height);
  const note=$('history-note');
  const data=state.series.map(key=>{const index=METRICS.findIndex(m=>m.key===key);return {key,label:METRICS[index].label,color:palette.series[index],css:`var(--series-${index+1})`,points:state.history.filter(x=>Number.isFinite(x[key]))};}).filter(d=>d.points.length);
  $('history-empty').hidden=data.length>0;$('history-empty').textContent=state.history.length?'No current measurements':'No measurements yet';
  if(!data.length){note.replaceChildren();return;}
  for(const d of data)Object.assign(d,seriesScale(d.points.map(x=>x[d.key])));
  const single=data.length===1;ctx.font=`11px ${palette.font}`;
  const ticks=single?[0,1,2].map(i=>number(data[0].hi-i*data[0].span/2,Math.min(8,Math.max(3,Math.ceil(-Math.log10(data[0].span))+1)))):[];
  const pad={l:single?Math.max(...ticks.map(t=>ctx.measureText(t).width))+28:16,r:20,t:14,b:26},w=Math.max(1,rect.width-pad.l-pad.r),h=Math.max(1,rect.height-pad.t-pad.b);
  const first=Math.min(...data.map(d=>d.points[0].iteration)),last=Math.max(...data.map(d=>d.points.at(-1).iteration)),px=entry=>pad.l+(entry.iteration-first)/Math.max(1,last-first)*w,py=(d,entry)=>pad.t+(d.hi-entry[d.key])/d.span*h;
  ctx.lineWidth=1;ctx.strokeStyle=palette.grid;ctx.fillStyle=palette.label;ctx.textAlign='right';ctx.textBaseline='middle';for(let i=0;i<3;i++){const y=Math.round(pad.t+i*h/2)+.5;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+w,y);ctx.stroke();if(single)ctx.fillText(ticks[i],pad.l-14,y);}
  ctx.textBaseline='alphabetic';ctx.textAlign='left';ctx.fillText(`step ${first}`,pad.l,rect.height-8);if(last>first){ctx.textAlign='right';ctx.fillText(String(last),pad.l+w,rect.height-8);}
  const trace=d=>{ctx.beginPath();d.points.forEach((entry,i)=>i?ctx.lineTo(px(entry),py(d,entry)):ctx.moveTo(px(entry),py(d,entry)));};
  for(const d of data){
    if(single&&d.points.length>1){const fill=ctx.createLinearGradient(0,pad.t,0,pad.t+h);fill.addColorStop(0,`rgba(${palette.rgb(d.color)},.18)`);fill.addColorStop(1,`rgba(${palette.rgb(d.color)},0)`);trace(d);ctx.lineTo(px(d.points.at(-1)),pad.t+h);ctx.lineTo(px(d.points[0]),pad.t+h);ctx.closePath();ctx.fillStyle=fill;ctx.fill();}
    if(d.points.length>1){trace(d);ctx.strokeStyle=d.color;ctx.lineWidth=1.75;ctx.lineJoin='round';ctx.stroke();}
  }
  let at=null;if(historyHover!==null){const target=first+(historyHover-pad.l)/w*Math.max(1,last-first);at=state.history.reduce((best,entry)=>Math.abs(entry.iteration-target)<Math.abs(best.iteration-target)?entry:best).iteration;const x=Math.round(pad.l+(at-first)/Math.max(1,last-first)*w)+.5;ctx.strokeStyle=palette.axis;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,pad.t+h);ctx.stroke();}
  const readout=[];
  for(const d of data){const shown=at===null?d.points.at(-1):d.points.reduce((best,entry)=>Math.abs(entry.iteration-at)<Math.abs(best.iteration-at)?entry:best);
    ctx.beginPath();ctx.arc(px(shown),py(d,shown),3.5,0,2*Math.PI);ctx.fillStyle=d.color;ctx.fill();ctx.lineWidth=2;ctx.strokeStyle=palette.panel;ctx.stroke();
    readout.push({d,shown});}
  const stepLabel=document.createElement('span');stepLabel.textContent=`step ${at??readout[0].shown.iteration}`;
  note.replaceChildren(stepLabel,...readout.map(({d,shown})=>{const item=document.createElement('span');item.style.setProperty('--swatch',d.css);item.append(document.createElement('i'),document.createTextNode(single?number(shown[d.key],7):`${d.label} ${number(shown[d.key],5)}`));return item;}));
}
$('history').onpointermove=e=>{historyHover=e.clientX-$('history').getBoundingClientRect().left;drawHistory();};
$('history').onpointerleave=()=>{historyHover=null;drawHistory();};
new ResizeObserver(()=>{view.draw();drawHistory();}).observe($('canvas-wrap'));
new ResizeObserver(drawHistory).observe($('history'));
updateGroup();syncSeries();view.draw();
function gpuBadge(stateName,text){$('gpu-badge').dataset.state=stateName;$('gpu-badge').replaceChildren(Object.assign(document.createElement('i'),{className:'dot'}),document.createTextNode(text));$('gpu-badge').title=text;document.body.classList.remove('connecting');}
try{state.engine=await Zxtalori.create({onCompile:({done,total})=>{if(done<total)status(`Compiling GPU kernels · ${done}/${total}`,'First use');else if(total)log('Compiled',`${total} GPU kernel${total===1?'':'s'}`);}});gpuBadge('ready',state.engine.info);log('GPU ready',String(state.engine.info||''));status('GPU ready','Generate a sample to begin');$('empty-copy').textContent='Generate a sample';busy(false);$('generate-button').click();}catch(error){gpuBadge('error','GPU unavailable');$('empty-copy').textContent=error.message;showError(error);busy(false);}
