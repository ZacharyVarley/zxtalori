// Local covering step: disjoint jobs around the worst zone caps, one workgroup each.
// Cap rows come from covering.wgsl cap_rows_kernel: [sources], then per vertex
// (d rho / d left move of that vertex's source, rho). A job frees the sources of
// one offending cap (its "core"), optionally with their one-ring. Its region is
// every source on a cap touching a free source. Jobs whose regions intersect are
// never selected together, so no cap couples two jobs.
const SLOTS:u32=256u;
const FREE:u32=24u;
const VARS:u32=72u;
const ID_MASK:u32=0x3fffffu;
const NO_SLOT:u32=0xffffffffu;
// ring: 0 core tet, 1 plus face neighbours. fixed: held source + 1, or 0.
// lower: shell grid origin (xyz) and cells per unit (w); grid: cells per axis,
// shell sites, inner local rounds.
struct LocalParams{n:u32,rows:u32,jobs:u32,iterations:u32,band:f32,top:f32,grow:f32,shrink:f32,ring:u32,fixed:u32,seed:u32,zero:u32,value:f32,floor:f32,pad2:f32,pad3:f32,lower:vec4<f32>,grid:vec4<u32>}
@group(0) @binding(0) var<uniform> L:LocalParams;
@group(0) @binding(1) var<storage,read> rows:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> core_claim:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> free_claim:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> region_claim:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> lose:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> job_of:array<u32>;
@group(0) @binding(7) var<storage,read_write> job_core:array<u32>;
@group(0) @binding(8) var<storage,read_write> job_free:array<u32>;
@group(0) @binding(9) var<storage,read_write> job_nfree:array<atomic<u32>>;
@group(0) @binding(10) var<storage,read_write> job_rows:array<u32>;
@group(0) @binding(11) var<storage,read_write> job_n:array<atomic<u32>>;
@group(0) @binding(12) var<storage,read_write> job_old:array<atomic<u32>>;
@group(0) @binding(13) var<storage,read_write> job_new:array<atomic<u32>>;
@group(0) @binding(14) var<storage,read_write> delta:array<vec4<f32>>;
@group(0) @binding(15) var<storage,read_write> trust:array<f32>;
@group(0) @binding(16) var<storage,read_write> q:array<vec4<f32>>;
@group(0) @binding(17) var<storage,read_write> q_old:array<vec4<f32>>;
@group(0) @binding(18) var<storage,read_write> counters:array<atomic<u32>>; // jobs, improved, skipped; [7] slot overflow
// Indirect workgroup counts written by job_launch: [jobs,1,1] then [ceil(jobs/256),1,1].
@group(0) @binding(25) var<storage,read_write> launch:array<u32>;
// Jobs actually picked (pick_jobs counts every claim, then keeps the first L.jobs).
fn job_count()->u32{return min(atomicLoad(&counters[0]),L.jobs);}
@group(0) @binding(19) var<storage,read_write> taken:array<u32>; // inside a chosen job's region
@group(0) @binding(20) var<storage,read_write> near:array<u32>;  // shares a cap with a taken source
@group(0) @binding(21) var<storage,read_write> near2:array<u32>; // one more hop, for one-ring jobs

fn sources(r:u32)->vec4<u32>{return bitcast<vec4<u32>>(rows[r*5u]);}
fn radius(r:u32)->f32{return rows[r*5u+1u].w;}
// A row slot with no cap (the live triangulation keeps one row per cell).
fn dead(r:u32)->bool{return !(radius(r)>0.0);}
// Nonzero for candidate caps; larger radius first, ties broken by row id.
fn priority(r:u32)->u32{
    let x=radius(r)/L.top;let lo=1.0-L.band;if(x<lo){return 0u;}
    let level=u32(clamp((x-lo)/max(L.band,1e-6),0.0,1.0)*1023.0);
    return (level<<22u)|((r+1u)&ID_MASK);
}
fn max_core(s:vec4<u32>)->u32{var m=0u;for(var k=0u;k<4u;k++){m=max(m,atomicLoad(&core_claim[s[k]]));}return m;}
fn max_free(s:vec4<u32>)->u32{var m=0u;for(var k=0u;k<4u;k++){m=max(m,atomicLoad(&free_claim[s[k]]));}return m;}

// ── selection: one independent-set round, repeated with chosen regions blocked ──
@compute @workgroup_size(256)
fn mark_near(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let s=sources(r);var hit=false;
    for(var k=0u;k<4u;k++){if(taken[s[k]]!=0u){hit=true;}}if(!hit){return;}
    for(var k=0u;k<4u;k++){near[s[k]]=1u;}
}
@compute @workgroup_size(256)
fn mark_near2(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let s=sources(r);var hit=false;
    for(var k=0u;k<4u;k++){if(near[s[k]]!=0u){hit=true;}}if(!hit){return;}
    for(var k=0u;k<4u;k++){near2[s[k]]=1u;}
}
@compute @workgroup_size(256)
fn claim_core(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let p=priority(r);if(p==0u){return;}let s=sources(r);
    for(var k=0u;k<4u;k++){if(select(near[s[k]],near2[s[k]],L.ring!=0u)!=0u){return;}}
    for(var k=0u;k<4u;k++){atomicMax(&core_claim[s[k]],p);}
}
// Free set by mode: 0 the core tet; 1 the core plus the apex of each cap sharing a
// face with it (>= 3 core vertices).
fn core_hits(s:vec4<u32>,c:u32)->u32{var hits=0u;for(var t=0u;t<4u;t++){if(atomicLoad(&core_claim[s[t]])==c){hits++;}}return hits;}
fn job_hits(s:vec4<u32>,j:u32)->u32{var hits=0u;for(var t=0u;t<4u;t++){if(job_core[s[t]]==j){hits++;}}return hits;}
fn needed()->u32{return 3u;}
@compute @workgroup_size(256)
fn claim_free(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let s=sources(r);
    for(var k=0u;k<4u;k++){let c=atomicLoad(&core_claim[s[k]]);if(c==0u){continue;}
        if(L.ring==0u){atomicMax(&free_claim[s[k]],c);}
        else if(core_hits(s,c)>=needed()){for(var t=0u;t<4u;t++){atomicMax(&free_claim[s[t]],c);}}}
}
@compute @workgroup_size(256)
fn claim_region(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let s=sources(r);let m=max_free(s);if(m==0u){return;}
    for(var k=0u;k<4u;k++){atomicMax(&region_claim[s[k]],m);}
}
// A candidate loses if a higher one claims any of its free vertices, or if its
// region meets a higher candidate's region or a taken source.
@compute @workgroup_size(256)
fn mark_losers(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let s=sources(r);
    for(var k=0u;k<4u;k++){
        let c=atomicLoad(&core_claim[s[k]]);
        if(c!=0u&&L.ring!=0u&&core_hits(s,c)>=needed()){for(var t=0u;t<4u;t++){if(atomicLoad(&free_claim[s[t]])!=c){atomicStore(&lose[(c&ID_MASK)-1u],1u);}}}
        let f=atomicLoad(&free_claim[s[k]]);if(f==0u){continue;}
        for(var t=0u;t<4u;t++){if(taken[s[t]]!=0u||atomicLoad(&region_claim[s[t]])>f){atomicStore(&lose[(f&ID_MASK)-1u],1u);}}
    }
}
@compute @workgroup_size(256)
fn pick_jobs(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let p=priority(r);if(p==0u||atomicLoad(&lose[r])!=0u){return;}
    let s=sources(r);for(var k=0u;k<4u;k++){if(atomicLoad(&core_claim[s[k]])!=p){return;}}
    let j=atomicAdd(&counters[0],1u);if(j>=L.jobs){return;}
    for(var k=0u;k<4u;k++){job_core[s[k]]=j+1u;}
}
@compute @workgroup_size(256)
fn assign_free(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let s=sources(r);
    for(var k=0u;k<4u;k++){let j=job_core[s[k]];if(j==0u){continue;}
        if(L.ring==0u){job_of[s[k]]=j;}
        else if(job_hits(s,j)>=needed()){for(var t=0u;t<4u;t++){job_of[s[t]]=j;}}}
}
@compute @workgroup_size(256)
fn mark_taken(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let s=sources(r);var hit=false;
    for(var k=0u;k<4u;k++){if(job_of[s[k]]!=0u){hit=true;}}if(!hit){return;}
    for(var k=0u;k<4u;k++){taken[s[k]]=1u;}
}

// ── per-job data ──
@compute @workgroup_size(256)
fn list_free(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=L.n||job_of[s]==0u){return;}let j=job_of[s]-1u;
    let slot=atomicAdd(&job_nfree[j],1u);if(slot<FREE){job_free[j*FREE+slot]=s;}else{atomicStore(&counters[7],1u);}
}
@compute @workgroup_size(256)
fn gather_rows(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let s=sources(r);var j=0u;
    for(var k=0u;k<4u;k++){if(job_of[s[k]]!=0u){j=job_of[s[k]];}}if(j==0u){return;}j-=1u;
    let slot=atomicAdd(&job_n[j],1u);
    if(slot<SLOTS){job_rows[j*SLOTS+slot]=r;}else{atomicStore(&counters[7],1u);}
    atomicMax(&job_old[j],bitcast<u32>(radius(r)));
}

// ── one workgroup per job ──
// Each thread holds one cap: its free slots and gradient stay in registers. Sums of
// per-variable terms over caps use fixed-point workgroup atomics (WGSL has no float
// atomics), scaled from the job's largest gradient so no partial sum can overflow.
struct Cap{slot:vec4<u32>,g:array<vec3<f32>,4>}
var<workgroup> free_ids:array<u32,FREE>;
var<workgroup> bound:array<f32,FREE>;   // per free source: trust radius * rho_max
var<workgroup> acc:array<atomic<i32>,VARS>;
var<workgroup> red:array<f32,SLOTS>;
var<workgroup> d:array<f32,VARS>;
var<workgroup> bar:array<f32,VARS>;
var<workgroup> best:array<f32,VARS>;
var<workgroup> y:array<f32,VARS>;
fn dot_bar(c:Cap)->f32{var s=0.0;for(var v=0u;v<4u;v++){let k=c.slot[v];if(k!=NO_SLOT){s+=dot(c.g[v],vec3<f32>(bar[3u*k],bar[3u*k+1u],bar[3u*k+2u]));}}return s;}
fn dot_d(c:Cap)->f32{var s=0.0;for(var v=0u;v<4u;v++){let k=c.slot[v];if(k!=NO_SLOT){s+=dot(c.g[v],vec3<f32>(d[3u*k],d[3u*k+1u],d[3u*k+2u]));}}return s;}
// y = sum over rows of weight * gradient: each thread adds its cap row and, when it
// owns one, its separation-pair row.
fn accumulate2(i:u32,c:Cap,w:f32,p:Cap,wp:f32,scale:f32,live:bool,paired:bool){
    if(i<VARS){atomicStore(&acc[i],0);}workgroupBarrier();
    if(live){for(var v=0u;v<4u;v++){let k=c.slot[v];if(k!=NO_SLOT){for(var e=0u;e<3u;e++){atomicAdd(&acc[3u*k+e],i32(round(w*c.g[v][e]*scale)));}}}}
    if(paired){for(var v=0u;v<2u;v++){let k=p.slot[v];if(k!=NO_SLOT){for(var e=0u;e<3u;e++){atomicAdd(&acc[3u*k+e],i32(round(wp*p.g[v][e]*scale)));}}}}
    workgroupBarrier();
    if(i<VARS){y[i]=f32(atomicLoad(&acc[i]))/scale;}workgroupBarrier();
}
fn accumulate(i:u32,c:Cap,w:f32,scale:f32,live:bool){var none:Cap;none.slot=vec4<u32>(NO_SLOT);accumulate2(i,c,w,none,0.0,scale,live,false);}
fn reduce1(i:u32,v:f32,take_max:bool)->f32{
    red[i]=v;workgroupBarrier();
    for(var stride=SLOTS/2u;stride>0u;stride>>=1u){if(i<stride){red[i]=select(red[i]+red[i+stride],max(red[i],red[i+stride]),take_max);}workgroupBarrier();}
    let out=red[0];workgroupBarrier();return out;
}

// min over |d_k|_inf <= bound[k] of the largest linearized cap rho + g.d over the
// thread-owned caps, subject to separation pairs theta + h.d >= L.floor when given,
// by Chambolle-Pock with an entropic (softmax) dual for the caps and clamped
// multipliers for the pairs. The best primal iterate by exact penalty is left in best[].
const MULTIPLIER:f32=8.0;
fn solve_minimax(i:u32,cap:Cap,rho:f32,live:bool,count:u32,pair:Cap,theta:f32,paired:bool,pairs:u32){
    if(i<VARS){d[i]=0.0;best[i]=0.0;bar[i]=sin(f32(i)*12.9898+1.0)*select(0.0,1.0,bound[i/3u]>0.0);}workgroupBarrier();
    var own=0.0;for(var k=0u;k<FREE;k++){own=max(own,bound[k]);}own=max(own,1e-12);
    var largest=0.0;
    if(live){for(var v=0u;v<4u;v++){if(cap.slot[v]!=NO_SLOT){let a=abs(cap.g[v]);largest=max(largest,max(a.x,max(a.y,a.z)));}}}
    if(paired){for(var v=0u;v<2u;v++){if(pair.slot[v]!=NO_SLOT){let a=abs(pair.g[v]);largest=max(largest,max(a.x,max(a.y,a.z)));}}}
    let gmax=max(reduce1(i,largest,true),1e-20);
    let power_scale=1073741824.0/(f32(max(count+pairs,1u))*3.5*gmax*gmax);
    let dual_scale=1073741824.0/(gmax*(1.0+MULTIPLIER*f32(pairs)));
    // ||K||_2 of the stacked cap and pair rows by power iteration on K^T K.
    var norm=1e-12;
    for(var it=0u;it<20u;it++){
        accumulate2(i,cap,select(0.0,dot_bar(cap),live),pair,select(0.0,dot_bar(pair),paired),power_scale,live,paired);
        var length=0.0;for(var k=0u;k<VARS;k++){length+=y[k]*y[k];}length=sqrt(length);
        if(length>0.0){norm=sqrt(length);}
        if(i<VARS){bar[i]=select(0.0,y[i]/length,length>0.0);}workgroupBarrier();
    }
    if(i<VARS){bar[i]=0.0;}workgroupBarrier();
    let tau=0.9*own/norm;let sigma=0.9/(own*norm);
    var logit=0.0;var lambda=0.0;
    var best_value=reduce1(i,select(-1e30,rho,live),true)+MULTIPLIER*reduce1(i,select(0.0,max(0.0,L.floor-theta),paired),true);
    for(var it=0u;it<L.iterations;it++){
        logit+=sigma*(rho+dot_bar(cap));
        if(paired){lambda=clamp(lambda+sigma*(L.floor-theta-dot_bar(pair)),0.0,MULTIPLIER);}
        let top=reduce1(i,select(-1e30,logit,live),true);
        let e=select(0.0,exp(logit-top),live);let weights=reduce1(i,e,false);
        accumulate2(i,cap,e/max(weights,1e-30),pair,-lambda,dual_scale,live,paired);
        if(i<VARS){let b=bound[i/3u];let next=clamp(d[i]-tau*y[i],-b,b);bar[i]=2.0*next-d[i];d[i]=next;}workgroupBarrier();
        let value=reduce1(i,select(-1e30,rho+dot_d(cap),live),true)+MULTIPLIER*reduce1(i,select(0.0,max(0.0,L.floor-theta-dot_d(pair)),paired),true);
        if(value<best_value){best_value=value;if(i<VARS){best[i]=d[i];}}workgroupBarrier();
    }
}

// A job whose caps overflowed its slots is left unmoved.
@compute @workgroup_size(256)
fn solve_jobs(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) i:u32){
    let j=wid.x;let nfree=min(atomicLoad(&job_nfree[j]),FREE);let total=atomicLoad(&job_n[j]);
    let complete=total<=SLOTS;let count=min(total,SLOTS);let live=complete&&i<count;
    if(i<FREE){free_ids[i]=select(NO_SLOT,job_free[j*FREE+i],i<nfree);}workgroupBarrier();
    // Each free source moves within its own trust radius; a held identity has radius 0.
    if(i<FREE){bound[i]=select(0.0,trust[free_ids[i]]*L.top,i<nfree);}workgroupBarrier();
    var cap:Cap;cap.slot=vec4<u32>(NO_SLOT);var rho=-1e30;
    if(live){let r=job_rows[j*SLOTS+i];let s=sources(r);rho=radius(r);
        for(var v=0u;v<4u;v++){for(var k=0u;k<FREE;k++){if(k<nfree&&free_ids[k]==s[v]){cap.slot[v]=k;}}
            if(cap.slot[v]!=NO_SLOT){cap.g[v]=rows[r*5u+1u+v].xyz;}}}
    var none:Cap;none.slot=vec4<u32>(NO_SLOT);solve_minimax(i,cap,rho,live,count,none,0.0,false,0u);
    if(i<nfree&&complete){delta[free_ids[i]]=vec4<f32>(best[3u*i],best[3u*i+1u],best[3u*i+2u],0.0);}
}

// ── local re-tessellation: one workgroup per job ──
// The job gathers every orbit point within 2 R + 2 D of its free sources from the
// covering shell's grid (R: its largest cap, D: the displacement budget), then
// repeats: Delaunay stars of the free sources, the minimax LP over their caps, the
// move, and new stars that keep the move only if the largest cap dropped. A cap of
// radius <= R through a free point lies within 2 R of it, so every such cap found is
// empty and none is missed. A star is complete when its tets' solid angles at the
// free point sum to 4 pi; a larger cap leaves a deficit, so the move is rejected.
//
// Points are defined as the exact covering pass defines them: each site's f32
// stereographic coordinate, inverted in double-single. A star is computed in the
// stereographic chart of its centre P (P at the origin), where a cap through P is a
// sphere through the origin of radius r and angular radius atan(2 r).
const LOCAL:u32=192u;
const NEAR:u32=32u;
const CAPS:u32=256u;
const HASH:u32=512u;
const SPHERE:f32=12.566370614359172;
const ANGLE_SCALE:f32=16777216.0;
@group(0) @binding(23) var<storage,read> shell_index:array<u32>;
@group(0) @binding(24) var<uniform> table:array<vec4<f32>,60>;
var<workgroup> lp:array<vec4<f32>,LOCAL>;    // local points (f32), for neighbour filters and gradients
var<workgroup> l_si:array<u32,LOCAL>;      // source | (2 * operator + negative) << 25
var<workgroup> l_slot:array<u32,LOCAL>;      // free slot of the point's source, or NO_SLOT
var<workgroup> seen:array<atomic<u32>,HASH>; // gathered site ids
var<workgroup> canon:array<u32,FREE>;        // local id of each free source's own site
var<workgroup> rot:array<vec4<f32>,FREE>;    // accumulated left rotation per free source
var<workgroup> core_now:array<vec4<f32>,FREE>;   // f32 cores as the apply kernel will write them
var<workgroup> core_trial:array<vec4<f32>,FREE>;
var<workgroup> moved:array<f32,FREE>;        // accumulated displacement bound
var<workgroup> mult:array<f32,FREE>;         // local trust multiplier
var<workgroup> star_ids:array<u32,NEAR>;
var<workgroup> star_y:array<vec4<f32>,NEAR>; // neighbours in the centre's stereographic chart
var<workgroup> frame:array<vec2<f32>,4>;     // centre quaternion, double-single
var<workgroup> keys_cur:array<u32,CAPS>;
var<workgroup> rad_cur:array<f32,CAPS>;
var<workgroup> keys_new:array<u32,CAPS>;
var<workgroup> rad_new:array<f32,CAPS>;
var<workgroup> angle_sum:array<atomic<i32>,FREE>;
// Each job's scalar state, packed: Metal passes every workgroup variable as its own
// threadgroup argument and allows at most 31 per kernel.
struct JobTally{l_n:atomic<u32>,star_n:atomic<u32>,caps_n:atomic<u32>,fail:atomic<u32>,pair_n:atomic<u32>}
struct JobState{
    verdict:u32,count_cur:u32,result_max:f32,free_n:u32,
    // Why a tessellation failed: 2 local points, 4 star neighbours, 8 caps (or
    // separation pairs) overflowed; 16 another job's free source nearby; 32 beyond
    // the shell; 64 no own site; 128 a star did not close.
    reason:u32,
    pair_ok:u32,
}
var<workgroup> tally:JobTally;
var<workgroup> job:JobState;

// Double-single arithmetic as in dt_double.wgsl; the runtime zero blocks reassociation.
fn rounded(x:f32)->f32{return bitcast<f32>(bitcast<u32>(x)^L.zero);}
fn ds_add(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
    let s=rounded(a.x+b.x);let v=rounded(s-a.x);let e=rounded(rounded(rounded(a.x-rounded(s-v))+rounded(b.x-v))+rounded(a.y+b.y));let h=rounded(s+e);return vec2<f32>(h,rounded(e-rounded(h-s)));
}
fn ds_sub(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{return ds_add(a,-b);}
fn ds_mul(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
    let h=rounded(a.x*b.x);let ah=bitcast<f32>(bitcast<u32>(a.x)&0xfffff000u);let bh=bitcast<f32>(bitcast<u32>(b.x)&0xfffff000u);let al=rounded(a.x-ah);let bl=rounded(b.x-bh);
    let err=rounded(rounded(rounded(rounded(ah*bh-h)+rounded(ah*bl))+rounded(al*bh))+rounded(al*bl));
    let low=rounded(err+rounded(rounded(a.x*b.y)+rounded(a.y*b.x)));let high=rounded(h+low);return vec2<f32>(high,rounded(low-rounded(high-h)));
}
fn ds_divide(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{let q=rounded(a.x/b.x);let error=ds_sub(a,ds_mul(b,vec2<f32>(q,0.0)));return ds_add(vec2<f32>(q,0.0),vec2<f32>(rounded(rounded(error.x+error.y)/b.x),0.0));}

fn det3(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>)->f32{return dot(a,cross(b,c));}
// Generalized cross product: orthogonal to a, b, c in R4.
fn cross4(a:vec4<f32>,b:vec4<f32>,c:vec4<f32>)->vec4<f32>{
    return vec4<f32>(det3(a.yzw,b.yzw,c.yzw),-det3(a.xzw,b.xzw,c.xzw),det3(a.xyw,b.xyw,c.xyw),-det3(a.xyz,b.xyz,c.xyz));
}
fn shell_point(site:u32,cells:u32)->vec2<u32>{let o=shell_index[cells+L.grid.y+site];return vec2<u32>(o%L.n,o/L.n);}
fn image_of(core:vec4<f32>,image:u32)->vec4<f32>{
    var v=normalize(qmul(normalize(core),normalize(table[image/2u])));if((image&1u)!=0u){v=-v;}return v;
}
// The f32 stereographic coordinate the shell build gives this image.
fn site_coordinate(core:vec4<f32>,image:u32)->vec3<f32>{let v=image_of(core,image);return v.yzw/(1.0+v.x);}
fn ds_quaternion(x:vec3<f32>)->array<vec2<f32>,4>{
    var r=vec2<f32>(0.0);for(var k=0u;k<3u;k++){r=ds_add(r,ds_mul(vec2<f32>(x[k],0.0),vec2<f32>(x[k],0.0)));}
    let den=ds_add(vec2<f32>(1.0,0.0),r);var v:array<vec2<f32>,4>;v[0]=ds_divide(ds_sub(vec2<f32>(1.0,0.0),r),den);
    for(var k=0u;k<3u;k++){v[k+1u]=ds_divide(vec2<f32>(2.0*x[k],0.0),den);}return v;
}
// conj(frame) * v in double-single, then the stereographic chart around the frame.
fn chart(v:array<vec2<f32>,4>)->vec3<f32>{
    let f=frame;var s=vec2<f32>(0.0);for(var k=0u;k<4u;k++){s=ds_add(s,ds_mul(f[k],v[k]));}
    var w:array<vec2<f32>,3>;
    w[0]=ds_sub(ds_sub(ds_mul(f[0],v[1]),ds_mul(v[0],f[1])),ds_sub(ds_mul(f[2],v[3]),ds_mul(f[3],v[2])));
    w[1]=ds_sub(ds_sub(ds_mul(f[0],v[2]),ds_mul(v[0],f[2])),ds_sub(ds_mul(f[3],v[1]),ds_mul(f[1],v[3])));
    w[2]=ds_sub(ds_sub(ds_mul(f[0],v[3]),ds_mul(v[0],f[3])),ds_sub(ds_mul(f[1],v[2]),ds_mul(f[2],v[1])));
    let den=1.0+(s.x+s.y);return vec3<f32>(w[0].x+w[0].y,w[1].x+w[1].y,w[2].x+w[2].y)/den;
}
fn core_of(id:u32,trial:bool)->vec4<f32>{let slot=l_slot[id];if(slot==NO_SLOT){return q[l_si[id]&0x1ffffffu];}if(trial){return core_trial[slot];}return core_now[slot];}
fn exp_move(v:vec3<f32>)->vec4<f32>{let angle=length(v);return vec4<f32>(cos(angle),select(1.0,sin(angle)/angle,angle>0.0)*v);}
fn trial_rotation(slot:u32)->vec4<f32>{return normalize(qmul(exp_move(vec3<f32>(best[3u*slot],best[3u*slot+1u],best[3u*slot+2u])),rot[slot]));}
// f32 position for neighbour filters: at the trial move of its free source, or where it is.
fn position(id:u32,trial:bool)->vec4<f32>{
    let slot=l_slot[id];if(!trial||slot==NO_SLOT){return lp[id];}
    return normalize(qmul(exp_move(vec3<f32>(best[3u*slot],best[3u*slot+1u],best[3u*slot+2u])),lp[id]));
}
fn solid_angle(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>)->f32{
    let la=length(a);let lb=length(b);let lc=length(c);
    return 2.0*atan2(abs(det3(a,b,c)),la*lb*lc+dot(a,b)*lc+dot(a,c)*lb+dot(b,c)*la);
}
fn sort4(a:u32,b:u32,c:u32,e:u32)->u32{
    var x=array<u32,4>(a,b,c,e);
    for(var s=0u;s<3u;s++){for(var t=0u;t<3u-s;t++){if(x[t]>x[t+1u]){let h=x[t];x[t]=x[t+1u];x[t+1u]=h;}}}
    return x[0]|(x[1]<<8u)|(x[2]<<16u)|(x[3]<<24u);
}
// Stars of every free source at the current (trial=false) or trial positions,
// keeping caps no larger than `limit`. Leaves the caps in keys/rad_new (or _cur),
// their count in tally.caps_n, the largest radius in job.result_max, and job.verdict = 1 when
// every star is complete and nothing overflowed.
fn tessellate(i:u32,trial:bool,limit:f32){
    if(i<FREE){atomicStore(&angle_sum[i],0);}
    if(i==0u){atomicStore(&tally.caps_n,0u);}
    workgroupBarrier();
    let nfree=workgroupUniformLoad(&job.free_n);
    let points=min(atomicLoad(&tally.l_n),LOCAL);let reach=2.0*sin(min(limit,0.78));let reach2=reach*reach*(1.0+1e-3);
    for(var k=0u;k<nfree;k++){
        let centre=canon[k];
        if(i==0u){atomicStore(&tally.star_n,0u);let f=ds_quaternion(site_coordinate(core_of(centre,trial),(l_si[centre]>>25u)));frame=f;}
        workgroupBarrier();
        let pc=position(centre,trial);
        for(var t=i;t<points;t+=256u){if(t==centre){continue;}let x=position(t,trial)-pc;
            if(dot(x,x)<=reach2){let slot=atomicAdd(&tally.star_n,1u);
                if(slot<NEAR){star_ids[slot]=t;star_y[slot]=vec4<f32>(chart(ds_quaternion(site_coordinate(core_of(t,trial),(l_si[t]>>25u)))),0.0);}else{atomicOr(&tally.fail,4u);}}}
        workgroupBarrier();
        let m=min(atomicLoad(&tally.star_n),NEAR);
        for(var t=i;t<NEAR*NEAR;t+=256u){let a=t/NEAR;let b=t%NEAR;
            if(a<b&&b<m){let ya=star_y[a].xyz;let yb=star_y[b].xyz;
                for(var c=b+1u;c<m;c++){let yc=star_y[c].xyz;
                    // Sphere through the origin and ya, yb, yc: 2 x.y = |y|^2 for each.
                    let det=det3(ya,yb,yc);if(!(abs(det)>1e-30)){continue;}
                    let h=0.5*vec3<f32>(dot(ya,ya),dot(yb,yb),dot(yc,yc));
                    let centre3=(h.x*cross(yb,yc)+h.y*cross(yc,ya)+h.z*cross(ya,yb))/det;
                    let r2=dot(centre3,centre3);let rho=atan(2.0*sqrt(r2));if(rho>limit){continue;}
                    var empty=true;
                    for(var e=0u;e<m;e++){if(e==a||e==b||e==c){continue;}let d=star_y[e].xyz-centre3;if(dot(d,d)<r2*(1.0-1e-6)){empty=false;break;}}
                    if(!empty){continue;}
                    atomicAdd(&angle_sum[k],i32(solid_angle(ya,yb,yc)*ANGLE_SCALE));
                    let at=atomicAdd(&tally.caps_n,1u);
                    if(at<CAPS){let key=sort4(centre,star_ids[a],star_ids[b],star_ids[c]);if(trial){keys_new[at]=key;rad_new[at]=rho;}else{keys_cur[at]=key;rad_cur[at]=rho;}}else{atomicOr(&tally.fail,8u);}}}}
        workgroupBarrier();
    }
    // One copy of each cap (a cap with several free vertices is found once per star).
    let count=min(atomicLoad(&tally.caps_n),CAPS);var mine=-1.0;
    if(i<count){let key=select(keys_cur[i],keys_new[i],trial);var first=true;
        for(var s=0u;s<i;s++){if(select(keys_cur[s],keys_new[s],trial)==key){first=false;break;}}
        if(first){mine=select(rad_cur[i],rad_new[i],trial);}}
    workgroupBarrier();
    if(i<count&&mine<0.0){if(trial){rad_new[i]=-1.0;}else{rad_cur[i]=-1.0;}}
    let largest=reduce1(i,mine,true);
    if(i==0u){var why=atomicLoad(&tally.fail);
        for(var k=0u;k<nfree;k++){if(f32(atomicLoad(&angle_sum[k]))<SPHERE*ANGLE_SCALE*(1.0-1e-3)){why|=128u;}}
        job.reason=why;job.verdict=select(0u,1u,why==0u);job.result_max=largest;}
    workgroupBarrier();
}
// The cap thread i owns at current positions, linearized in the moves of its free sources.
fn local_cap(i:u32,count:u32)->Cap{
    var cap:Cap;cap.slot=vec4<u32>(NO_SLOT);
    if(i>=count||rad_cur[i]<0.0){return cap;}
    let key=keys_cur[i];let ids=vec4<u32>(key&255u,(key>>8u)&255u,(key>>16u)&255u,key>>24u);
    let v0=lp[ids.x];let v1=lp[ids.y];let v2=lp[ids.z];let v3=lp[ids.w];
    var c=cross4(v1-v0,v2-v0,v3-v0);let length=length(c);if(!(length>1e-30)){return cap;}c=c/length;if(dot(c,v0)<0.0){c=-c;}
    let rho=rad_cur[i];
    let m=mat4x4<f32>(v0,v1,v2,v3);let det=determinant(m);if(!(abs(det)>1e-30)){return cap;}
    let cot=cos(rho)/sin(rho);
    for(var v=0u;v<4u;v++){let slot=l_slot[ids[v]];if(slot==NO_SLOT){continue;}
        var mv=m;mv[v]=c;let lambda=determinant(mv)/det;let w=m[v];
        let a=vec3<f32>(dot(c,qmul(vec4<f32>(0.0,1.0,0.0,0.0),w)),dot(c,qmul(vec4<f32>(0.0,0.0,1.0,0.0),w)),dot(c,qmul(vec4<f32>(0.0,0.0,0.0,1.0),w)));
        cap.slot[v]=slot;cap.g[v]=-cot*lambda*a;}
    return cap;
}

// Separation pairs (when L.floor > 0): each free source's own site with every local
// point nearer than `reach`, other sources only, since images of one source keep
// their distances under its left move. Stored in seen[], which the gather is done with.
const PAIRS:u32=256u;
const SOURCE_MASK:u32=0x1ffffffu;
fn find_pairs(i:u32,nfree:u32,points:u32,reach:f32){
    if(i==0u){atomicStore(&tally.pair_n,0u);}workgroupBarrier();
    let chord=2.0*sin(0.5*min(reach,3.0));let limit=chord*chord;
    for(var t=i;t<nfree*LOCAL;t+=256u){let k=t/LOCAL;let v=t%LOCAL;let u=canon[k];
        if(v>=points||v==u||(l_si[v]&SOURCE_MASK)==(l_si[u]&SOURCE_MASK)){continue;}
        let vs=l_slot[v];if(vs!=NO_SLOT&&canon[vs]==v&&vs<k){continue;}
        let x=lp[u]-lp[v];if(dot(x,x)<limit){let at=atomicAdd(&tally.pair_n,1u);if(at<PAIRS){atomicStore(&seen[at],u|(v<<8u));}else{atomicOr(&tally.fail,8u);}}}
    workgroupBarrier();
}
// The pair thread i owns: its S3 angle at current or trial positions and, from
// those positions, its linearization in the left moves of its free sources.
fn pair_row(i:u32,count:u32,trial:bool,angle:ptr<function,f32>)->Cap{
    var row:Cap;row.slot=vec4<u32>(NO_SLOT);*angle=1e30;
    if(i>=count){return row;}
    let packed=atomicLoad(&seen[i]);let u=packed&255u;let v=packed>>8u;
    let a=position(u,trial);let b=position(v,trial);let theta=2.0*asin(min(1.0,0.5*length(a-b)));*angle=theta;
    let k=-1.0/max(sin(theta),1e-12);
    let su=l_slot[u];if(su!=NO_SLOT){row.slot[0]=su;row.g[0]=k*vec3<f32>(dot(b,qmul(vec4<f32>(0.0,1.0,0.0,0.0),a)),dot(b,qmul(vec4<f32>(0.0,0.0,1.0,0.0),a)),dot(b,qmul(vec4<f32>(0.0,0.0,0.0,1.0),a)));}
    let sv=l_slot[v];if(sv!=NO_SLOT){row.slot[1]=sv;row.g[1]=k*vec3<f32>(dot(a,qmul(vec4<f32>(0.0,1.0,0.0,0.0),b)),dot(a,qmul(vec4<f32>(0.0,0.0,1.0,0.0),b)),dot(a,qmul(vec4<f32>(0.0,0.0,0.0,1.0),b)));}
    return row;
}

@compute @workgroup_size(256)
fn retessellate_jobs(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) i:u32){
    let j=wid.x;let nfree=min(atomicLoad(&job_nfree[j]),FREE);
    let cells=L.grid.x*L.grid.x*L.grid.x;
    if(i<FREE){free_ids[i]=select(NO_SLOT,job_free[j*FREE+i],i<nfree);rot[i]=vec4<f32>(1.0,0.0,0.0,0.0);moved[i]=0.0;mult[i]=1.0;canon[i]=NO_SLOT;
        if(i<nfree){core_now[i]=normalize(qmul(rot[i],q[free_ids[i]]));}}
    if(i==0u){atomicStore(&tally.l_n,0u);atomicStore(&tally.fail,select(0u,2u,atomicLoad(&job_nfree[j])>FREE));job.free_n=nfree;}
    for(var t=i;t<HASH;t+=256u){atomicStore(&seen[t],NO_SLOT);}
    workgroupBarrier();
    let r0=bitcast<f32>(atomicLoad(&job_old[j]));let limit0=r0*(1.0+1e-3);let budget=0.1*r0;
    let gather=2.0*limit0+2.0*budget+1e-5;let gather_chord=2.0*sin(0.5*min(gather,3.0));
    // The shell holds every orbit point within L.value of the zone; a wider gather would miss some.
    if(i==0u&&gather>L.value){atomicOr(&tally.fail,32u);}
    // Gather orbit points near each free source through the shell grid.
    for(var k=0u;k<nfree;k++){
        let p=normalize(q[free_ids[k]]);let x=p.yzw/(1.0+p.x);
        let h=1.01*gather_chord/max(1e-3,1.0+p.x-gather_chord);
        let lo=vec3<i32>(clamp(floor((x-vec3<f32>(h)-L.lower.xyz)*L.lower.w),vec3<f32>(0.0),vec3<f32>(f32(L.grid.x-1u))));
        let hi=vec3<i32>(clamp(floor((x+vec3<f32>(h)-L.lower.xyz)*L.lower.w),vec3<f32>(0.0),vec3<f32>(f32(L.grid.x-1u))));
        let extent=vec3<u32>(hi-lo)+vec3<u32>(1u);
        for(var t=i;t<extent.x*extent.y*extent.z;t+=256u){
            let cell=vec3<u32>(lo)+vec3<u32>(t%extent.x,(t/extent.x)%extent.y,t/(extent.x*extent.y));
            var site=shell_index[cell.x+L.grid.x*(cell.y+L.grid.x*cell.z)];
            while(site!=NO_SLOT){let o=shell_point(site,cells);let v=image_of(q[o.x],o.y);let dv=v-p;
                if(dot(dv,dv)<=gather_chord*gather_chord){
                    let other=job_of[o.x];if(other!=0u&&other!=j+1u){atomicOr(&tally.fail,16u);}
                    // The first sight of a site claims its local id (open addressing).
                    var h=pcg(site)&(HASH-1u);
                    for(var probe=0u;probe<2u*HASH;probe++){
                        let r=atomicCompareExchangeWeak(&seen[h],NO_SLOT,site);
                        if(r.exchanged){let id=atomicAdd(&tally.l_n,1u);
                            if(id<LOCAL){lp[id]=v;l_si[id]=o.x|(o.y<<25u);var slot=NO_SLOT;
                                for(var f=0u;f<nfree;f++){if(free_ids[f]==o.x){slot=f;if(o.y==0u){canon[f]=id;}}}
                                l_slot[id]=slot;}else{atomicOr(&tally.fail,2u);}
                            break;}
                        if(r.old_value==site){break;}
                        if(r.old_value!=NO_SLOT){h=(h+1u)&(HASH-1u);}
                    }}
                site=shell_index[cells+site];}
        }
    }
    workgroupBarrier();
    workgroupBarrier();
    if(i==0u){for(var k=0u;k<nfree;k++){if(canon[k]==NO_SLOT){atomicOr(&tally.fail,64u);}}}
    workgroupBarrier();
    // Separation pairs a move within this step's budget could bring below the floor.
    let floor_on=L.floor>0.0;
    if(floor_on){find_pairs(i,nfree,min(atomicLoad(&tally.l_n),LOCAL),L.floor+2.0*budget+1e-4);}
    let npairs=select(0u,min(atomicLoad(&tally.pair_n),PAIRS),floor_on);
    // Stars at the start; the job proceeds only if they are complete.
    tessellate(i,false,limit0);
    var current=workgroupUniformLoad(&job.result_max);let start_ok=workgroupUniformLoad(&job.verdict);let start_reason=workgroupUniformLoad(&job.reason);
    if(i==0u){job.count_cur=min(atomicLoad(&tally.caps_n),CAPS);}
    var accepted=0u;let start=current;
    if(start_ok==1u){
        for(var round=0u;round<L.grid.z;round++){
            let count=workgroupUniformLoad(&job.count_cur);
            if(i<FREE){bound[i]=select(0.0,max(0.0,min(trust[free_ids[i]]*L.top*mult[i],budget-moved[i])),i<nfree);}
            workgroupBarrier();
            let cap=local_cap(i,count);let live=cap.slot.x!=NO_SLOT||cap.slot.y!=NO_SLOT||cap.slot.z!=NO_SLOT||cap.slot.w!=NO_SLOT;
            let rho=select(-1e30,rad_cur[min(i,CAPS-1u)],i<count&&rad_cur[min(i,CAPS-1u)]>=0.0);
            var theta=1e30;let pair=pair_row(i,npairs,false,&theta);let paired=i<npairs;
            solve_minimax(i,cap,rho,live,count,pair,theta,paired,npairs);
            let points=min(atomicLoad(&tally.l_n),LOCAL);
            if(i<nfree){core_trial[i]=normalize(qmul(trial_rotation(i),q[free_ids[i]]));}
            if(i==0u){atomicStore(&tally.fail,0u);}
            workgroupBarrier();
            tessellate(i,true,current);
            let trial_max=workgroupUniformLoad(&job.result_max);let complete=workgroupUniformLoad(&job.verdict);
            // A move may not bring a pair below the floor, nor a pair already below it nearer.
            var trial_theta=1e30;let unused=pair_row(i,npairs,true,&trial_theta);
            let nearest=-reduce1(i,select(-1e30,-theta,paired),true);let trial_nearest=-reduce1(i,select(-1e30,-trial_theta,paired),true);
            if(i==0u){job.pair_ok=select(0u,1u,trial_nearest>=min(L.floor,nearest)*(1.0-1e-6));}
            workgroupBarrier();
            let separated=workgroupUniformLoad(&job.pair_ok);
            let keep=complete==1u&&separated==1u&&trial_max<current*(1.0-2e-7);
            if(keep){
                for(var t=i;t<points;t+=256u){lp[t]=position(t,true);}
                if(i<CAPS){keys_cur[i]=keys_new[i];rad_cur[i]=rad_new[i];}
                workgroupBarrier();
                if(i<nfree){let s=vec3<f32>(best[3u*i],best[3u*i+1u],best[3u*i+2u]);rot[i]=trial_rotation(i);core_now[i]=core_trial[i];moved[i]+=length(s);mult[i]=min(mult[i]*L.grow,4.0);}
                if(i==0u){job.count_cur=min(atomicLoad(&tally.caps_n),CAPS);}
                current=trial_max;accepted++;
            }else{if(i<nfree){mult[i]*=L.shrink;}}
            workgroupBarrier();
        }
    }
    // One rotation per free source; trust radii follow the local accept rate.
    if(i<nfree){delta[free_ids[i]]=rot[i];
        if(free_ids[i]+1u!=L.fixed){trust[free_ids[i]]=clamp(trust[free_ids[i]]*select(L.shrink,L.grow,accepted>0u),1e-5,0.3);}}
    if(i==0u){delta[L.n+j]=vec4<f32>(f32(accepted),start,current,f32(start_reason));}
}
@compute @workgroup_size(256)
fn apply_rotations(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=L.n||job_of[s]==0u||s+1u==L.fixed){return;}
    // Only rotations a job wrote; sources beyond its free slots stay where they are, and
    // so do sources whose job kept them (the identity leaves q's bits, and its images, as
    // they are).
    let r=delta[s];if(dot(r,r)>0.5&&any(r!=vec4<f32>(1.0,0.0,0.0,0.0))){q[s]=normalize(qmul(r,q[s]));}
}
// Sizes the per-job dispatches on the GPU, so the job count is never read back mid-step.
@compute @workgroup_size(1)
fn job_launch(){let jobs=job_count();launch[0]=jobs;launch[1]=1u;launch[2]=1u;launch[3]=(jobs+255u)/256u;launch[4]=1u;launch[5]=1u;}
// counters: [1] jobs that improved, [2] jobs skipped, [3] accepted local rounds,
// [8 + b] skipped jobs with job.reason bit b.
@compute @workgroup_size(256)
fn tally_local(@builtin(global_invocation_id) id:vec3<u32>){
    let j=id.x;if(j>=job_count()){return;}let r=delta[L.n+j];
    if(r.x>0.0){atomicAdd(&counters[1],1u);atomicAdd(&counters[3],u32(r.x));}
    if(r.w>0.0){atomicAdd(&counters[2],1u);let why=u32(r.w);for(var b=0u;b<8u;b++){if((why&(1u<<b))!=0u){atomicAdd(&counters[8u+b],1u);}}}
}

// ── apply, re-check, accept ──
fn qmul(a:vec4<f32>,b:vec4<f32>)->vec4<f32>{return vec4<f32>(a.x*b.x-dot(a.yzw,b.yzw),a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));}
fn left_move(s:u32,v:vec3<f32>){
    let angle=length(v);let e=vec4<f32>(cos(angle),select(1.0,sin(angle)/angle,angle>0.0)*v);
    q[s]=normalize(qmul(e,q[s]));
}
@compute @workgroup_size(256)
fn apply_moves(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=L.n||job_of[s]==0u||s+1u==L.fixed){return;}
    left_move(s,delta[s].xyz);
}

// ── basin hopping ──
fn pcg(v:u32)->u32{let state=v*747796405u+2891336453u;let word=((state>>((state>>28u)+4u))^state)*277803737u;return (word>>22u)^word;}
fn uniform01(key:u32)->f32{return (f32(pcg(key)>>8u)+0.5)/16777216.0;}
fn normal(key:u32)->f32{return sqrt(-2.0*log(uniform01(key)))*cos(6.283185307179586*uniform01(key^0x68e31da4u));}
// Isotropic left kick of RMS angle `value` (S3 radians) for every free source.
@compute @workgroup_size(256)
fn kick(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=L.n||s+1u==L.fixed){return;}let key=pcg(s^pcg(L.seed));
    left_move(s,L.value*0.5773502691896258*vec3<f32>(normal(key),normal(key+0x9e3779b9u),normal(key+0x3c6ef372u)));
}
@compute @workgroup_size(256)
fn fill_trust(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=L.n){return;}trust[s]=select(L.value,0.0,s+1u==L.fixed);
}
@compute @workgroup_size(256)
fn new_local(@builtin(global_invocation_id) id:vec3<u32>){
    let r=id.x;if(r>=L.rows||dead(r)){return;}let s=sources(r);var j=0u;
    for(var k=0u;k<4u;k++){if(job_of[s[k]]!=0u){j=job_of[s[k]];}}if(j==0u){return;}
    atomicMax(&job_new[j-1u],bitcast<u32>(radius(r)));
}
// Keep a job only if its exact local maximum dropped; adapt each source's trust radius.
// A held identity keeps radius 0 and never moves.
@compute @workgroup_size(256)
fn accept_jobs(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=L.n||job_of[s]==0u||s+1u==L.fixed){return;}let j=job_of[s]-1u;
    let after=atomicLoad(&job_new[j]);let ok=after!=0u&&after<atomicLoad(&job_old[j]);
    if(ok){trust[s]=min(0.3,trust[s]*L.grow);}else{trust[s]=max(1e-5,trust[s]*L.shrink);q[s]=q_old[s];}
}
@compute @workgroup_size(256)
fn count_improved(@builtin(global_invocation_id) id:vec3<u32>){
    let j=id.x;if(j>=job_count()){return;}let after=atomicLoad(&job_new[j]);
    if(after!=0u&&after<atomicLoad(&job_old[j])){atomicAdd(&counters[1],1u);}
}
// Check variant: each job's local maximum against the exact maximum over the
// caps touching its free sources after a full pass (job_new).
// counters: [4] mismatches above 2e-5 relative, [5] largest relative difference.
@compute @workgroup_size(256)
fn verify_local(@builtin(global_invocation_id) id:vec3<u32>){
    let j=id.x;if(j>=job_count()){return;}let r=delta[L.n+j];if(r.w>0.0){return;}
    let exact=bitcast<f32>(atomicLoad(&job_new[j]));let difference=abs(exact-r.z)/max(r.z,1e-20);
    atomicMax(&counters[5],bitcast<u32>(difference));if(difference>2e-5){atomicAdd(&counters[4],1u);}
}
