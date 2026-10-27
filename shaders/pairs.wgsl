// All-pairs orbit sums. Only the zone representatives and |K| operators are stored.
// A workgroup transforms one shared 64-site tile at a time; antipodes are analytic.
// Quaternions are scalar first and symmetry acts on the right.
const PI: f32 = 3.14159265358979323846;
const FOUR_PI2: f32 = 39.4784176043574345;
// flags: bit0 force-only; bits8..15 first operator; bits16..23 operator count
// (count0 means all remaining operators, retained for direct kernel callers).
// Long sweeps are split into dispatches over operator and site ranges, so that no
// thread runs long enough to trip a GPU watchdog: sites [j_first, j_end) of every
// image are summed (j_end 0 means n, j_first a multiple of 64), and resume != 0
// continues the sums a previous dispatch left in the rows.
struct Params { n: u32, order: u32, flags: u32, pad: u32, j_first: u32, j_end: u32, resume: u32, pad2: u32 }
struct Row { energy: vec4<f32>, extra: vec4<f32>, force: vec4<f32> }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> quats: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> operators: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> rows: array<Row>;
var<workgroup> tile: array<vec4<f32>, 64>;
var<workgroup> tile_high: array<vec4<f32>, 64>;
var<workgroup> tile_low: array<vec4<f32>, 64>;

fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.x*b.x-dot(a.yzw,b.yzw), a.x*b.yzw+b.x*a.yzw+cross(a.yzw,b.yzw));
}
// p.pad is a runtime zero. Crossing an integer XOR makes rounding observable,
// preventing fast-math reassociation from erasing compensation/error terms.
fn materialize(a:f32) -> f32 { return bitcast<f32>(bitcast<u32>(a)^p.pad); }
fn materialize4(a:vec4<f32>) -> vec4<f32> { return bitcast<vec4<f32>>(bitcast<vec4<u32>>(a)^vec4<u32>(p.pad)); }
fn add4(s: ptr<function,vec4<f32>>, c: ptr<function,vec4<f32>>, v: vec4<f32>) {
    let y = materialize4(v - *c);
    let t = materialize4(*s + y);
    *c = materialize4(materialize4(t - *s) - y);
    *s = t;
}
// Double-single arithmetic used for the logarithmic remainder: pairwise f32 log
// errors otherwise accumulate systematically when the O(N^2) mean is removed.
fn ds(a: f32) -> vec2<f32> { return vec2<f32>(a,0.0); }
fn dadd(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = materialize(a.x+b.x);
    let v = materialize(s-a.x);
    let e1=materialize(a.x-materialize(s-v));
    let e2=materialize(b.x-v);
    let e=materialize(materialize(materialize(e1+e2)+a.y)+b.y);
    let h = materialize(s+e);
    return vec2<f32>(h,materialize(e-materialize(h-s)));
}
fn dneg(a: vec2<f32>) -> vec2<f32> { return -a; }
fn dmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let h = materialize(a.x*b.x);
    let e0=materialize(fma(a.x,b.x,-h));
    let e=materialize(materialize(materialize(e0+materialize(a.x*b.y))+materialize(a.y*b.x))+materialize(a.y*b.y));
    return dadd(ds(h),ds(e));
}
fn ddiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let q = materialize(a.x/b.x);
    let r = dadd(a,dneg(dmul(ds(q),b)));
    return dadd(ds(q),ds(materialize(materialize(r.x+r.y)/b.x)));
}
fn dsqrt(a: vec2<f32>) -> vec2<f32> {
    let q = materialize(sqrt(a.x));
    let r = dadd(a,dneg(dmul(ds(q),ds(q))));
    return dadd(ds(q),ds((r.x+r.y)/(2.0*q)));
}
fn dnorm(a: vec4<f32>) -> vec2<f32> {
    var s = ds(0.0);
    for(var i=0u;i<4u;i++) { s=dadd(s,dmul(ds(a[i]),ds(a[i]))); }
    return s;
}
fn dnorm2(a: vec4<f32>, low:vec4<f32>) -> vec2<f32> {
    var s = ds(0.0);
    for(var i=0u;i<4u;i++) { let v=vec2<f32>(a[i],low[i]); s=dadd(s,dmul(v,v)); }
    return s;
}
struct DQuat { high:vec4<f32>, low:vec4<f32> }
fn accurate_multiply(a:vec4<f32>,b:vec4<f32>) -> DQuat {
    var result:DQuat;
    let aw=dmul(ds(a.x),ds(b.x));
    var w=aw;
    for(var i=1u;i<4u;i++) { w=dadd(w,dneg(dmul(ds(a[i]),ds(b[i])))); }
    result.high.x=w.x;result.low.x=w.y;
    for(var i=0u;i<3u;i++) {
        let j=(i+1u)%3u;let k=(i+2u)%3u;
        var v=dadd(dmul(ds(a.x),ds(b[i+1u])),dmul(ds(b.x),ds(a[i+1u])));
        v=dadd(v,dmul(ds(a[j+1u]),ds(b[k+1u])));
        v=dadd(v,dneg(dmul(ds(a[k+1u]),ds(b[j+1u]))));
        result.high[i+1u]=v.x;result.low[i+1u]=v.y;
    }
    return result;
}
fn ddist(a: vec4<f32>, b: vec4<f32>, low:vec4<f32>, sign: f32) -> vec2<f32> {
    var s = ds(0.0);
    for(var i=0u;i<4u;i++) {
        let v=dadd(ds(a[i]),sign*vec2<f32>(b[i],low[i]));
        s=dadd(s,dmul(v,v));
    }
    return s;
}
fn dlog(a: vec2<f32>) -> vec2<f32> {
    // Range reduce to [sqrt(1/2),sqrt(2)], then use the atanh series.
    let bits=bitcast<u32>(a.x);
    var exponent=i32((bits>>23u)&255u)-127;
    var m=vec2<f32>(bitcast<f32>((bits&0x7fffffu)|0x3f800000u), a.y*exp2(f32(-exponent)));
    if(m.x>1.4142135623730951) { m*=0.5; exponent+=1; }
    let y=ddiv(dadd(m,ds(-1.0)),dadd(m,ds(1.0)));
    let z=dmul(y,y);
    var term=y;
    var sum=y;
    for(var k=3u;k<=19u;k+=2u) {
        term=dmul(term,z);
        sum=dadd(sum,ddiv(term,ds(f32(k))));
    }
    let ln2=vec2<f32>(0.693147182464599609375,-1.904654299957768e-9);
    return dadd(dmul(ds(2.0),sum),dmul(ds(f32(exponent)),ln2));
}
fn accurate_log_pair(x:vec4<f32>, y:vec4<f32>, low:vec4<f32>) -> vec2<f32> {
    let a=dnorm(x); let b=dnorm2(y,low);
    let t=dsqrt(dmul(a,b));
    let diff=dadd(a,dneg(b));
    let correction=ddiv(dmul(diff,diff),dadd(dadd(a,b),dmul(ds(2.0),t)));
    let dm=ddiv(dadd(ddist(x,y,low,-1.0),dneg(correction)),t);
    let dp=ddiv(dadd(ddist(x,y,low,1.0),dneg(correction)),t);
    let product=dmul(dm,dp);
    if(product.x<=0.0) { return ds(1e30); }
    return dadd(dmul(ds(-0.5),dlog(product)),ds(0.5));
}

// Per-site double-single factors of the log term. With a = |x|^2 and b = |y|^2 (the
// image, from its high/low parts), the unit chords satisfy
//   chord(x,y)^2 chord(x,-y)^2 = n_minus n_plus / (a b),  n_(-/+) = |x -/+ y|^2 - (sqrt a - sqrt b)^2,
// so a pair needs two distances and products; square roots and 1/b are per site.
struct LogSite { root: vec2<f32>, inverse: vec2<f32> }
fn log_site(high:vec4<f32>, low:vec4<f32>) -> LogSite {
    let b=dnorm2(high,low);
    return LogSite(dsqrt(b),ddiv(ds(1.0),b));
}
// n_minus n_plus / b for one pair; the caller divides by a once per tile.
fn log_pair(x:vec4<f32>, x_root:vec2<f32>, high:vec4<f32>, low:vec4<f32>, site:LogSite) -> vec2<f32> {
    let gap=dadd(x_root,dneg(site.root));let correction=dmul(gap,gap);
    let minus=dadd(ddist(x,high,low,-1.0),dneg(correction));
    let plus=dadd(ddist(x,high,low,1.0),dneg(correction));
    return dmul(dmul(minus,plus),site.inverse);
}
// Exact power-of-two rescaling keeps a long double-single product in range.
fn rescale(product:ptr<function,vec2<f32>>, exponent:ptr<function,i32>) {
    let parts=frexp((*product).x);
    *product=vec2<f32>(ldexp((*product).x,-parts.exp),ldexp((*product).y,-parts.exp));
    *exponent+=parts.exp;
}
var<workgroup> tile_log: array<LogSite, 64>;

@compute @workgroup_size(64)
fn pairs(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
    let i=gid.x;
    let live=i<p.n;
    let force_only=(p.flags&1u)!=0u;
    let first=(p.flags>>8u)&255u;
    let span=(p.flags>>16u)&255u;
    let end=select(min(p.order,first+span),p.order,span==0u);
    var x=vec4<f32>(1.0,0.0,0.0,0.0);
    if(live) { x=quats[i]; }
    var energy=vec4<f32>(0.0); var ec=vec4<f32>(0.0);
    var force=vec4<f32>(0.0); var fc=vec4<f32>(0.0);
    var extra=vec4<f32>(0.0); var xc=vec4<f32>(0.0);
    var logarithm=ds(0.0);
    var nearest=4.0;
    let j_end=select(p.j_end,p.n,p.j_end==0u);
    if(live && (first>0u || p.resume!=0u)) {
        energy=rows[i].energy;extra=rows[i].extra;force=rows[i].force;
        logarithm=vec2<f32>(energy.w,extra.w);nearest=extra.z;
    }
    // Per-thread log factors: sqrt(a) for the pair terms, log(a) once per tile.
    let x_norm=dnorm(x);let x_root=dsqrt(x_norm);let x_log=dlog(x_norm);
    var degenerate=false;
    for(var k=first;k<end;k++) {
        let op=operators[k];
        for(var start=p.j_first;start<j_end;start+=64u) {
            let j=start+lid.x;
            if(j<j_end) {
                // Keep the identity image bit-for-bit identical; repeated f32
                // normalization can otherwise conceal an exact input duplicate.
                tile[lid.x]=select(normalize(qmul(quats[j],op)),quats[j],k==0u);
                if(!force_only) {
                    let refined=accurate_multiply(quats[j],op);
                    tile_high[lid.x]=refined.high;tile_low[lid.x]=refined.low;
                    tile_log[lid.x]=log_site(refined.high,refined.low);
                }
            }
            workgroupBarrier();
            if(live) {
                let count=min(64u,j_end-start);
                // Plain sums over one tile, merged into the compensated row sums once per tile;
                // the log terms multiply into one double-single product, logged once per tile.
                var tile_energy=vec4<f32>(0.0);var tile_extra=vec2<f32>(0.0);var tile_force=vec4<f32>(0.0);
                var product=ds(1.0);var exponent=0i;var terms=0u;
                for(var t=0u;t<count;t++) {
                    let other=start+t;
                    let y=tile[t];
                    let minus=x-y; let plus=x+y;
                    let dm2=dot(minus,minus); let dp2=dot(plus,plus);
                    let is_self=other==i && k==0u;
                    if(!is_self) { nearest=min(nearest,min(dm2,dp2)); }
                    let pair=quey_pair(x,y,dm2,dp2);
                    if(!is_self && !force_only) {
                        // Float32 differences resolve near-coincident sites where 1-dot fails.
                        let im=inverseSqrt(max(dm2,1e-30)); let ip=inverseSqrt(max(dp2,1e-30));
                        tile_energy+=vec4<f32>(im+ip,im*im+ip*ip,im*im*im+ip*ip*ip,0.0);
                        // The paired Green expression is sign invariant.
                        let cot=abs(dp2-dm2)*0.5*im*ip;
                        tile_extra.x+=((PI-2.0*pair.angle)*cot-1.0)/FOUR_PI2;
                        let term=log_pair(x,x_root,tile_high[t],tile_low[t],tile_log[t]);
                        degenerate=degenerate||!(term.x>0.0);
                        product=dmul(product,term);rescale(&product,&exponent);terms++;
                    }
                    // Own-orbit interactions are constant under a source move: exclude all.
                    if(other!=i) { tile_force+=pair.force;tile_extra.y+=pair.energy; }
                }
                add4(&energy,&ec,tile_energy);
                add4(&extra,&xc,vec4<f32>(tile_extra,0.0,0.0));
                add4(&force,&fc,tile_force);
                if(terms>0u) {
                    // sum of -0.5 log(n_minus n_plus / (a b)) + 0.5 over the tile's terms.
                    let ln2=vec2<f32>(0.693147182464599609375,-1.904654299957768e-9);
                    var total=dadd(dlog(product),dmul(ds(f32(exponent)),ln2));
                    total=dadd(total,dneg(dmul(ds(f32(terms)),x_log)));
                    logarithm=dadd(logarithm,dadd(dmul(ds(-0.5),total),ds(0.5*f32(terms))));
                }
            }
            workgroupBarrier();
        }
    }
    if(live) {
        if(degenerate) { logarithm=ds(1e30); }
        if(!force_only && end==p.order && j_end==p.n) {
            // The omitted identity self pair still has a real antipode.
            energy+=vec4<f32>(0.5,0.25,0.125,0.0);
            extra.x+=-3.0/(2.0*FOUR_PI2);
            logarithm=dadd(logarithm,vec2<f32>(-0.4431471824645996,1.904654299957768e-9));
        }
        energy.w=logarithm.x;
        extra.z=nearest;
        extra.w=logarithm.y;
        rows[i]=Row(energy,extra,force);
    }
}

// One Quey pair, with angle a = d(x, y) on S3 (the nearer of y and -y) and b = pi - a:
// force magnitude 1/(a^2+e) - 1/(b^2+e) along the geodesic, curvature 1/(a^3+e) + 1/(b^3+e)
// and energy 1/a + 1/b. With t = tan(a/2) = chord(near)/chord(far), sin a = 2t/(1+t^2):
// one square root, one arctangent and four divisions per pair.
struct QueyPair { force: vec4<f32>, energy: f32, angle: f32 }
fn quey_pair(x:vec4<f32>,y:vec4<f32>,dm2:f32,dp2:f32)->QueyPair{
    let near=max(min(dm2,dp2),1e-30);let far=max(max(dm2,dp2),1e-30);
    let t=sqrt(near/far);
    let a=max(2.0*atan(t),1e-12);let b=PI-a;
    let a2=a*a+1e-9;let b2=b*b+1e-9;let a3=a*a*a+1e-9;let b3=b*b*b+1e-9;
    let sine=2.0*t/(1.0+t*t)+1e-9;
    let sign=select(-1.0,1.0,dm2<=dp2);
    let direction=x.x*y.yzw-y.x*x.yzw+cross(x.yzw,y.yzw);
    let along=sign*(b2-a2)/(a2*b2*sine);
    let curvature=(a3+b3)/(a3*b3);
    return QueyPair(vec4<f32>(-along*direction,curvature),PI/(a*b),a);
}
// Fast Quey path. Here flags is the number of operators per partial row.
// Parallel operator batches keep small orientation sets from launching only
// ~16 workgroups. The stored intermediate is partial sums, never image quats.
@compute @workgroup_size(64)
fn quey_partial(@builtin(global_invocation_id) gid:vec3<u32>,
                @builtin(local_invocation_id) lid:vec3<u32>,
                @builtin(workgroup_id) workgroup:vec3<u32>) {
    let i=gid.x;let live=i<p.n;
    let first=workgroup.y*p.flags;
    let end=min(p.order,first+p.flags);
    var x=vec4<f32>(1.0,0.0,0.0,0.0);
    if(live){x=quats[i];}
    var force=vec4<f32>(0.0);var fc=vec4<f32>(0.0);
    var extra=vec4<f32>(0.0);var xc=vec4<f32>(0.0);
    var near=4.0;
    let j_end=select(p.j_end,p.n,p.j_end==0u);
    if(live && p.resume!=0u) {
        let before=rows[workgroup.y*p.n+i];
        force=before.force;extra=vec4<f32>(before.extra.xy,0.0,0.0);near=before.extra.z;
    }
    for(var k=first;k<end;k++) {
        let op=operators[k];
        for(var start=p.j_first;start<j_end;start+=64u) {
            let j=start+lid.x;
            if(j<j_end){tile[lid.x]=select(normalize(qmul(quats[j],op)),quats[j],k==0u);}
            workgroupBarrier();
            if(live) {
                // Plain sums over one tile, merged into the compensated row sums once per tile.
                var tile_force=vec4<f32>(0.0);var tile_energy=0.0;
                for(var t=0u;t<min(64u,j_end-start);t++) {
                    let other=start+t;let y=tile[t];
                    let minus=x-y;let plus=x+y;
                    let dm2=dot(minus,minus);let dp2=dot(plus,plus);
                    if(other!=i || k!=0u){near=min(near,min(dm2,dp2));}
                    if(other!=i) {
                        let pair=quey_pair(x,y,dm2,dp2);
                        tile_force+=pair.force;tile_energy+=pair.energy;
                    }
                }
                add4(&force,&fc,tile_force);add4(&extra,&xc,vec4<f32>(0.0,tile_energy,0.0,0.0));
            }
            workgroupBarrier();
        }
    }
    if(live){extra.z=near;rows[workgroup.y*p.n+i]=Row(vec4<f32>(0.0),extra,force);}
}

@group(0) @binding(6) var<storage,read> partial_rows:array<Row>;
@group(0) @binding(7) var<storage,read_write> final_rows:array<Row>;
@compute @workgroup_size(64)
fn quey_reduce(@builtin(global_invocation_id) gid:vec3<u32>) {
    let i=gid.x;if(i>=p.n){return;}
    var force=vec4<f32>(0.0);var fc=vec4<f32>(0.0);
    var extra=vec4<f32>(0.0);var xc=vec4<f32>(0.0);
    var near=4.0;
    for(var batch=0u;batch<(p.order+p.flags-1u)/p.flags;batch++) {
        let row=partial_rows[batch*p.n+i];
        add4(&force,&fc,row.force);
        add4(&extra,&xc,vec4<f32>(row.extra.xy,0.0,0.0));
        near=min(near,row.extra.z);
    }
    extra.z=near;
    final_rows[i]=Row(vec4<f32>(0.0),extra,force);
}

// Near-field Quey forces (src/near.zig; Quey et al. 2018, section 2.3): each source sums
// quey_pair over the orbit points within S3 angle r of it, its own orbit excluded, as in
// quey_partial. The sites are every orbit image within r of the zone (shell_build), sorted
// by cell of a stereographic grid. The sources are sorted in Morton order of a fine grid,
// so 64 consecutive ones are close together: a workgroup takes such a batch and one of
// `slices` interleaved shares of the site rows its box can reach, and writes partial rows
// (slice * count + position in the chunk) that near_reduce sums. A pass runs in chunks of
// `count` sorted sources from `first`, bounding the partial rows' memory. A site counts when its chord to the source
// is at most 2 sin(r/2): each orbit point once (its near twin, when the far one is the
// site, is also a site). Its quaternion is rebuilt from its code exactly as quey_partial
// builds images.
struct NearParams { slices:u32, sdim:u32, n:u32, first:u32, count:u32, pad0:u32, pad1:u32, pad2:u32, lower:vec4<f32>, cut:vec4<f32> } // lower.w: cells per unit; cut: chord^2, chord, taper start angle, 1/taper width
@group(0) @binding(4) var<uniform> near:NearParams;
@group(0) @binding(8) var<storage,read> near_images:array<vec4<f32>>;         // by site cell: the site's quaternion (near_images)
@group(0) @binding(9) var<storage,read> near_site_offsets:array<u32>;
@group(0) @binding(10) var<storage,read> near_codes:array<u32>;                // by site cell: its source, high bit its identity image
@group(0) @binding(11) var<storage,read> near_sources:array<vec4<f32>>;        // Morton order: position, w the source
var<workgroup> near_source:array<u32,64>;  // the site's source; high bit: its identity image
var<workgroup> near_low:array<vec3<f32>,64>;
var<workgroup> near_high:array<vec3<f32>,64>;
var<workgroup> near_span:vec2<u32>;
// The pair potential times a window falling smoothly (C1) from 1 at the taper start to 0
// at the cutoff, and its exact force: the cutoff's jump in force would otherwise shake
// every source whose neighbourhood edge a point crosses.
fn near_pair(x:vec4<f32>,y:vec4<f32>,dm2:f32,dp2:f32)->QueyPair{
    let pair=quey_pair(x,y,dm2,dp2);
    let s=clamp((pair.angle-near.cut.z)*near.cut.w,0.0,1.0);
    if(s<=0.0){return pair;}
    let window=1.0-s*s*(3.0-2.0*s);let slope=-6.0*s*(1.0-s)*near.cut.w;
    let sign=select(-1.0,1.0,dm2<=dp2);
    let direction=x.x*y.yzw-y.x*x.yzw+cross(x.yzw,y.yzw);
    let extra=sign*pair.energy*slope/max(sin(pair.angle),1e-9);
    return QueyPair(vec4<f32>(window*pair.force.xyz+extra*direction,window*pair.force.w),window*pair.energy,pair.angle);
}
fn near_cell(x:vec3<f32>)->vec3<i32>{return vec3<i32>(clamp(floor((x-near.lower.xyz)*near.lower.w),vec3<f32>(0.0),vec3<f32>(f32(near.sdim-1u))));}
@compute @workgroup_size(64)
fn near_force(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let batch=wid.x;let slice=wid.y;let local=batch*64u+lane;let at=near.first+local;let live=local<near.count;
    var id=0u;var x=vec4<f32>(1.0,0.0,0.0,0.0);var low=vec3<f32>(3e38);var high=vec3<f32>(-3e38);
    if(live){let s=near_sources[at];id=bitcast<u32>(s.w);x=quats[id];low=s.xyz;high=s.xyz;}
    // The sources' box, and how far in the chart a point within the cutoff chord can be:
    // stereographic distance = chord/2 * sqrt((1+|p|^2)(1+|p'|^2)), bounded by iteration.
    near_low[lane]=low;near_high[lane]=high;workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){
        if(lane<stride){near_low[lane]=min(near_low[lane],near_low[lane+stride]);near_high[lane]=max(near_high[lane],near_high[lane+stride]);}workgroupBarrier();
    }
    let box_low=workgroupUniformLoad(&near_low[0]);let box_high=workgroupUniformLoad(&near_high[0]);
    let reach=length(max(abs(box_low),abs(box_high)));let lift=1.0+reach*reach;
    var d=0.5*near.cut.y*lift;
    for(var k=0u;k<4u;k++){d=0.5*near.cut.y*sqrt(lift*(1.0+(reach+d)*(reach+d)));}
    d=d*1.01+1e-6;
    let lo=near_cell(box_low-vec3<f32>(d));let hi=near_cell(box_high+vec3<f32>(d));
    let width=1.0/near.lower.w;let ny=u32(hi.y-lo.y)+1u;let row_count=ny*(u32(hi.z-lo.z)+1u);
    var force=vec4<f32>(0.0);var fc=vec4<f32>(0.0);var extra=vec4<f32>(0.0);var xc=vec4<f32>(0.0);var nearest=4.0;
    for(var r=slice;r<row_count;r+=near.slices){
        let y=lo.y+i32(r%ny);let z=lo.z+i32(r/ny);
        // The row's cells reach no closer to the box than gap in y and z; its run is
        // trimmed in x to the cells within d of the box.
        let cell_low=near.lower.yz+vec2<f32>(f32(y),f32(z))*width;
        let gap=max(vec2<f32>(0.0),max(cell_low-box_high.yz,box_low.yz-(cell_low+width)));
        let rest=d*d-dot(gap,gap);
        if(rest<0.0){continue;} // uniform: every lane has the same row
        let span=sqrt(rest);
        let x0=near_cell(vec3<f32>(box_low.x-span,0.0,0.0)).x;let x1=near_cell(vec3<f32>(box_high.x+span,0.0,0.0)).x;
        // One row of cells is one contiguous run of sites.
        if(lane==0u){let row=near.sdim*(u32(y)+near.sdim*u32(z));near_span=vec2<u32>(near_site_offsets[row+u32(x0)],near_site_offsets[row+u32(x1)+1u]);}
        let run=workgroupUniformLoad(&near_span);
        for(var start=run.x;start<run.y;start+=64u){
            let k=start+lane;
            workgroupBarrier(); // the previous tile is done
            if(k<run.y){tile[lane]=near_images[k];near_source[lane]=near_codes[k];}
            workgroupBarrier();
            if(live){
                var tile_force=vec4<f32>(0.0);var tile_energy=0.0;
                for(var t=0u;t<min(64u,run.y-start);t++){
                    let other=tile[t];let minus=x-other;let dm2=dot(minus,minus);
                    if(dm2>near.cut.x){continue;}
                    let source=near_source[t];
                    // Its own orbit: no force; its other images bound the separation.
                    if((source&0x7fffffffu)==id){if(source==id){nearest=min(nearest,dm2);}continue;}
                    nearest=min(nearest,dm2);
                    let plus=x+other;let pair=near_pair(x,other,dm2,dot(plus,plus));tile_force+=pair.force;tile_energy+=pair.energy;
                }
                add4(&force,&fc,tile_force);add4(&extra,&xc,vec4<f32>(0.0,tile_energy,0.0,0.0));
            }
        }
    }
    if(live){extra.z=nearest;rows[slice*near.count+local]=Row(vec4<f32>(0.0),extra,force);}
}
// Each chunk source's slices, summed into its row.
@compute @workgroup_size(64)
fn near_reduce(@builtin(global_invocation_id) gid:vec3<u32>){
    let k=gid.x;if(k>=near.count){return;}
    var force=vec4<f32>(0.0);var fc=vec4<f32>(0.0);var extra=vec4<f32>(0.0);var xc=vec4<f32>(0.0);var nearest=4.0;
    for(var slice=0u;slice<near.slices;slice++){
        let row=partial_rows[slice*near.count+k];
        add4(&force,&fc,row.force);add4(&extra,&xc,vec4<f32>(row.extra.xy,0.0,0.0));nearest=min(nearest,row.extra.z);
    }
    extra.z=nearest;final_rows[bitcast<u32>(near_sources[near.first+k].w)]=Row(vec4<f32>(0.0),extra,force);
}
