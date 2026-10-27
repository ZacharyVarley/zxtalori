// Per-site halfspace intersection. Stars are private, so no cross-star locks
// or CUDA warp-synchronous assumptions enter the WGSL topology updates.
fn stride()->u32{return 32u+2u*p.cap;}
@group(0) @binding(11) var<storage,read_write> spatial_order:array<u32>;
var<workgroup> order_prefix:array<u32,128>;
var<workgroup> shared_faces:array<vec4<u32>,FACE_CAP>;
var<workgroup> shared_coefficients:array<vec4<f32>,FACE_CAP>;
var<workgroup> shared_bounds:array<vec4<f32>,FACE_CAP>;
fn face_load(s:u32,i:u32)->vec4<u32>{if(SHARED_FACES){return shared_faces[i];}return faces[i*p.seeds+s];}
fn face_ids(s:u32,i:u32,f:vec4<u32>){if(SHARED_FACES){shared_faces[i]=f;}else{faces[i*p.seeds+s]=f;}}
fn face_coeff(s:u32,i:u32)->vec4<f32>{if(SHARED_FACES){return shared_coefficients[i];}return bitcast<vec4<f32>>(faces[(p.cap+i)*p.seeds+s]);}
fn face_bound(s:u32,i:u32)->vec4<f32>{if(SHARED_FACES){return shared_bounds[i];}return bitcast<vec4<f32>>(faces[(2u*p.cap+i)*p.seeds+s]);}
fn face_geometry(s:u32,i:u32,c:vec4<f32>,b:vec4<f32>){
    if(SHARED_FACES){shared_coefficients[i]=c;shared_bounds[i]=b;}
    else{faces[(p.cap+i)*p.seeds+s]=bitcast<vec4<u32>>(c);faces[(2u*p.cap+i)*p.seeds+s]=bitcast<vec4<u32>>(b);}
}
struct FaceGeometry{coefficient:vec4<f32>,bound:vec4<f32>}
fn calculate_geometry(seed:u32,f:vec4<u32>)->FaceGeometry{
    var coeff=vec4<f32>(0.0);var bound=vec4<f32>(0.0);
    if(bounded_point(seed)&&bounded_point(f.x)&&bounded_point(f.y)&&bounded_point(f.z)){
        let o=point(seed);let a=point(f.x)-o;let b=point(f.y)-o;let c=point(f.z)-o;
        let lift=vec3<f32>(dot(a,a),dot(b,b),dot(c,c));
        coeff=vec4<f32>(lift.x*cross(b,c)+lift.y*cross(c,a)+lift.z*cross(a,b),det(a,b,c));
        let bcp=abs(b.yzx*c.zxy)+abs(b.zxy*c.yzx);let cap=abs(c.yzx*a.zxy)+abs(c.zxy*a.yzx);let abp=abs(a.yzx*b.zxy)+abs(a.zxy*b.yzx);
        bound=vec4<f32>(lift.x*bcp+lift.y*cap+lift.z*abp,permanent(a,b,c));
        let magnitude=max(vec3<f32>(1.0),max(max(abs(a),abs(b)),abs(c)));let m=max(max(magnitude.x,magnitude.y),magnitude.z);let noise=1e-29*m*m;
        bound+=vec4<f32>(vec3<f32>(noise*m),noise);
    }
    return FaceGeometry(coeff,bound);
}
fn face_store(s:u32,i:u32,f:vec4<u32>){face_ids(s,i,f);if(CACHED_FACES){let g=calculate_geometry(current_star.more.z,f);face_geometry(s,i,g.coefficient,g.bound);}}
fn face_copy(s:u32,to:u32,source:u32){let f=face_load(s,source);face_ids(s,to,f);if(CACHED_FACES){let c=face_coeff(s,source);let b=face_bound(s,source);face_geometry(s,to,c,b);}}
fn face_mark(s:u32,i:u32,status:u32){var f=face_load(s,i);f.w=status;face_ids(s,i,f);}
fn face_filter(s:u32,i:u32,q:u32)->i32{
    let seed=seed_id(s);let f=face_load(s,i);if(any(vec4<u32>(seed,f.xyz)==vec4<u32>(q))){return 0;}
    if(!CACHED_FACES){return sphere_quick(vec4<u32>(seed,f.xyz),q);}
    let c=face_coeff(s,i);let b=face_bound(s,i);if(b.w==0.0||!bounded_point(q)){return 2;}
    let d=point(q)-point(seed);let lift=dot(d,d);let value=rounded(c.w*lift-dot(c.xyz,d));let error=max(1e-28,4e-6*(b.w*lift+dot(b.xyz,abs(d))));
    if(abs(value)>error&&abs(value)<3e37){return select(-1,1,value>0.0);}return 2;
}
fn prepare_face_query(s:u32,i:u32){
    let mem=32u;var c:vec4<f32>;var b:vec4<f32>;
    if(CACHED_FACES){c=face_coeff(s,i);b=face_bound(s,i);}else{let g=calculate_geometry(seed_id(s),face_load(s,i));c=g.coefficient;b=g.bound;}
    let o=point(seed_id(s));
    for(var k=0u;k<4u;k++){set_word(s,mem+8u+2u*k,bitcast<u32>(c[k]));set_word(s,mem+9u+2u*k,0u);set_word(s,mem+16u+k,bitcast<u32>(b[k]));}
    for(var k=0u;k<3u;k++){set_word(s,mem+20u+k,bitcast<u32>(o[k]));}
    set_word(s,mem+7u,select(0u,256u,b.w>0.0));set_word(s,mem+23u,0u);set_word(s,mem+24u,0u);
}
fn seed_id(s:u32)->u32{return cached_seed;}
// Gao section 5.2: certify mixed-order faces only. A strict lexicographic
// linear order handles equal x coordinates without moving any input point.
fn before(a:u32,b:u32)->bool{
    let x=point(a);let y=point(b);
    if(x.x!=y.x){return x.x<y.x;}if(x.y!=y.y){return x.y<y.y;}if(x.z!=y.z){return x.z<y.z;}return a<b;
}
fn greater_count(seed:u32,f:vec3<u32>)->u32{return u32(before(seed,f.x))+u32(before(seed,f.y))+u32(before(seed,f.z));}
fn owns(seed:u32,f:vec3<u32>)->bool{
    if(p.pad==1u){return greater_count(seed,f)==1u;}return seed<min(min(f.x,f.y),f.z);
}
fn cell_of(x:vec3<f32>)->vec3<u32>{return vec3<u32>(clamp(floor((x-p.lower.xyz)*p.scale.x),vec3<f32>(0),vec3<f32>(f32(p.dim-1u))));}
fn cell_index(c:vec3<u32>)->u32{return c.x+p.dim*(c.y+p.dim*c.z);}
@compute @workgroup_size(128)
fn grid_clear(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<p.dim*p.dim*p.dim){atomicStore(&heads[id.x],NONE);atomicStore(&heads[id.x+p.dim*p.dim*p.dim],0u);}}
@compute @workgroup_size(128)
fn grid_build(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<p.n){let cell=cell_index(cell_of(point(id.x)));next_site[id.x]=atomicExchange(&heads[cell],id.x);atomicAdd(&heads[cell+p.dim*p.dim*p.dim],1u);}}
@compute @workgroup_size(128)
fn grid_order(@builtin(local_invocation_index) lane:u32){
    let cells=p.dim*p.dim*p.dim;let block=(cells+127u)/128u;let begin=lane*block;let end=min(cells,begin+block);var count=0u;
    for(var c=begin;c<end;c++){count+=atomicLoad(&heads[c+cells]);}order_prefix[lane]=count;workgroupBarrier();
    for(var step=1u;step<128u;step*=2u){var prior=0u;if(lane>=step){prior=order_prefix[lane-step];}workgroupBarrier();order_prefix[lane]+=prior;workgroupBarrier();}
    var offset=order_prefix[lane]-count;
    for(var c=begin;c<end;c++){
        next_site[p.n+2u*c]=offset;
        var q=atomicLoad(&heads[c]);while(q!=NONE){spatial_order[offset]=q;offset++;q=next_site[q];}
        next_site[p.n+2u*c+1u]=offset;
    }
    storageBarrier();workgroupBarrier();
    for(var i=lane;i<p.n;i+=128u){next_site[i]=spatial_order[i];}
}
fn open_cell(s:u32){let cell=cell_index(current_star.cell.xyz);let first=next_site[p.n+2u*cell];let end=next_site[p.n+2u*cell+1u];current_star.cell.w=select(first,NONE,first==end);current_star.more.x=end+1u;}
fn query_point(s:u32)->u32{return next_site[current_star.cell.w];}
fn consume_point(s:u32){let at=current_star.cell.w+1u;current_star.cell.w=select(at,NONE,at>=current_star.more.x-1u);}

@compute @workgroup_size(64)
fn star_init(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=p.seeds){return;}var seed=p.first+s;if(p.pad==1u){seed=spatial_order[seed];}let base=s*p.cap;let mem=0u;
    current_star=Star(vec4<u32>(1u,4u,0u,0u),vec4<u32>(0),vec4<u32>(NONE,1u,NONE,0u),vec4<u32>(0),vec4<u32>(0),vec4<u32>(0),vec4<u32>(0));
    current_star.more.z=seed;
    face_store(s,0u,vec4<u32>(p.n+1u,p.n+3u,p.n+2u,0u));face_store(s,1u,vec4<u32>(p.n,p.n+2u,p.n+3u,0u));
    face_store(s,2u,vec4<u32>(p.n,p.n+3u,p.n+1u,0u));face_store(s,3u,vec4<u32>(p.n,p.n+1u,p.n+2u,0u));
    for(var k=0u;k<8u;k++){cache[s*8u+k].more.w=0u;}
    var near:array<u32,32>;var distance:array<f32,32>;var count=0u;
    for(var k=0u;k<32u;k++){near[k]=NONE;distance[k]=3e38;}
    let centre=vec3<i32>(cell_of(point(seed)));let lo=max(centre-vec3<i32>(1),vec3<i32>(0));let hi=min(centre+vec3<i32>(1),vec3<i32>(i32(p.dim)-1));
    for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
        if(abs(x-centre.x)+abs(y-centre.y)+abs(z-centre.z)>1){continue;}
        let cell=cell_index(vec3<u32>(u32(x),u32(y),u32(z)));let first=next_site[p.n+2u*cell];let end=next_site[p.n+2u*cell+1u];
        for(var position=first;position<end;position++){let q=next_site[position];if(q!=seed){if(all(signed_bits(coordinates[q].xyz)==signed_bits(coordinates[seed].xyz))){fail(s,9u);stars[s]=current_star;return;}let v=point(q)-point(seed);let d=dot(v,v);var at=min(count,32u);
            for(var k=0u;k<min(count,32u);k++){if(d<distance[k]||(d==distance[k]&&q<near[k])){at=k;break;}}
            if(at<32u){for(var k=31u;k>at;k--){near[k]=near[k-1u];distance[k]=distance[k-1u];}near[at]=q;distance[at]=d;count=min(32u,count+1u);}
        }}
    }}}
    for(var k=0u;k<count;k++){set_word(s,mem+k,near[k]);}current_star.info.w=count;stars[s]=current_star;
}

// Outward-rounded interval Cramer solve. An uncertain/singular enclosure scans
// the whole grid; it never certifies a narrowed search from an unstable centre.
struct Box{lo:vec3<u32>,hi:vec3<u32>}
fn sphere_box(s:u32,seed:u32,f:vec3<u32>)->Box{
    let mem=32u;set_word(s,mem+6u,0u);
    let full=Box(vec3<u32>(0),vec3<u32>(p.dim-1u));var rows:array<array<Interval,3>,3>;var lift:array<Interval,3>;
    let origin=point(seed);for(var i=0u;i<3u;i++){let v=point(f[i]);lift[i]=iv(0.0);for(var k=0u;k<3u;k++){rows[i][k]=minus(iv(v[k]),iv(origin[k]));lift[i]=plus(lift[i],times(rows[i][k],rows[i][k]));}}
    let determinant=determinant_i(rows[0],rows[1],rows[2]);if(determinant.lo<=0.0&&determinant.hi>=0.0){return full;}
    var low:vec3<f32>;var high:vec3<f32>;var radius=iv(0.0);
    for(var k=0u;k<3u;k++){
        var m=rows;for(var i=0u;i<3u;i++){m[i][k]=lift[i];}let c=times(iv(0.5),quotient(determinant_i(m[0],m[1],m[2]),determinant));
        if(!(max(abs(c.lo),abs(c.hi))<1e15)){return full;}
        set_word(s,mem+k*2u,bitcast<u32>(c.lo));set_word(s,mem+k*2u+1u,bitcast<u32>(c.hi));
        let bound=max(abs(c.lo),abs(c.hi));radius=plus(radius,times(iv(bound),iv(bound)));
        low[k]=down(origin[k]+c.lo);high[k]=up(origin[k]+c.hi);
    }
    let clipped=clipped_sphere(low,high,radius.hi);low=clipped[0];high=clipped[1];
    set_word(s,mem+6u,1u);
    // One extra cell also covers float rounding in the spatial index itself.
    let a=vec3<i32>(cell_of(low));let b=vec3<i32>(cell_of(high));return Box(vec3<u32>(max(a-vec3<i32>(1),vec3<i32>(0))),vec3<u32>(min(b+vec3<i32>(1),vec3<i32>(i32(p.dim)-1))));
}
// A sphere's AABB includes many irrelevant sites. The same interval centre
// proves |q-seed|^2 - 2(q-seed).centre > 0 before an expensive exact-filtered
// determinant. Boundary/uncertain cases still use the ordinary predicate.
fn outside_sphere(s:u32,seed:u32,q:u32)->bool{
    let mem=32u;if(word(s,mem+6u)==0u){return false;}
    let a=point(seed);let b=point(q);var lift=iv(0.0);var inner=iv(0.0);
    for(var k=0u;k<3u;k++){
        let v=minus(iv(b[k]),iv(a[k]));let c=Interval(bitcast<f32>(word(s,mem+k*2u)),bitcast<f32>(word(s,mem+k*2u+1u)));
        lift=plus(lift,times(v,v));inner=plus(inner,times(v,c));
    }
    return minus(lift,times(iv(2.0),inner)).lo>0.0;
}
fn fail(s:u32,code:u32){current_star.info.x=10u;current_star.more.z=code;}
fn advance_cell(s:u32)->bool {
    var c=vec3<i32>(current_star.cell.xyz);let lo=vec3<i32>(current_star.low.xyz);let hi=vec3<i32>(current_star.high.xyz);
    let packed=current_star.info.z;let centre=vec3<i32>(i32(packed%p.dim),i32((packed/p.dim)%p.dim),i32(packed/(p.dim*p.dim)));
    var radius=i32(current_star.high.w);
    loop {
        let a=max(lo,centre-vec3<i32>(radius));let b=min(hi,centre+vec3<i32>(radius));
        c.x++;if(c.x>b.x){c.x=a.x;c.y++;if(c.y>b.y){c.y=a.y;c.z++;}}
        if(c.z>b.z){
            if(all(a==lo)&&all(b==hi)){return false;}
            radius++;current_star.high.w=u32(radius);c=max(lo,centre-vec3<i32>(radius));
        }
        if(max(max(abs(c.x-centre.x),abs(c.y-centre.y)),abs(c.z-centre.z))>=radius){break;}
    }
    current_star.cell=vec4<u32>(vec3<u32>(c),NONE);current_star.more.x=0u;return true;
}
fn boundary_edges(s:u32)->u32{
    let base=s*p.cap;let mem=32u;var edges=0u;
    for(var i=0u;i<current_star.info.y;i++){
        let f=face_load(s,i);if((f.w&2u)==0u){continue;}
        for(var k=0u;k<3u;k++){
            let a=f[k];let b=f[(k+1u)%3u];var reverse=NONE;
            for(var j=0u;j<edges;j++){if(word(s,mem+2u*j)==b&&word(s,mem+2u*j+1u)==a){reverse=j;break;}}
            if(reverse!=NONE){edges--;set_word(s,mem+2u*reverse,word(s,mem+2u*edges));set_word(s,mem+2u*reverse+1u,word(s,mem+2u*edges+1u));}
            else{if(edges==p.cap){return NONE;}set_word(s,mem+2u*edges,a);set_word(s,mem+2u*edges+1u,b);edges++;}
        }
    }
    return edges;
}
fn horizon(s:u32)->bool {
    let base=s*p.cap;let mem=32u;let n=current_star.info.y;var edges=boundary_edges(s);
    // A large temporary boundary can exceed the final boundary's capacity.
    // Retain the complete quadratic extraction for that uncommon case.
    if(edges==NONE){edges=0u;
    for(var i=0u;i<n;i++){let f=face_load(s,i);if((f.w&2u)==0u){continue;}
        for(var k=0u;k<3u;k++){let a=f[k];let b=f[(k+1u)%3u];var canceled=false;
            for(var j=0u;j<n;j++){let other=face_load(s,j);if((other.w&2u)==0u||j==i){continue;}for(var l=0u;l<3u;l++){if(other[l]==b&&other[(l+1u)%3u]==a){canceled=true;}}}
            if(!canceled){if(edges>=p.cap){fail(s,1u);return false;}set_word(s,mem+2u*edges,a);set_word(s,mem+2u*edges+1u,b);edges++;}
        }
    }
    }
    if(edges<3u){fail(s,2u);return false;}
    // Exactly one simple directed cycle, rather than a best-effort cavity.
    var current=word(s,mem);let start=current;
    for(var step=0u;step<edges;step++){var next=NONE;var degree=0u;for(var j=0u;j<edges;j++){if(word(s,mem+j*2u)==current){next=word(s,mem+j*2u+1u);degree++;}}
        if(degree!=1u||next==NONE||(next==start&&step+1u!=edges)){fail(s,2u);return false;}current=next;
    }
    if(current!=start){fail(s,2u);return false;}
    var keep=0u;for(var i=0u;i<n;i++){if((face_load(s,i).w&2u)==0u){face_copy(s,keep,i);keep++;}}
    if(keep+edges>p.cap){fail(s,1u);return false;}
    current_star.info.y=keep;current_star.low.w=edges;current_star.cursor.z=0u;return true;
}
@compute @workgroup_size(64)
fn star_step(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=p.seeds){return;}current_star=stars[s];cached_seed=current_star.more.z;if(current_star.info.x>=9u||current_star.query.z!=NONE){return;}
    local_star_active=true;run_star(s);stars[s]=current_star;
}
fn run_star(s:u32){
    let seed=current_star.more.z;let base=s*p.cap;let mem=0u;
    for(var work=0u;work<p.budget;work++){
        let phase=current_star.info.x;
        if(phase==1u){ // initialize with nearby halfspaces
            let k=current_star.info.z;if(k>=current_star.info.w){current_star.info.x=2u;continue;}
            current_star.query.x=word(s,mem+k);current_star.query.y=1u;current_star.cursor.y=0u;current_star.cursor.w=0u;current_star.info.x=3u;
        } else if(phase==2u){ // choose an uncertified link face
            var f=NONE;for(var i=0u;i<current_star.info.y;i++){if(face_load(s,i).w==0u){if(p.pad==1u){let count=greater_count(seed,face_load(s,i).xyz);if(count==0u||count==3u){face_mark(s,i,4u);continue;}}f=i;break;}}
            if(f==NONE){
                current_star.info.x=select(7u,9u,p.pad==1u);
                if(FAST_PASS){for(var i=0u;i<current_star.info.y;i++){if((face_load(s,i).w&8u)!=0u){current_star.info.x=12u;break;}}}
                current_star.cursor=vec4<u32>(0);return;
            }
            current_star.cursor.x=f;let v=vec4<u32>(seed,face_load(s,f).xyz);prepare_face_query(s,f);let box=prepared_box(s,v,4e-6);let centre=clamp(cell_of(point(seed)),box.lo,box.hi);current_star.info.z=cell_index(centre);current_star.low=vec4<u32>(box.lo,0u);current_star.high=vec4<u32>(box.hi,0u);current_star.cell=vec4<u32>(centre,NONE);current_star.more.x=0u;current_star.info.x=4u;
        } else if(phase==3u){ // intersect with the candidate's bisector
            if(cooperative_query){return;}
            let i=current_star.cursor.y;
            if(i==current_star.info.y){
                if(current_star.cursor.w==0u){if(current_star.query.y==0u){fail(s,3u);return;}current_star.info.z++;current_star.info.x=1u;continue;}
                current_star.info.x=5u;continue;
            }
            let f=face_load(s,i);var sign=face_filter(s,i,current_star.query.x);
            if(sign==2&&FAST_PASS){
                for(var j=0u;j<current_star.info.y;j++){let old=face_load(s,j);if((old.w&2u)!=0u){face_mark(s,j,old.w&~2u);}}
                if(current_star.query.y!=0u){current_star.info.z++;current_star.info.x=1u;}
                else{face_mark(s,current_star.cursor.x,face_load(s,current_star.cursor.x).w|8u);current_star.info.x=2u;}
                current_star.cursor.w=0u;continue;
            }
            if(sign==2){sign=inside_fast(s,vec4<u32>(seed,f.xyz),current_star.query.x);}if(sign==2){return;}
            if(sign<0){if(f.w==1u){fail(s,4u);return;}face_mark(s,i,f.w|2u);current_star.cursor.w++;}current_star.cursor.y++;
        } else if(phase==4u){ // certify by searching a conservative sphere enclosure
            if(cooperative_query){return;}
            if(current_star.more.x==0u){open_cell(s);}
            let position=current_star.cell.w;
            if(position==NONE){if(!advance_cell(s)){face_mark(s,current_star.cursor.x,1u);current_star.info.x=2u;}continue;}
            let q=query_point(s);let f=face_load(s,current_star.cursor.x);let v=vec4<u32>(seed,f.xyz);var sign=prepared_fast(s,q);if(any(v==vec4<u32>(q))){sign=0;}else if(sign==2&&outside_sphere(s,seed,q)){sign=1;}
            if(sign==2&&FAST_PASS){face_mark(s,current_star.cursor.x,face_load(s,current_star.cursor.x).w|8u);current_star.info.x=2u;continue;}
            if(sign==2){sign=inside_fast(s,v,q);}if(sign==2){return;}
            if(sign<0){current_star.query.x=q;current_star.query.y=0u;current_star.cursor.y=0u;current_star.cursor.w=0u;current_star.info.x=3u;}else{consume_point(s);}
        } else if(phase==5u){
            if(!horizon(s)){return;}current_star.info.x=6u;
        } else if(phase==6u){
            let i=current_star.cursor.z;let edges=current_star.low.w;
            if(i==edges){current_star.more.y++;if(current_star.more.y>p.n*2u+1024u){fail(s,5u);return;}if(current_star.query.y!=0u){current_star.info.z++;current_star.info.x=1u;}else{current_star.info.x=2u;}continue;}
            let a=word(s,mem+32u+2u*i);let b=word(s,mem+32u+2u*i+1u);let q=current_star.query.x;
            if(!FAST_PASS){let sign=predicate_fast(s,vec4<u32>(seed,q,a,b),NONE,0u);if(sign==2){return;}if(sign<=0){fail(s,6u);return;}}
            face_store(s,current_star.info.y,vec4<u32>(q,a,b,0u));current_star.info.y++;current_star.cursor.z++;
        } else if(phase==7u){ // verify the complete oriented link, edge by edge
            if(FAST_PASS){return;}
            let i=current_star.cursor.x;if(i==current_star.info.y){current_star.info.x=select(8u,9u,p.pad==1u);current_star.cursor=vec4<u32>(0);continue;}
            let f=face_load(s,i);for(var k=0u;k<3u;k++){var count=0u;for(var j=0u;j<current_star.info.y;j++){if(i==j){continue;}let g=face_load(s,j);for(var l=0u;l<3u;l++){if(f[k]==g[(l+1u)%3u]&&f[(k+1u)%3u]==g[l]){count++;}}}if(count!=1u){fail(s,7u);return;}}
            current_star.cursor.x++;
        } else if(phase==8u){ // finite enclosing vertices must not truncate the hull
            if(FAST_PASS){return;}
            let i=current_star.cursor.x;if(i==current_star.info.y){current_star.info.x=9u;return;}let f=face_load(s,i);
            var outside=NONE;var real:vec2<u32>;var count=0u;
            for(var k=0u;k<3u;k++){if(f[k]>=p.n){outside=f[k];}else{if(count<2u){real[count]=f[k];}count++;}}
            if(count!=2u||seed>min(real.x,real.y)){current_star.cursor.x++;current_star.cursor.y=0u;continue;}
            let q=current_star.cursor.y;if(q>=p.n){current_star.cursor.x++;current_star.cursor.y=0u;continue;}
            if(q==seed||q==real.x||q==real.y){current_star.cursor.y++;continue;}
            let side=predicate_fast(s,vec4<u32>(seed,real.x,real.y,outside),NONE,0u);if(side==2){return;}
            let sign=predicate_fast(s,vec4<u32>(seed,real.x,real.y,q),NONE,0u);if(sign==2){return;}if(sign!=0&&sign==side){fail(s,8u);return;}current_star.cursor.y++;
        } else {return;}
    }
}
@compute @workgroup_size(128)
fn star_summary(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=p.seeds){return;}let state=stars[s];let phase=state.info.x;
    if(phase==9u){atomicAdd(&counters[0],1u);}else if(phase==10u){atomicAdd(&counters[2],1u);atomicOr(&counters[6],1u<<state.more.z);}else{atomicAdd(&counters[1],1u);}
    atomicAdd(&counters[3],state.more.w);atomicMax(&counters[4],state.info.y);
    if(phase==9u){for(var i=0u;i<state.info.y;i++){let f=face_load(s,i);if(all(f.xyz<vec3<u32>(p.n))&&owns(state.more.z,f.xyz)){atomicAdd(&counters[5],1u);}}}
}
@compute @workgroup_size(128)
fn star_emit(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=p.seeds||stars[s].info.x!=9u){return;}let seed=stars[s].more.z;
    for(var i=0u;i<stars[s].info.y;i++){let f=face_load(s,i);if(all(f.xyz<vec3<u32>(p.n))&&owns(seed,f.xyz)){let at=atomicAdd(&counters[7],1u);tetrahedra[at]=vec4<u32>(seed,f.xyz);}}
}
