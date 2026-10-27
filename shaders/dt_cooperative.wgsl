// Gao's difficult-site phase: one workgroup owns a star. The topology state
// remains private to lane zero; candidate sphere tests use all 128 lanes.
var<workgroup> post_kind:u32;
var<workgroup> post_count:u32;
var<workgroup> post_exhausted:u32;
var<workgroup> post_vertices:vec4<u32>;
var<workgroup> post_query:u32;
var<workgroup> post_points:array<u32,128>;
var<workgroup> post_signs:array<i32,128>;
var<workgroup> post_best:atomic<u32>;
var<workgroup> post_uncertain:atomic<u32>;

fn direct_sign(s:u32,v:vec4<u32>,e:u32,kind:u32,lane:u32)->i32{
    if(lane==0u){
        var sign=2i;if(kind==0u){sign=orient_quick(v);}else{sign=sphere_quick(v,e);}
        if(sign==2){sign=double_filter(v,e,kind);}cold_result=sign;
    }
    workgroupBarrier();var sign=workgroupUniformLoad(&cold_result);
    if(sign==2){sign=crt_value(Cache(v,vec4<u32>(e,kind,4u,1u)),lane);if(lane==0u){stars[s].more.w++;}}
    return sign;
}
fn direct_inside(s:u32,v:vec4<u32>,e:u32,lane:u32)->i32{
    if(lane==0u){cold_result=select(1i,0i,any(v==vec4<u32>(e)));}
    workgroupBarrier();if(workgroupUniformLoad(&cold_result)==0){return 0;}
    let raw=direct_sign(s,v,e,1u,lane);if(raw!=0){return raw;}
    var ids=array<u32,5>(v.x,v.y,v.z,v.w,e);
    if(lane==0u){
        cold_result=1;
        for(var i=0u;i<5u;i++){for(var j=i+1u;j<5u;j++){if(all(signed_bits(coordinates[ids[i]].xyz)==signed_bits(coordinates[ids[j]].xyz))){cold_result=0;}}}
    }
    workgroupBarrier();if(workgroupUniformLoad(&cold_result)==0){return 0;}
    for(var rank=0u;rank<5u;rank++){
        var row=0u;for(var j=1u;j<5u;j++){if(ids[j]<ids[row]){row=j;}}
        var other:vec4<u32>;var at=0u;for(var j=0u;j<5u;j++){if(j!=row){other[at]=select(v[j%4u],e,j==4u);at++;}}
        let sign=direct_sign(s,other,NONE,0u,lane);if(sign!=0){return select(sign,-sign,(row&1u)!=0u);}ids[row]=NONE;
    }
    return 0;
}
@compute @workgroup_size(POST_LANES)
fn cooperative_step(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let s=wid.x;
    if(lane==0u){post_kind=select(0u,1u,s<p.seeds&&(stars[s].info.x<9u||stars[s].info.x==12u));}
    workgroupBarrier();if(workgroupUniformLoad(&post_kind)==0u){return;}
    cached_seed=stars[s].more.z;
    for(var i=lane;i<stars[s].info.y;i+=POST_LANES){shared_faces[i]=faces[i*p.seeds+s];if(CACHED_FACES){shared_coefficients[i]=bitcast<vec4<f32>>(faces[(p.cap+i)*p.seeds+s]);shared_bounds[i]=bitcast<vec4<f32>>(faces[(2u*p.cap+i)*p.seeds+s]);}}
    for(var i=lane;i<stride();i+=POST_LANES){shared_words[i]=scratch[i*p.seeds+s];}
    workgroupBarrier();
    if(lane==0u){for(var i=0u;i<stars[s].info.y;i++){let f=face_load(s,i);if((f.w&8u)!=0u){face_mark(s,i,f.w&~8u);}}if(stars[s].info.x==12u){stars[s].info.x=2u;}}
    storageBarrier();workgroupBarrier();
    for(var iteration=0u;iteration<256u;iteration++){
        if(lane==0u){
            current_star=stars[s];local_star_active=true;cooperative_query=true;
            if(current_star.query.z==NONE){run_star(s);}stars[s]=current_star;
            post_kind=0u;
            if(current_star.info.x<9u){
                if(current_star.query.z!=NONE){post_kind=1u;}
                else if(current_star.info.x==4u){post_kind=2u;}
                else if(current_star.info.x==3u){post_kind=4u;}
                else{post_kind=3u;}
            }
        }
        storageBarrier();workgroupBarrier();let mode=workgroupUniformLoad(&post_kind);
        if(mode==0u){break;}
        if(mode==1u){
            if(lane==0u){
                let slot=stars[s].query.z;let c=cache[s*8u+slot];let sign=double_filter(c.ids,c.more.x,c.more.y);
                if(sign!=2){cache[s*8u+slot].more.z=u32(sign+2);stars[s].query.z=NONE;stars[s].more.w--;}
            }
            storageBarrier();workgroupBarrier();crt_pending(s,lane);storageBarrier();workgroupBarrier();continue;
        }
        if(mode==3u){continue;}
        if(mode==4u){
            if(lane==0u){
                post_count=min(POST_LANES,stars[s].info.y-stars[s].cursor.y);post_query=stars[s].query.x;
                atomicStore(&post_uncertain,NONE);
            }
            workgroupBarrier();let count=workgroupUniformLoad(&post_count);let q=workgroupUniformLoad(&post_query);let begin=stars[s].cursor.y;
            if(lane<count){
                let v=vec4<u32>(seed_id(s),face_load(s,begin+lane).xyz);var sign=0i;
                sign=face_filter(s,begin+lane,q);
                post_signs[lane]=sign;if(sign==2){atomicMin(&post_uncertain,lane);}
            }
            workgroupBarrier();
            loop {
                if(lane==0u){post_kind=atomicLoad(&post_uncertain);}
                workgroupBarrier();let at=workgroupUniformLoad(&post_kind);if(at==NONE){break;}
                let v=vec4<u32>(seed_id(s),face_load(s,begin+at).xyz);let sign=direct_inside(s,v,q,lane);
                if(lane==0u){post_signs[at]=sign;atomicStore(&post_uncertain,NONE);for(var j=at+1u;j<count;j++){if(post_signs[j]==2){atomicMin(&post_uncertain,j);}}}
                storageBarrier();workgroupBarrier();
            }
            if(lane==0u){
                current_star=stars[s];
                for(var j=0u;j<count;j++){if(post_signs[j]<0){let f=face_load(s,begin+j);if(f.w==1u){fail(s,4u);break;}face_mark(s,begin+j,f.w|2u);current_star.cursor.w++;}}
                current_star.cursor.y+=count;
                if(current_star.cursor.y==current_star.info.y&&current_star.info.x!=10u){
                    if(current_star.cursor.w!=0u){current_star.info.x=5u;}
                    else if(current_star.query.y!=0u){current_star.info.z++;current_star.info.x=1u;}
                    else{fail(s,3u);}
                }
                stars[s]=current_star;
            }
            storageBarrier();workgroupBarrier();continue;
        }
        if(lane==0u){
            let seed=seed_id(s);let f=face_load(s,stars[s].cursor.x);post_vertices=vec4<u32>(seed,f.xyz);
            let mem=32u;
            if(word(s,mem+23u)==0u){prepare_query(s,post_vertices);refine_query_box(s,post_vertices);}
            current_star=stars[s];post_count=0u;post_exhausted=0u;
            for(var i=0u;i<POST_LANES;i++){
                loop {
                    if(current_star.more.x==0u){open_cell(s);}
                    if(current_star.cell.w!=NONE){break;}
                    if(!advance_cell(s)){post_exhausted=1u;break;}
                }
                if(post_exhausted!=0u){break;}
                let q=query_point(s);post_points[post_count]=q;post_count++;consume_point(s);
            }
            stars[s]=current_star;atomicStore(&post_best,NONE);atomicStore(&post_uncertain,NONE);
        }
        storageBarrier();workgroupBarrier();let count=workgroupUniformLoad(&post_count);let v=workgroupUniformLoad(&post_vertices);
        var sign=0i;if(lane<count){let q=post_points[lane];if(!any(v==vec4<u32>(q))){sign=prepared_fast(s,q);if(sign==2&&outside_sphere(s,v.x,q)){sign=1;}}post_signs[lane]=sign;if(sign<0){atomicMin(&post_best,q);}else if(sign==2){atomicMin(&post_uncertain,lane);}}
        workgroupBarrier();
        loop {
            if(lane==0u){post_kind=select(0u,1u,atomicLoad(&post_best)==NONE&&atomicLoad(&post_uncertain)!=NONE);post_query=atomicLoad(&post_uncertain);}
            workgroupBarrier();if(workgroupUniformLoad(&post_kind)==0u){break;}
            let at=workgroupUniformLoad(&post_query);let q=post_points[at];let exact=direct_inside(s,v,q,lane);
            if(lane==0u){post_signs[at]=exact;if(exact<0){atomicStore(&post_best,q);}atomicStore(&post_uncertain,NONE);for(var j=0u;j<count;j++){if(post_signs[j]==2){atomicMin(&post_uncertain,j);}}}
            storageBarrier();workgroupBarrier();
        }
        if(lane==0u){
            let q=atomicLoad(&post_best);
            if(q!=NONE){stars[s].query.x=q;stars[s].query.y=0u;stars[s].cursor.y=0u;stars[s].cursor.w=0u;stars[s].info.x=3u;}
            else if(post_exhausted!=0u){face_mark(s,stars[s].cursor.x,1u);stars[s].info.x=2u;}
        }
        storageBarrier();workgroupBarrier();
    }
    workgroupBarrier();
    for(var i=lane;i<stride();i+=POST_LANES){scratch[i*p.seeds+s]=shared_words[i];}
    for(var i=lane;i<stars[s].info.y;i+=POST_LANES){faces[i*p.seeds+s]=shared_faces[i];if(CACHED_FACES){faces[(p.cap+i)*p.seeds+s]=bitcast<vec4<u32>>(shared_coefficients[i]);faces[(2u*p.cap+i)*p.seeds+s]=bitcast<vec4<u32>>(shared_bounds[i]);}}
}
