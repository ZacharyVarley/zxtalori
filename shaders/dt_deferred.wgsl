// Cooperative clipping with batched cold predicates. No multi-precision matrix
// or integer determinant is reachable from this entry point.
@compute @workgroup_size(128)
fn cooperative_deferred(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let s=wid.x;
    if(lane==0u){post_kind=select(0u,1u,s<p.seeds&&(stars[s].info.x<9u||stars[s].info.x==12u)&&stars[s].query.z==NONE);}
    workgroupBarrier();if(workgroupUniformLoad(&post_kind)==0u){return;}
    cached_seed=stars[s].more.z;
    for(var i=lane;i<stars[s].info.y;i+=128u){shared_faces[i]=faces[i*p.seeds+s];if(CACHED_FACES){shared_coefficients[i]=bitcast<vec4<f32>>(faces[(p.cap+i)*p.seeds+s]);shared_bounds[i]=bitcast<vec4<f32>>(faces[(2u*p.cap+i)*p.seeds+s]);}}
    for(var i=lane;i<stride();i+=128u){shared_words[i]=scratch[i*p.seeds+s];}
    workgroupBarrier();
    if(lane==0u){for(var i=0u;i<stars[s].info.y;i++){let f=face_load(s,i);if((f.w&8u)!=0u){face_mark(s,i,f.w&~8u);}}if(stars[s].info.x==12u){stars[s].info.x=2u;}}
    storageBarrier();workgroupBarrier();
    for(var iteration=0u;iteration<64u;iteration++){
        if(lane==0u){
            current_star=stars[s];local_star_active=true;cooperative_query=true;
            if(current_star.query.z==NONE){run_star(s);}stars[s]=current_star;
            post_kind=0u;
            if(current_star.info.x<9u&&current_star.query.z==NONE){
                if(current_star.info.x==3u){post_kind=3u;}
                else if(current_star.info.x==4u){post_kind=4u;}
                else{post_kind=1u;}
            }
        }
        storageBarrier();workgroupBarrier();let mode=workgroupUniformLoad(&post_kind);
        if(mode==0u){break;}if(mode==1u){continue;}
        if(mode==3u){
            let begin=stars[s].cursor.y;let count=min(128u,stars[s].info.y-begin);let q=stars[s].query.x;
            if(lane<count){post_signs[lane]=face_filter(s,begin+lane,q);}
            workgroupBarrier();
            if(lane==0u){
                current_star=stars[s];
                for(var j=0u;j<count;j++){
                    let f=face_load(s,begin+j);var sign=post_signs[j];
                    if(sign==2){sign=inside_fast(s,vec4<u32>(seed_id(s),f.xyz),q);}
                    if(sign==2){break;}
                    if(sign<0){if(f.w==1u){fail(s,4u);break;}face_mark(s,begin+j,f.w|2u);current_star.cursor.w++;}
                    current_star.cursor.y++;
                }
                if(current_star.cursor.y==current_star.info.y&&current_star.info.x!=10u){
                    if(current_star.cursor.w!=0u){current_star.info.x=5u;}
                    else if(current_star.query.y!=0u){current_star.info.z++;current_star.info.x=1u;}
                    else{fail(s,3u);}
                }
                stars[s]=current_star;
            }
            storageBarrier();workgroupBarrier();continue;
        }
        // Keep a partially certified candidate block across cold dispatches.
        // Query geometry occupies [32,56); points occupy [64,160), signs [160,256).
        let mem=0u;
        if(lane==0u){
            current_star=stars[s];post_count=current_star.low.w;
            if(post_count==0u){
                let v=vec4<u32>(seed_id(s),face_load(s,current_star.cursor.x).xyz);for(var k=0u;k<4u;k++){set_word(s,mem+60u+k,v[k]);}
                current_star.cursor.z=0u;
                for(var i=0u;i<96u;i++){
                    loop {
                        if(current_star.more.x==0u){open_cell(s);}
                        if(current_star.cell.w!=NONE){break;}
                        if(!advance_cell(s)){current_star.cursor.z=1u;break;}
                    }
                    if(current_star.cursor.z!=0u){break;}
                    let q=query_point(s);set_word(s,mem+64u+post_count,q);post_count++;consume_point(s);
                }
                current_star.low.w=post_count;
                post_exhausted=0u;
            }else{post_exhausted=1u;}
            stars[s]=current_star;
        }
        storageBarrier();workgroupBarrier();let count=workgroupUniformLoad(&post_count);let retained=workgroupUniformLoad(&post_exhausted);
        if(lane<count){
            let q=word(s,mem+64u+lane);post_points[lane]=q;
            if(retained!=0u){post_signs[lane]=bitcast<i32>(word(s,mem+160u+lane));}
            else{
                let v=vec4<u32>(seed_id(s),face_load(s,stars[s].cursor.x).xyz);var sign=0i;
                if(!any(v==vec4<u32>(q))){sign=prepared_fast(s,q);if(sign==2&&outside_sphere(s,v.x,q)){sign=1;}}
                post_signs[lane]=sign;
            }
        }
        workgroupBarrier();
        if(lane==0u){
            current_star=stars[s];let f=face_load(s,current_star.cursor.x);var candidate=NONE;
            for(var j=0u;j<count;j++){if(post_signs[j]<0){candidate=post_points[j];break;}}
            if(candidate==NONE){
                for(var j=0u;j<count;j++){
                    if(post_signs[j]!=2){continue;}
                    let sign=inside_fast(s,vec4<u32>(seed_id(s),f.xyz),post_points[j]);post_signs[j]=sign;
                    if(sign==2){break;}if(sign<0){candidate=post_points[j];break;}
                }
            }
            if(candidate!=NONE){current_star.query.x=candidate;current_star.query.y=0u;current_star.cursor.y=0u;current_star.cursor.w=0u;current_star.low.w=0u;current_star.info.x=3u;}
            else if(current_star.query.z==NONE){
                current_star.low.w=0u;
                if(current_star.cursor.z!=0u){face_mark(s,current_star.cursor.x,1u);current_star.info.x=2u;}
            }
            stars[s]=current_star;
        }
        workgroupBarrier();
        if(lane<count){set_word(s,mem+160u+lane,bitcast<u32>(post_signs[lane]));}
        storageBarrier();workgroupBarrier();
    }
    workgroupBarrier();
    for(var i=lane;i<stride();i+=128u){scratch[i*p.seeds+s]=shared_words[i];}
    for(var i=lane;i<stars[s].info.y;i+=128u){faces[i*p.seeds+s]=shared_faces[i];if(CACHED_FACES){faces[(p.cap+i)*p.seeds+s]=bitcast<vec4<u32>>(shared_coefficients[i]);faces[(2u*p.cap+i)*p.seeds+s]=bitcast<vec4<u32>>(shared_bounds[i]);}}
}
