// Scheduler entry points; integer arithmetic lives in dt_integer.wgsl.
@compute @workgroup_size(128)
fn exact_sign(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    crt_pending(wid.x,lane);
}
@compute @workgroup_size(128)
fn adaptive_sign(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let s=wid.x;
    if(lane==0u){
        cold_active=0u;
        if(stars[s].query.z!=NONE){
            let slot=stars[s].query.z;let c=cache[s*8u+slot];var sign=2i;
            if(stars[s].info.x==4u&&c.more.y==1u&&p.pad==1u){let mem=32u;if(word(s,mem+23u)!=1u){prepare_query(s,c.ids);}sign=prepared_sign(s,c.more.x);}
            if(sign==2){sign=double_filter(c.ids,c.more.x,c.more.y);}
            if(sign!=2){cache[s*8u+slot].more.z=u32(sign+2);stars[s].query.z=NONE;stars[s].more.w--;}
            else{cold_active=1u;}
        }
    }
    workgroupBarrier();let pending=workgroupUniformLoad(&cold_active);if(pending==0u){return;}
    crt_pending(s,lane);
}
@compute @workgroup_size(64)
fn double_sign(@builtin(global_invocation_id) id:vec3<u32>){
    let s=id.x;if(s>=p.seeds||stars[s].query.z==NONE){return;}
    let slot=stars[s].query.z;let c=cache[s*8u+slot];var sign=2i;
    if(stars[s].info.x==4u&&c.more.y==1u&&p.pad==1u){let mem=32u;if(word(s,mem+23u)==0u){prepare_query(s,c.ids);refine_query_box(s,c.ids);}if(word(s,mem+23u)==1u){sign=prepared_sign(s,c.more.x);}}
    if(sign==2){sign=double_filter(c.ids,c.more.x,c.more.y);}
    if(sign!=2){cache[s*8u+slot].more.z=u32(sign+2);stars[s].query.z=NONE;stars[s].more.w--;}
}
// Resolve a whole retained search block in parallel. One difficult candidate
// must not force dozens of separate dispatches for ordinary neighbouring sites.
@compute @workgroup_size(128)
fn query_double_sign(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let s=wid.x;if(s>=p.seeds||stars[s].info.x!=4u||stars[s].query.z==NONE||lane>=stars[s].low.w){return;}
    let mem=0u;if(bitcast<i32>(word(s,mem+160u+lane))!=2){return;}
    let v=vec4<u32>(word(s,mem+60u),word(s,mem+61u),word(s,mem+62u),word(s,mem+63u));let q=word(s,mem+64u+lane);
    let sign=double_filter(v,q,1u);if(sign!=2){set_word(s,mem+160u+lane,bitcast<u32>(sign));}
}
fn crt_pending(seed:u32,lane:u32){
    if(lane==0u){cold_active=select(0u,1u,stars[seed].query.z!=NONE);}
    workgroupBarrier();let enabled=workgroupUniformLoad(&cold_active);if(enabled==0u){return;}
    let slot=stars[seed].query.z;let c=cache[seed*8u+slot];
    let sign=crt_value(c,lane);
    if(lane==0u){cache[seed*8u+slot].more.z=u32(sign+2);stars[seed].query.z=NONE;}
}
