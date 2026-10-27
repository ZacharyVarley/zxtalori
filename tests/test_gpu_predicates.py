"""Actual integer-only WGSL exact signs, checked against Fraction determinants."""
from pathlib import Path
import sys,json,time,math
import numpy as np
import wgpu
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from exact_reference import make_cases,reference


def receipt(root, name):
    """Validation receipts stay local, in the ignored research/validation folder."""
    folder = root / "research/validation"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name

def primes_table():
    primes=[]
    for n in range(65521,32768,-2):
        if all(n%d for d in range(3,math.isqrt(n)+1,2)):
            primes.append(n)
            if len(primes)==96:break
    assert math.prod(primes).bit_length()>1400
    table=np.zeros(96+96*96+96*256,np.uint32);table[:96]=primes
    for j in range(96):
        for i in range(j+1,96):table[96+j*96+i]=pow(primes[j],primes[i]-2,primes[i])
    for i,prime in enumerate(primes):
        value=1
        for j in range(256):table[9312+i*256+j]=value;value=value*2%prime
    return table

def test_gpu_exact_predicates():
    cases=make_cases();points=np.zeros((len(cases)*5,4),np.float32);count=len(cases)*2
    states=np.zeros((count,28),np.uint32);cache=np.zeros((count,8,8),np.uint32);expected=[]
    for i,(_,q,ids) in enumerate(cases):
        points[i*5:i*5+5,:3]=q
        signs,*_=reference(q,ids);expected.extend(signs[:2])
        for kind in [0,1]:
            seed=i*2+kind;cache[seed,0,:4]=np.arange(i*5,i*5+4);cache[seed,0,4:]=[i*5+4,kind,4,1]
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync()
    code='const POST_LANES=128u; const FAST_PASS=false; const POST_INLINE=true; const SHARED_FACES=false; const FACE_CAP=128u; const CACHED_FACES=true;\n'+(ROOT/'shaders/dt_common.wgsl').read_text()+(ROOT/'shaders/dt_math.wgsl').read_text()+(ROOT/'shaders/dt_double.wgsl').read_text()+(ROOT/'shaders/dt_interval.wgsl').read_text()+(ROOT/'shaders/dt_scratch.wgsl').read_text()+(ROOT/'shaders/dt_prepared.wgsl').read_text()+(ROOT/'shaders/dt_integer.wgsl').read_text()+(ROOT/'shaders/dt_exact.wgsl').read_text()
    start=time.perf_counter();module=device.create_shader_module(code=code);pipeline=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'exact_sign'})
    u=wgpu.BufferUsage;buffers={1:device.create_buffer_with_data(data=points,usage=u.STORAGE),5:device.create_buffer_with_data(data=states,usage=u.STORAGE),6:device.create_buffer_with_data(data=cache,usage=u.STORAGE|u.COPY_SRC),8:device.create_buffer_with_data(data=primes_table(),usage=u.STORAGE)}
    bind=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{'binding':k,'resource':{'buffer':b}} for k,b in buffers.items()])
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass();work.set_pipeline(pipeline);work.set_bind_group(0,bind);work.dispatch_workgroups(count);work.end();device.queue.submit([encoder.finish()]);actual=np.frombuffer(device.queue.read_buffer(buffers[6]),np.uint32).reshape(count,8,8)[:,0,6].astype(int)-2
    wrong=np.flatnonzero(actual!=expected)
    for i in wrong[:10]:print(i,cases[i//2][0],actual[i],expected[i])
    assert not len(wrong)
    print(len(cases),'cases,',count,'GPU exact orientation/insphere signs passed;',time.perf_counter()-start,'seconds including compile')
    receipt(ROOT, 'gpu-predicate-validation.json').write_text(json.dumps({'cases':len(cases),'signs':count,'all_finite_binary32':True,'integer_only_gpu':True,'passed':True},indent=2)+'\n')
    for b in buffers.values():b.destroy()
    device.destroy()
def test_gpu_filtered_and_sos_predicates():
    cases=make_cases();count=len(cases)*3;points=np.zeros((len(cases)*5,4),np.float32);states=np.zeros((count,28),np.uint32);states[:,10]=0xffffffff;calls=np.zeros((count,6),np.uint32);expected=[]
    for i,(_,q,ids) in enumerate(cases):
        order=np.argsort(ids);points[i*5:i*5+5,:3]=np.array(q)[order];mapping={int(old):i*5+new for new,old in enumerate(order)}
        signs,*_=reference(q,ids);expected.extend(signs)
        for kind in range(3):calls[i*3+kind]=[*[mapping[k] for k in range(5)],kind]
    code='const POST_LANES=128u; const FAST_PASS=false; const POST_INLINE=true; const SHARED_FACES=false; const FACE_CAP=128u; const CACHED_FACES=true;\n'+(ROOT/'shaders/dt_common.wgsl').read_text()+(ROOT/'shaders/dt_math.wgsl').read_text()+(ROOT/'shaders/dt_double.wgsl').read_text()+(ROOT/'shaders/dt_interval.wgsl').read_text()+(ROOT/'shaders/dt_scratch.wgsl').read_text()+(ROOT/'shaders/dt_prepared.wgsl').read_text()+(ROOT/'shaders/dt_integer.wgsl').read_text()+(ROOT/'shaders/dt_exact.wgsl').read_text()+'''
@compute @workgroup_size(64)
fn predicate_check(@builtin(global_invocation_id) id:vec3<u32>){
 let s=id.x;if(s>=p.seeds||stars[s].query.z!=NONE){return;}let b=s*6u;let v=vec4<u32>(scratch[b],scratch[b+1u],scratch[b+2u],scratch[b+3u]);let e=scratch[b+4u];let kind=scratch[b+5u];var result=0i;
 if(kind==2u){result=inside(s,v,e);}else{result=predicate(s,v,e,kind);}if(result!=2){atomicStore(&counters[s],u32(result+2));}
}
'''
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync();module=device.create_shader_module(code=code);pipes={k:device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':k}) for k in ['predicate_check','exact_sign']};u=wgpu.BufferUsage
    params=np.zeros(16,np.uint32);params[1]=count
    buffers={0:device.create_buffer_with_data(data=params,usage=u.UNIFORM),1:device.create_buffer_with_data(data=points,usage=u.STORAGE),5:device.create_buffer_with_data(data=states,usage=u.STORAGE),6:device.create_buffer(size=count*8*32,usage=u.STORAGE),7:device.create_buffer_with_data(data=calls,usage=u.STORAGE),8:device.create_buffer_with_data(data=primes_table(),usage=u.STORAGE),9:device.create_buffer(size=count*4,usage=u.STORAGE|u.COPY_SRC)}
    slots={'predicate_check':[0,1,5,6,7,9],'exact_sign':[1,5,6,8]};binds={k:device.create_bind_group(layout=pipes[k].get_bind_group_layout(0),entries=[{'binding':v,'resource':{'buffer':buffers[v]}} for v in slots[k]]) for k in pipes}
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass()
    for _ in range(16):
        for name,groups in [('predicate_check',(count+63)//64),('exact_sign',count)]:work.set_pipeline(pipes[name]);work.set_bind_group(0,binds[name]);work.dispatch_workgroups(groups)
    work.end();device.queue.submit([encoder.finish()]);actual=np.frombuffer(device.queue.read_buffer(buffers[9]),np.uint32).astype(int)-2
    wrong=np.flatnonzero(actual!=expected)
    for i in wrong[:10]:print('FILTER/SOS',i,cases[i//3][0],actual[i],expected[i])
    assert not len(wrong)
    print(count,'adaptive GPU orientation/insphere/SoS results match Fraction.')
    path=receipt(ROOT, 'gpu-predicate-validation.json')
    report=json.loads(path.read_text()) if path.exists() else {}
    report.update({'adaptive_signs':count,'adaptive_modulus':True,'symbolic_ties':True,'passed':True})
    path.write_text(json.dumps(report,indent=2)+'\n')
    for b in buffers.values():b.destroy()
    device.destroy()
def test_gpu_prepared_sphere_filter():
    cases=make_cases();count=len(cases);points=np.zeros((count*5,4),np.float32);expected=[]
    for i,(_,q,ids) in enumerate(cases):
        points[i*5:i*5+5,:3]=q;expected.append(reference(q,ids)[0][1])
    code='const POST_LANES=128u; const FAST_PASS=false; const POST_INLINE=true; const SHARED_FACES=false; const FACE_CAP=128u; const CACHED_FACES=true;\n'+''.join((ROOT/'shaders'/name).read_text() for name in ['dt_common.wgsl','dt_math.wgsl','dt_double.wgsl','dt_interval.wgsl','dt_scratch.wgsl','dt_prepared.wgsl'])+'''
@compute @workgroup_size(64)
fn prepared_check(@builtin(global_invocation_id) id:vec3<u32>){
 let s=id.x;if(s>=p.seeds){return;}let b=s*5u;
 prepare_query(s,vec4<u32>(b,b+1u,b+2u,b+3u));atomicStore(&counters[s],u32(prepared_sign(s,b+4u)+2));
}
'''
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync()
    module=device.create_shader_module(code=code);pipe=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'prepared_check'});u=wgpu.BufferUsage
    params=np.zeros(16,np.uint32);params[1]=count;params[3]=128
    buffers={0:device.create_buffer_with_data(data=params,usage=u.UNIFORM),1:device.create_buffer_with_data(data=points,usage=u.STORAGE),7:device.create_buffer(size=count*288*4,usage=u.STORAGE),9:device.create_buffer(size=count*4,usage=u.STORAGE|u.COPY_SRC)}
    bind=device.create_bind_group(layout=pipe.get_bind_group_layout(0),entries=[{'binding':k,'resource':{'buffer':v}} for k,v in buffers.items()])
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass();work.set_pipeline(pipe);work.set_bind_group(0,bind);work.dispatch_workgroups((count+63)//64);work.end();device.queue.submit([encoder.finish()])
    actual=np.frombuffer(device.queue.read_buffer(buffers[9]),np.uint32).astype(int)-2
    wrong=np.flatnonzero((actual!=2)&(actual!=expected))
    for i in wrong[:10]:print('PREPARED',i,cases[i][0],actual[i],expected[i])
    assert not len(wrong);assert np.count_nonzero(actual!=2)>25
    print('Prepared sphere filter:',np.count_nonzero(actual!=2),'resolved signs match Fraction;',count,'cases checked.')
    for b in buffers.values():b.destroy()
    device.destroy()

def test_gpu_cached_face_and_bounded_filters():
    cases=make_cases();count=len(cases);points=np.zeros((count*5,4),np.float32);expected=[]
    for i,(_,q,ids) in enumerate(cases):
        points[i*5:i*5+5,:3]=q;signs=reference(q,ids)[0];expected.extend([signs[1],signs[1],signs[0]])
    code='const POST_LANES=128u; const FAST_PASS=false; const POST_INLINE=false; const SHARED_FACES=false; const FACE_CAP=128u; const CACHED_FACES=true;\n'+''.join((ROOT/'shaders'/name).read_text() for name in ['dt_common.wgsl','dt_math.wgsl','dt_double.wgsl','dt_interval.wgsl','dt_scratch.wgsl','dt_prepared.wgsl','dt_star.wgsl'])+"""
@compute @workgroup_size(64)
fn face_check(@builtin(global_invocation_id) id:vec3<u32>){
 let s=id.x;if(s>=p.seeds){return;}let b=s*5u;let v=vec4<u32>(b,b+1u,b+2u,b+3u);
 current_star.more.z=b;cached_seed=b;stars[s].more.z=b;face_store(s,0u,vec4<u32>(v.yzw,0u));
 atomicStore(&counters[s*3u],u32(face_filter(s,0u,b+4u)+2));
 atomicStore(&counters[s*3u+1u],u32(sphere_quick(v,b+4u)+2));
 atomicStore(&counters[s*3u+2u],u32(orient_quick(v)+2));
}
"""
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync()
    module=device.create_shader_module(code=code);pipe=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'face_check'});u=wgpu.BufferUsage
    params=np.zeros(16,np.uint32);params[1]=count;params[3]=128
    buffers={0:device.create_buffer_with_data(data=params,usage=u.UNIFORM),1:device.create_buffer_with_data(data=points,usage=u.STORAGE),4:device.create_buffer(size=count*128*48,usage=u.STORAGE),5:device.create_buffer(size=count*112,usage=u.STORAGE),9:device.create_buffer(size=count*3*4,usage=u.STORAGE|u.COPY_SRC)}
    bind=device.create_bind_group(layout=pipe.get_bind_group_layout(0),entries=[{'binding':k,'resource':{'buffer':v}} for k,v in buffers.items()])
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass();work.set_pipeline(pipe);work.set_bind_group(0,bind);work.dispatch_workgroups((count+63)//64);work.end();device.queue.submit([encoder.finish()])
    actual=np.frombuffer(device.queue.read_buffer(buffers[9]),np.uint32).astype(int)-2;wrong=np.flatnonzero((actual!=2)&(actual!=expected))
    for i in wrong[:10]:print('CACHED',i,cases[i//3][0],actual[i],expected[i])
    assert not len(wrong);assert np.count_nonzero(actual!=2)>50
    for b in buffers.values():b.destroy()
    device.destroy()

def test_gpu_cooperative_sos_predicates():
    cases=make_cases()+[('fcc_query_tie',[[-1.,-1.,-2.],[-1.,1.,-2.],[-2.,1.,-3.],[-2.,1.,-1.],[-2.,-1.,-1.]],[56,63,38,39,32])]
    count=len(cases);points=np.zeros((count*5,4),np.float32);calls=np.zeros((count,5),np.uint32);expected=[]
    for i,(_,q,ids) in enumerate(cases):
        order=np.argsort(ids);points[i*5:i*5+5,:3]=np.array(q)[order];mapping={int(old):i*5+new for new,old in enumerate(order)}
        calls[i]=[mapping[k] for k in range(5)];expected.append(reference(q,ids)[0][2])
    code='const POST_LANES=32u; const FAST_PASS=false; const POST_INLINE=false; const SHARED_FACES=false; const FACE_CAP=128u; const CACHED_FACES=false;\n'+''.join((ROOT/'shaders'/name).read_text() for name in ['dt_common.wgsl','dt_math.wgsl','dt_double.wgsl','dt_interval.wgsl','dt_scratch.wgsl','dt_prepared.wgsl','dt_star.wgsl','dt_integer.wgsl','dt_exact.wgsl','dt_cooperative.wgsl'])+"""
@compute @workgroup_size(32)
fn cooperative_check(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let s=wid.x;let b=s*5u;let v=vec4<u32>(scratch[b],scratch[b+1u],scratch[b+2u],scratch[b+3u]);let e=scratch[b+4u];
 let sign=direct_inside(s,v,e,lane);if(lane==0u){atomicStore(&counters[s],u32(sign+2));}
}
"""
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync();module=device.create_shader_module(code=code);pipe=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'cooperative_check'});u=wgpu.BufferUsage
    params=np.zeros(16,np.uint32);params[1]=count;params[3]=128
    buffers={0:device.create_buffer_with_data(data=params,usage=u.UNIFORM),1:device.create_buffer_with_data(data=points,usage=u.STORAGE),5:device.create_buffer(size=count*112,usage=u.STORAGE),7:device.create_buffer_with_data(data=calls,usage=u.STORAGE),8:device.create_buffer_with_data(data=primes_table(),usage=u.STORAGE),9:device.create_buffer(size=count*4,usage=u.STORAGE|u.COPY_SRC)}
    bind=device.create_bind_group(layout=pipe.get_bind_group_layout(0),entries=[{'binding':k,'resource':{'buffer':v}} for k,v in buffers.items()])
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass();work.set_pipeline(pipe);work.set_bind_group(0,bind);work.dispatch_workgroups(count);work.end();device.queue.submit([encoder.finish()])
    actual=np.frombuffer(device.queue.read_buffer(buffers[9]),np.uint32).astype(int)-2;wrong=np.flatnonzero(actual!=expected)
    for i in wrong[:10]:print('COOPERATIVE',i,cases[i][0],actual[i],expected[i])
    assert not len(wrong)
    for b in buffers.values():b.destroy()
    device.destroy()

if __name__=='__main__':
    test_gpu_exact_predicates()
    test_gpu_filtered_and_sos_predicates()
    test_gpu_prepared_sphere_filter()


