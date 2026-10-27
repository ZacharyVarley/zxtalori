"""Parallel lane-local exact signs and conservative Gao grid-cell rejection."""
from pathlib import Path
import json
import sys
import numpy as np
import pytest
import wgpu
from test_gpu_predicates import primes_table
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from exact_reference import make_cases,reference


@pytest.mark.parametrize('cooperative',[False,True])
def test_gao_parallel_exact_signs_and_cell_rejection(cooperative):
    cases=make_cases()
    structures=[
        [[.125,.25,.5],[.125,.75,.5],[.125,.25,1],[.125,.75,1],[.3,.4,.6]],
        [[.125,.125,.25],[.375,.375,.25],[.125,.125,.75],[.375,.375,.75],[.2,.3,.4]],
        [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],
        [[.125,.25,.5],[-.125,.25,.5],[.125,-.25,.5],[.125,.25,-.5],[-.25,.5,.125]],
        [[0,.5,.125],[.125,.375,.25],[.25,.25,.5],[.375,.125,1],[.5,0,-.5]],
        [[0,0,.5],[.125,0,.375],[0,.25,.25],[.25,.125,.125],[-.125,.5,.125]],
    ]
    for i,q in enumerate(structures):cases.append((f'structure-{i}',np.asarray(q,np.float32).astype(float).tolist(),[0,1,2,3,4]))
    reflection=json.loads((ROOT/'tests/fixtures/gao-reflection-predicates.json').read_text())
    for i,bits in enumerate(reflection['binary32_xyz_bits']):
        cases.append((f'structure-reflection-{i}',np.array(bits,np.uint32).view(np.float32).reshape(5,3).astype(float).tolist(),[0,1,2,3,4]))
    for operation in range(3):
        q=np.array([[.125,.25,.5],[0,0,0],[.375,-.25,1.25],[0,0,0],[.3,.4,.6]],np.float32)
        for a,b in [(0,1),(2,3)]:
            q[b]=q[a]
            if operation==0:q[b,0]*=-1
            else:
                q[b,:2]=q[b,[1,0]]
                if operation==2:q[b,:2]*=-1
        cases.append((f'structure-mirror-{operation}',q.astype(float).tolist(),[0,1,2,3,4]))
    # Candidate positions immediately on both sides of bucket boundaries test
    # the inverse bucket enclosure separately from ordinary sphere predicates.
    rng=np.random.default_rng(7391)
    for i in range(512):
        q=rng.uniform(-8,8,(5,3)).astype(np.float32)
        edge=np.float32(rng.choice(np.arange(-8,9,2)));axis=i%3
        q[4,axis]=np.nextafter(edge,np.float32([-np.inf,np.inf][i%2]))
        cases.append((f'cell-edge-{i}',q.tolist(),[0,1,2,3,4]))
    points=np.zeros((len(cases)*5,4),np.float32);expected=[]
    for i,(_,q,ids) in enumerate(cases):
        points[i*5:i*5+5,:3]=q
        points[i*5:i*5+5,3]=np.asarray(ids,np.uint32).view(np.float32)
        expected.append(reference(q,ids)[0])
    prefix='const GAO_FILTERED_ONLY=false;const GAO_INTEGER_PRIMES=96u;const GAO_POST=false;const GAO_MAX_P=64u;const GAO_MAX_T=128u;const POST_LANES=128u;const NONE=0xffffffffu;struct Cache{ids:vec4<u32>,more:vec4<u32>}\n'
    prefix=prefix.replace('POST_LANES=128u','POST_LANES=32u').replace('GAO_POST=false',f'GAO_POST={str(cooperative).lower()}')
    prefix+='fn gao_record_sign(c:Cache,sign:i32){}\n'
    code=prefix+''.join((ROOT/'shaders'/name).read_text() for name in ['gao_storage.wgsl','gao_topology.wgsl','gao_geometry.wgsl','dt_math.wgsl','dt_double.wgsl','dt_integer.wgsl','gao_predicates.wgsl','gao_double.wgsl'])+'''
@group(0) @binding(6) var<storage,read> prime_table:array<u32>;
@compute @workgroup_size(32)
fn verify(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 if(id.x>=arrayLength(&gao_states)){return;}let base=id.x*5u;gao_seed=base+3u;gao_lane=lane;gao_origin=gao_point(gao_seed);
 let v=vec4<u32>(base,base+1u,base+2u,base+3u);let q=base+4u;
 let a=gao_numeric_sign(v,GAO_NONE,0u);let b=gao_numeric_sign(v,q,1u);let inside=gao_inside_exact(v.xyz,q);
 let sphere=gao_prepare(v.xyz);let outside=gao_cell_outside(sphere,gao_cell(gao_point(q)));
 let double_sphere=gao_prepare_double(v.xyz,sphere);let prepared=gao_prepared_double_sign(double_sphere,q);
 let refined=gao_refined_sphere(double_sphere);let fast=gao_sphere_test(refined,q);
 let refined_outside=gao_cell_outside(refined,gao_cell(gao_point(q)));
 let classified=gao_inside_prepared(v.xyz,q);
 gao_states[id.x]=vec4<u32>(u32(a+2),u32(b+2),u32(inside),u32(outside)|(u32(prepared+2)<<1u)|(u32(fast.x)<<4u)|(u32(fast.y)<<5u)|(u32(refined_outside)<<6u)|(u32(classified)<<7u)|(gao_exact_count<<8u));
}
'''
    params=np.zeros(16,np.uint32);params[:2]=len(points),8
    params[4:7]=np.full(3,-8,np.float32).view(np.uint32);params[8:11]=np.full(3,8,np.float32).view(np.uint32);params[12:15]=np.full(3,.5,np.float32).view(np.uint32)
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync();usage=wgpu.BufferUsage
    module=device.create_shader_module(code=code);pipe=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'verify'})
    buffers={0:device.create_buffer_with_data(data=params,usage=usage.UNIFORM),1:device.create_buffer_with_data(data=points,usage=usage.STORAGE),5:device.create_buffer(size=len(cases)*16,usage=usage.STORAGE|usage.COPY_SRC),6:device.create_buffer_with_data(data=primes_table(),usage=usage.STORAGE)}
    bind=device.create_bind_group(layout=pipe.get_bind_group_layout(0),entries=[{'binding':i,'resource':{'buffer':b}} for i,b in buffers.items()])
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass();work.set_pipeline(pipe);work.set_bind_group(0,bind);work.dispatch_workgroups((len(cases)+31)//32);work.end();device.queue.submit([encoder.finish()])
    rows=np.frombuffer(device.queue.read_buffer(buffers[5]),np.uint32).reshape(-1,4)
    for (name,q,_),row,signs in zip(cases,rows,expected):
        assert tuple(row[:2].astype(int)-2)==tuple(signs[:2]),name
        assert bool(row[2])==(signs[2]<0),name
        if np.all(np.abs(q[4])<=8) and signs[1]<=0:
            assert (row[3]&1)==0,('cell rejection excluded an inside or boundary point',name)
            assert not row[3]&64,('refined cell rejection',name)
        prepared=int((row[3]>>1)&7)-2
        if prepared!=2:assert prepared==signs[1],('prepared double sign',name,prepared,signs[1])
        if row[3]&32:assert bool(row[3]&16)==(signs[1]<0) and signs[1]!=0,('refined filter',name)
        assert bool(row[3]&128)==(signs[2]<0),('prepared classification',name)
        if name.startswith(('structure-reflection-', 'structure-mirror-')):assert row[3]>>8==0,('reflection zero used CRT',name)
    for b in buffers.values():b.destroy()
    device.destroy()
