"""Independent dense f64 checks of the shared WGSL pair kernels on native WebGPU.

Run with .venv/Scripts/python.exe tests/test_pair_kernels.py; the operator
fixture is tests/fixtures/pair-operators.npz.
"""
from pathlib import Path
import json
import math
import numpy as np
import wgpu


def receipt(root, name):
    """Validation receipts stay local, in the ignored research/validation folder."""
    folder = root / "research/validation"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name

ROOT = Path(__file__).resolve().parents[1]


def multiply(a, b):
    return np.concatenate([(a[..., :1]*b[..., :1]-(a[..., 1:]*b[..., 1:]).sum(-1,keepdims=True)),
        a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]+np.cross(a[..., 1:],b[..., 1:])],axis=-1)


def reference(q, ops):
    q=np.asarray(q,dtype=np.float64)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    ops=np.asarray(ops,dtype=np.float64)
    ops=ops/np.linalg.norm(ops,axis=-1,keepdims=True)
    n=len(q)
    out=np.zeros((n,12),np.float64)
    out[:,6]=4
    for k,op in enumerate(ops):
        y=multiply(q,op)
        y/=np.linalg.norm(y,axis=-1,keepdims=True)
        minus=q[:,None,:]-y[None,:,:]
        plus=q[:,None,:]+y[None,:,:]
        dm2=(minus*minus).sum(-1)
        dp2=(plus*plus).sum(-1)
        mask=np.eye(n,dtype=bool) if k==0 else np.zeros((n,n),bool)
        safe_m=np.where(mask,1,dm2)
        dm=np.sqrt(safe_m); dp=np.sqrt(dp2)
        im=1/dm; ip=1/dp
        keep=~mask
        out[:,0]+=np.where(keep,im+ip,0).sum(-1)
        out[:,1]+=np.where(keep,im*im+ip*ip,0).sum(-1)
        out[:,2]+=np.where(keep,im**3+ip**3,0).sum(-1)
        out[:,3]+=np.where(keep,-.5*np.log(safe_m*dp2)+.5,0).sum(-1)
        angle=2*np.arctan2(dm,dp)
        cot=(dp2-safe_m)/(2*dm*dp)
        out[:,4]+=np.where(keep,((np.pi-2*angle)*cot-1)/(4*np.pi**2),0).sum(-1)
        out[:,6]=np.minimum(out[:,6],np.where(mask,4,np.minimum(dm2,dp2)).min(-1))
        dm=np.sqrt(np.maximum(dm2,1e-30)); dp=np.sqrt(np.maximum(dp2,1e-30))
        angle=np.maximum(2*np.arctan2(dm,dp),1e-12)
        anti=np.maximum(2*np.arctan2(dp,dm),1e-12)
        source_keep=~np.eye(n,dtype=bool)
        out[:,5]+=np.where(source_keep,1/angle+1/anti,0).sum(-1)
        mag=1/(angle*angle+1e-9)-1/(anti*anti+1e-9)
        direction=q[:,None,:1]*y[None,:,1:]-y[None,:,:1]*q[:,None,1:]+np.cross(q[:,None,1:],y[None,:,1:])
        force=-mag[...,None]*direction/(.5*dm*dp+1e-9)[...,None]
        out[:,8:11]+=np.where(source_keep[...,None],force,0).sum(1)
        curv=1/(angle**3+1e-9)+1/(anti**3+1e-9)
        out[:,11]+=np.where(source_keep,curv,0).sum(-1)
    out[:,:3]+=np.array([.5,.25,.125])
    out[:,3]+=-np.log(2)+.25
    out[:,4]+=-3/(8*np.pi**2)
    return out


def gpu_rows(device,pipeline,q,ops,force_only=False):
    q=np.ascontiguousarray(q,dtype=np.float32)
    ops=np.ascontiguousarray(ops,dtype=np.float32)
    params=np.array([len(q),len(ops),int(force_only),0,0,0,0,0],np.uint32)
    usage=wgpu.BufferUsage
    p=device.create_buffer_with_data(data=params,usage=usage.UNIFORM)
    qb=device.create_buffer_with_data(data=q,usage=usage.STORAGE)
    ob=device.create_buffer_with_data(data=ops,usage=usage.STORAGE)
    out=device.create_buffer(size=len(q)*48,usage=usage.STORAGE|usage.COPY_SRC)
    binds=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b}} for i,b in enumerate([p,qb,ob,out])])
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass()
    work.set_pipeline(pipeline);work.set_bind_group(0,binds);work.dispatch_workgroups((len(q)+63)//64);work.end();device.queue.submit([encoder.finish()])
    actual=np.frombuffer(device.queue.read_buffer(out),np.float32).reshape(len(q),12).astype(np.float64)
    actual[:,3]+=actual[:,7]
    for b in [p,qb,ob,out]:b.destroy()
    return actual


def gpu_parallel_rows(device,pipeline,reducer,q,ops,batch=1):
    q=np.ascontiguousarray(q,dtype=np.float32);ops=np.ascontiguousarray(ops,dtype=np.float32)
    params=np.array([len(q),len(ops),batch,0,0,0,0,0],np.uint32);usage=wgpu.BufferUsage
    p=device.create_buffer_with_data(data=params,usage=usage.UNIFORM)
    qb=device.create_buffer_with_data(data=q,usage=usage.STORAGE)
    ob=device.create_buffer_with_data(data=ops,usage=usage.STORAGE)
    batches=(len(ops)+batch-1)//batch
    partial=device.create_buffer(size=len(q)*48*batches,usage=usage.STORAGE)
    out=device.create_buffer(size=len(q)*48,usage=usage.STORAGE|usage.COPY_SRC)
    binds=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b}} for i,b in enumerate([p,qb,ob,partial])])
    reduced=device.create_bind_group(layout=reducer.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b}} for i,b in [(0,p),(6,partial),(7,out)]])
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass()
    work.set_pipeline(pipeline);work.set_bind_group(0,binds);work.dispatch_workgroups((len(q)+63)//64,batches)
    work.set_pipeline(reducer);work.set_bind_group(0,reduced);work.dispatch_workgroups((len(q)+63)//64)
    work.end();device.queue.submit([encoder.finish()])
    actual=np.frombuffer(device.queue.read_buffer(out),np.float32).reshape(len(q),12).astype(np.float64)
    for b in [p,qb,ob,partial,out]:b.destroy()
    return actual


def main():
    adapter=wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device=adapter.request_device_sync()
    shader=device.create_shader_module(code=(ROOT/'shaders/pairs.wgsl').read_text())
    pipeline=device.create_compute_pipeline(layout="auto",compute={"module":shader,"entry_point":"pairs"})
    parallel=device.create_compute_pipeline(layout="auto",compute={"module":shader,"entry_point":"quey_partial"})
    reducer=device.create_compute_pipeline(layout="auto",compute={"module":shader,"entry_point":"quey_reduce"})
    fixture=np.load(ROOT/'tests/fixtures/pair-operators.npz')
    rng=np.random.default_rng(752)
    q=rng.normal(size=(37,4));q/=np.linalg.norm(q,axis=-1,keepdims=True);q=q.astype(np.float32)
    errors={}
    for group in fixture.files:
        ops=np.ascontiguousarray(fixture[group],dtype=np.float32)
        actual=gpu_rows(device,pipeline,q,ops)
        expected=reference(q,ops)
        for batch in [1,3,64]:
            quick=gpu_parallel_rows(device,parallel,reducer,q,ops,batch)
            np.testing.assert_allclose(quick[:,[5,6,8,9,10,11]],expected[:,[5,6,8,9,10,11]],rtol=4e-4,atol=2e-4,err_msg=f'{group} parallel batch{batch}')
        for col in [0,1,2,3,4,5,6,8,9,10,11]:
            np.testing.assert_allclose(actual[:,col],expected[:,col],rtol=4e-4,atol=2e-4,err_msg=f"{group} col {col}")
        errors[group]={"max_relative_energy_error":float(np.max(np.abs(actual[:,:6]-expected[:,:6])/np.maximum(np.abs(expected[:,:6]),1))),"max_force_relative_error":float(np.linalg.norm(actual[:,8:11]-expected[:,8:11])/np.linalg.norm(expected[:,8:11])),"max_log_row_absolute_error":float(np.max(np.abs(actual[:,3]-expected[:,3])))}
        # Antipodal representative changes must leave the paired potential and force unchanged.
        signed=q.copy();signed[::2]*=-1
        changed=gpu_rows(device,pipeline,signed,ops)
        np.testing.assert_allclose(changed[:,:7],actual[:,:7],rtol=3e-5,atol=3e-4,err_msg=f'{group} antipodes')
        np.testing.assert_allclose(changed[:,8:],actual[:,8:],rtol=3e-5,atol=3e-4,err_msg=f'{group} antipodal force')
    # Exact duplicates are reported as zero separation; the own-orbit singularity is excluded.
    duplicate=np.stack([q[0],q[0]])
    duplicate_result=gpu_rows(device,pipeline,duplicate,fixture['C2'],True)
    assert (duplicate_result[:,6]==0).all()
    assert np.isfinite(duplicate_result[:,8:]).all()
    singleton=gpu_rows(device,pipeline,q[:1],fixture['C2'],True)
    assert (singleton[:,8:]==0).all()
    parallel_duplicate=gpu_parallel_rows(device,parallel,reducer,duplicate,fixture['C2'])
    assert (parallel_duplicate[:,6]==0).all() and np.isfinite(parallel_duplicate[:,8:]).all()
    parallel_singleton=gpu_parallel_rows(device,parallel,reducer,q[:1],fixture['C2'])
    assert (parallel_singleton[:,8:]==0).all()
    # Validate the force sign, left-tangent frame and factor of two against finite differences.
    base=q[:9].astype(np.float64);base/=np.linalg.norm(base,axis=-1,keepdims=True)
    operators=fixture['C2']
    force=reference(base,operators)[:,8:11]
    finite=[]
    h=1e-6
    for axis in range(3):
        plus=base.copy();minus=base.copy()
        d=np.zeros(4);d[0]=math.cos(h);d[axis+1]=math.sin(h)
        plus[0]=multiply(d,base[0]);d[axis+1]*=-1;minus[0]=multiply(d,base[0])
        derivative=(reference(plus,operators)[:,5].sum()-reference(minus,operators)[:,5].sum())/(2*h)
        np.testing.assert_allclose(derivative,-2*force[0,axis],rtol=2e-6,atol=2e-6)
        finite.append(float(derivative))
    errors['_edge_checks']={'duplicate_separation_zero':True,'singleton_force_zero':True,'antipodal_invariance':True,'parallel_operator_batches':[1,3,64],'parallel_duplicate_singleton':True,'finite_difference_derivatives':finite}
    print(json.dumps(errors,indent=2))
    (receipt(ROOT, 'pair-kernel-validation.json')).write_text(json.dumps(errors,indent=2)+'\n')
    device.destroy()


def test_pair_kernels():
    main()


if __name__=='__main__':main()
