"""Check actual compiler/hardware accuracy of double-single log arithmetic."""
from pathlib import Path
import json
import numpy as np
import wgpu


def receipt(root, name):
    """Validation receipts stay local, in the ignored research/validation folder."""
    folder = root / "research/validation"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name

ROOT=Path(__file__).resolve().parents[1]


def main():
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync()
    source=(ROOT/'shaders/pairs.wgsl').read_text()+'''
@compute @workgroup_size(64)
fn log_diagnostic(@builtin(global_invocation_id) gid:vec3<u32>) {
    if(gid.x>=p.n) { return; }
    let v=dlog(quats[gid.x].xy);
    rows[gid.x]=Row(vec4<f32>(v,0.0,0.0),vec4<f32>(0.0),vec4<f32>(0.0));
}
'''
    module=device.create_shader_module(code=source)
    pipeline=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'log_diagnostic'})
    rng=np.random.default_rng(173)
    data=np.zeros((200,4),np.float32)
    data[:,0]=np.exp(rng.uniform(-25,np.log(4),len(data)))
    data[:,1]=data[:,0]*rng.uniform(-3e-8,3e-8,len(data))
    usage=wgpu.BufferUsage
    uniform=device.create_buffer_with_data(data=np.array([len(data),1,0,0,0,0,0,0],np.uint32),usage=usage.UNIFORM)
    inp=device.create_buffer_with_data(data=data,usage=usage.STORAGE)
    out=device.create_buffer(size=len(data)*48,usage=usage.STORAGE|usage.COPY_SRC)
    binds=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{'binding':0,'resource':{'buffer':uniform}},{'binding':1,'resource':{'buffer':inp}},{'binding':3,'resource':{'buffer':out}}])
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass();work.set_pipeline(pipeline);work.set_bind_group(0,binds);work.dispatch_workgroups((len(data)+63)//64);work.end();device.queue.submit([encoder.finish()])
    result=np.frombuffer(device.queue.read_buffer(out),np.float32).reshape(len(data),12).astype(np.float64)
    expected=np.log(data[:,0].astype(np.float64)+data[:,1].astype(np.float64))
    errors=result[:,0]+result[:,1]-expected
    print('double-single log max absolute error:',np.abs(errors).max())
    print('double-single log median absolute error:',np.median(np.abs(errors)))
    print('max low term:',np.abs(result[:,1]).max())
    assert np.abs(result[:,1]).max()>0,'Compiler discarded double-single low terms'
    assert np.abs(errors).max()<5e-12,'Emulated log precision regressed'
    report={'samples':len(data),'maximum_absolute_error':float(np.abs(errors).max()),'median_absolute_error':float(np.median(np.abs(errors))),'maximum_low_term':float(np.abs(result[:,1]).max()),'runtime_integer_rounding_barrier':True,'passed':True}
    (receipt(ROOT, 'log-precision-validation.json')).write_text(json.dumps(report,indent=2)+'\n')
    for b in [uniform,inp,out]:b.destroy()
    device.destroy()


def test_log_precision():
    main()


if __name__=='__main__':main()
