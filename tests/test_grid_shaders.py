"""Focused T/O/I shader regression against the preserved CPU-f64 h=2 grids."""
from pathlib import Path
import math
import numpy as np
import wgpu

ROOT = Path(__file__).resolve().parents[1]

def generate_gpu(group, h, transform=None):
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    cut = min(range(1, 10*h+5, 2), key=lambda x: (abs(x-(1+math.sqrt(2))*(h+.5)),x))
    bound = 2*h if group == "T" else h
    ii = np.arange(-bound,bound+1,dtype=np.int32)
    index = np.array(np.meshgrid(ii,ii,ii,indexing="ij")).reshape(3,-1).T
    mask = index.sum(1)%2 == 0
    if group == "T": mask &= np.abs(index).sum(1) <= 2*h
    elif group == "O": mask &= np.abs(index).sum(1) < cut
    index = index[mask]
    step = ((math.pi**2/16)**(1/3) if group == "T" else math.pi**(2/3))/(2*h+1)
    source = np.zeros((len(index),4),np.float32)
    source[:,:3] = index.astype(np.float32)*np.float32(step)
    params = np.zeros(8,np.uint32)
    params[:3] = (len(index), {"T":8,"O":9,"I":10}[group],h)
    params.view(np.float32)[4] = cut/(h+.5)-1
    code = (ROOT/"shaders/grid_tables.wgsl").read_text()+ (ROOT/"shaders/grids.wgsl").read_text()
    shader = device.create_shader_module(code=transform(code) if transform else code)
    pipeline = device.create_compute_pipeline(layout="auto",compute={"module":shader,"entry_point":"grid_generate"})
    uniform = device.create_buffer_with_data(data=params,usage=wgpu.BufferUsage.UNIFORM)
    inputs = device.create_buffer_with_data(data=source,usage=wgpu.BufferUsage.STORAGE)
    output = device.create_buffer(size=source.nbytes,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
    bg = device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{"binding":0,"resource":{"buffer":uniform}},{"binding":1,"resource":{"buffer":inputs}},{"binding":2,"resource":{"buffer":output}}])
    encoder = device.create_command_encoder()
    compute = encoder.begin_compute_pass();compute.set_pipeline(pipeline);compute.set_bind_group(0,bg);compute.dispatch_workgroups((len(index)+63)//64);compute.end()
    device.queue.submit([encoder.finish()])
    result = np.frombuffer(device.queue.read_buffer(output),np.float32).reshape(-1,4).copy()
    device.destroy()
    return result

def test_transport_h2():
    ref = np.load(ROOT/"tests/fixtures/reference-small.npz")
    for group in ["T","O","I"]:
        actual = generate_gpu(group,2)
        expected = ref[group]
        assert actual.shape == expected.shape
        assert np.isfinite(actual).all()
        error = np.linalg.norm(actual-expected,axis=1)
        print(group,"max quaternion error",error.max(),"worst",error.argmax(),actual[error.argmax()],expected[error.argmax()])
        np.testing.assert_allclose(np.linalg.norm(actual,axis=1),1,atol=3e-7)
        assert error.max() < 3e-6

def test_transport_h8_h12():
    ref = np.load(ROOT/"tests/fixtures/transports-h8-h12.npz")
    for name in ref.files:
        group,h = name.split("_")
        actual = generate_gpu(group,int(h))
        expected = ref[name]
        assert actual.shape == expected.shape
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(np.linalg.norm(actual,axis=1),1,atol=3e-7)
        assert np.linalg.norm(actual-expected,axis=1).max() < 3e-6

if __name__ == "__main__":
    test_transport_h2()
    test_transport_h8_h12()
