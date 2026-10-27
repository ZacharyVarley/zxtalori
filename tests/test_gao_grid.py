"""GPU histogram, scan, scatter, and bounded within-bucket spatial ordering."""
from pathlib import Path
import numpy as np
import wgpu

ROOT=Path(__file__).resolve().parents[1]


def test_gao_grid_preserves_ids_and_orders_small_buckets():
    rng=np.random.default_rng(516)
    points=rng.uniform(-1,1,(1300,4)).astype(np.float32)
    points[:90,:3]=[-.95,-.95,-.95]  # The dense-bucket sort guard is exercised.
    points[90:94,:3]=[[0.,0.,0.],[-0.,0.,0.],[0.,-0.,0.],[0.,0.,-0.]]
    n=len(points);dim=8;cells=dim**3
    params=np.zeros(16,np.uint32);params[:2]=n,dim
    params[4:7]=np.full(3,-1,np.float32).view(np.uint32)
    params[8:11]=np.full(3,1,np.float32).view(np.uint32)
    params[12:15]=np.full(3,4,np.float32).view(np.uint32)
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync()
    usage=wgpu.BufferUsage
    buffers={
        0:device.create_buffer_with_data(data=params,usage=usage.UNIFORM),
        1:device.create_buffer_with_data(data=points,usage=usage.STORAGE),
        2:device.create_buffer(size=(cells+1)*4,usage=usage.STORAGE|usage.COPY_SRC),
        3:device.create_buffer(size=n*16,usage=usage.STORAGE|usage.COPY_SRC),
        4:device.create_buffer(size=(cells+1)*4,usage=usage.STORAGE),  # the prefix also stores the occupied-cell count past the cells
    }
    module=device.create_shader_module(code=(ROOT/'shaders/gao_grid.wgsl').read_text())
    schedule=[('clear',[0,2],(cells+128)//128),('count',[0,1,2],(n+127)//128),('prefix',[0,2,4],1),('scatter',[0,1,3,4],(n+127)//128),('order',[0,2,3],(cells+127)//128)]
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass()
    for entry,slots,groups in schedule:
        pipe=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'gao_grid_'+entry})
        bind=device.create_bind_group(layout=pipe.get_bind_group_layout(0),entries=[{'binding':k,'resource':{'buffer':buffers[k]}} for k in slots])
        work.set_pipeline(pipe);work.set_bind_group(0,bind);work.dispatch_workgroups(groups)
    work.end();device.queue.submit([encoder.finish()])
    offsets=np.frombuffer(device.queue.read_buffer(buffers[2]),np.uint32)
    sorted_bits=np.frombuffer(device.queue.read_buffer(buffers[3]),np.uint32).reshape(n,4)
    ids=sorted_bits[:,3]
    assert np.array_equal(np.sort(ids),np.arange(n))
    assert np.array_equal(sorted_bits[:,:3],points[ids,:3].copy().view(np.uint32))
    cell=np.clip(np.floor((points[:,:3]+1)*4).astype(int),0,dim-1)
    keys=cell[:,0]+dim*(cell[:,1]+dim*cell[:,2])
    expected=np.r_[0,np.cumsum(np.bincount(keys,minlength=cells))]
    assert np.array_equal(offsets,expected)
    dense=0
    for i in range(cells):
        a,b=map(int,offsets[i:i+2]);actual=ids[a:b]
        assert np.all(keys[actual]==i)
        if b-a>64:
            dense+=1
        elif b>a:
            q=points[actual,:3]
            assert np.array_equal(np.lexsort((actual,q[:,2],q[:,1],q[:,0])),np.arange(b-a))
    assert dense==1
    for b in buffers.values():b.destroy()
    device.destroy()
