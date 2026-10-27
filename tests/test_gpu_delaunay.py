"""Exercise the actual WGSL star scheduler; Qhull is only an independent oracle."""
from pathlib import Path
from collections import defaultdict
import sys,time,json
import numpy as np
import wgpu
from scipy.spatial import Delaunay,ConvexHull
from test_gpu_predicates import primes_table
ROOT=Path(__file__).resolve().parents[1]

def triangulate_gpu(points,*,capacity=128,verbose=False,local=False,approximate=False,cached=False):
    start=time.perf_counter();points=np.asarray(points,np.float32);n=len(points);bound=max(1,float(np.max(np.abs(points))));B=bound*2**20
    coords=np.zeros((n+4,4),np.float32);coords[:n,:3]=points;coords[n:,:3]=np.array([[-B,-B,-B],[B,B,-B],[B,-B,B],[-B,B,B]])
    dim=max(1,round((n/4)**(1/3)));lo=points.min(axis=0);width=max(float(np.ptp(points,axis=0).max()),1e-5)*1.00001;scale=dim/width
    params=np.zeros(16,np.uint32);params[:8]=[n,n,0,capacity,dim,0,256,0];params[8:12]=np.array([*lo,0],np.float32).view(np.uint32);params[12:16]=np.array([scale,0,0,0],np.float32).view(np.uint32)
    params[7]=int(local)
    params[13:16]=points.max(axis=0).view(np.uint32)
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync(required_limits={'max-storage-buffers-per-shader-stage':8})
    source='const POST_LANES=128u; const FAST_PASS=false; const POST_INLINE=true; const SHARED_FACES=false; const FACE_CAP=128u; const CACHED_FACES=true;\n'+''.join((ROOT/'shaders'/name).read_text() for name in ['dt_common.wgsl','dt_math.wgsl','dt_double.wgsl','dt_interval.wgsl','dt_scratch.wgsl','dt_prepared.wgsl','dt_star.wgsl','dt_integer.wgsl','dt_exact.wgsl','dt_cooperative.wgsl'])
    source=source.replace('const CACHED_FACES=true;',f'const CACHED_FACES={str(cached).lower()};');module=device.create_shader_module(code=source.replace('const FAST_PASS=false;',f'const FAST_PASS={str(approximate).lower()};'));post_module=device.create_shader_module(code=source.replace('const SHARED_FACES=false;','const SHARED_FACES=true;').replace('const POST_LANES=128u;','const POST_LANES=32u;'));names=['grid_clear','grid_build','grid_order','star_init','star_step','double_sign','exact_sign','cooperative_step','star_summary','star_emit'];pipes={name:device.create_compute_pipeline(layout='auto',compute={'module':post_module if name=='cooperative_step' else module,'entry_point':name}) for name in names}
    u=wgpu.BufferUsage
    def buf(size):return device.create_buffer(size=max(16,size),usage=u.STORAGE|u.COPY_SRC|u.COPY_DST)
    buffers={0:device.create_buffer_with_data(data=params,usage=u.UNIFORM),1:device.create_buffer_with_data(data=coords,usage=u.STORAGE),2:buf(dim**3*8),3:buf((n+2*dim**3)*4),4:buf(n*capacity*48),5:buf(n*112),6:buf(n*8*32),7:buf(n*(32+capacity*2)*4),8:device.create_buffer_with_data(data=primes_table(),usage=u.STORAGE),9:buf(32),10:buf(max(100,n*32)*16),11:buf(n*4)}
    slots={'grid_clear':[0,2],'grid_build':[0,1,2,3],'grid_order':[0,2,3,11],'star_init':[0,1,3,4,5,6,7,11],'star_step':[0,1,3,4,5,6,7],'double_sign':[0,1,5,6,7],'exact_sign':[1,5,6,8],'cooperative_step':[0,1,3,4,5,6,7,8],'star_summary':[0,1,4,5,9],'star_emit':[0,1,4,5,9,10]}
    binds={name:device.create_bind_group(layout=pipes[name].get_bind_group_layout(0),entries=[{'binding':k,'resource':{'buffer':buffers[k]}} for k in slots[name]]) for name in names}
    def run(entries):
        encoder=device.create_command_encoder();work=encoder.begin_compute_pass()
        for name,count in entries:work.set_pipeline(pipes[name]);work.set_bind_group(0,binds[name]);work.dispatch_workgroups(count)
        work.end();device.queue.submit([encoder.finish()])
    run([('grid_clear',(dim**3+127)//128),('grid_build',(n+127)//128),('grid_order',1),('star_init',(n+63)//64)])
    for block in range(1000):
        if local and block:run([('cooperative_step',n)])
        else:run([item for _ in range(8) for item in [('star_step',(n+63)//64),('double_sign',(n+63)//64),('exact_sign',n)]])
        device.queue.write_buffer(buffers[9],0,np.zeros(8,np.uint32));run([('star_summary',(n+127)//128)])
        stats=np.frombuffer(device.queue.read_buffer(buffers[9]),np.uint32).copy()
        if verbose and (block%20==0 or stats[1]==0 or stats[2]):print('block',block,stats.tolist(),flush=True)
        if stats[2] or stats[1]==0:break
    states=np.frombuffer(device.queue.read_buffer(buffers[5]),np.uint32).reshape(n,28).copy()
    if stats[2] or stats[1]:
        bad=np.flatnonzero(states[:,0]!=9);print('bad seeds',[(int(i),states[i].tolist()) for i in bad[:10]],flush=True)
        raise AssertionError(stats)
    count=int(stats[5]);run([('star_emit',(n+127)//128)]);tets=np.frombuffer(device.queue.read_buffer(buffers[10],0,count*16),np.uint32).reshape(count,4).copy()
    if verbose:print('GPU DT',n,'sites',len(tets),'tets',time.perf_counter()-start,'seconds',flush=True)
    for b in buffers.values():b.destroy()
    device.destroy();return tets

def check(points,tets,compare=True):
    p=points.astype(float);v=p[tets];volume=np.linalg.det(v[:,1:]-v[:,:1])/6;assert (volume>0).all();assert abs(volume.sum()-ConvexHull(p).volume)<1e-8
    assert len(set(map(tuple,np.sort(tets,axis=1))))==len(tets)
    assert set(np.unique(tets))==set(range(len(p)))
    incidence=defaultdict(list)
    for tet in tets:
        for opposite in range(4):
            face=tuple(int(tet[k]) for k in range(4) if k!=opposite)
            parity=sum(face[i]>face[j] for i in range(3) for j in range(i+1,3))
            incidence[tuple(sorted(face))].append((-1)**(opposite+parity))
    boundary=[]
    for face,signs in incidence.items():
        assert len(signs)<=2
        if len(signs)==2:assert sum(signs)==0
        else:boundary.append(face)
    for face in boundary:
        a,b,c=p[list(face)];side=(p-a)@np.cross(b-a,c-a)
        assert side.max()<1e-8 or side.min()>-1e-8
    if compare:assert set(map(tuple,np.sort(tets,axis=1)))==set(map(tuple,np.sort(Delaunay(p).simplices,axis=1)))
    c=np.linalg.solve(2*(v[:,1:]-v[:,:1]),(np.sum(v[:,1:]**2,axis=2)-np.sum(v[:,:1]**2,axis=2))[...,None])[...,0];r=np.sum((v[:,0]-c)**2,axis=1)
    for i in range(0,len(c),128):assert np.min(np.sum((c[i:i+128,None,:]-p[None,:,:])**2,axis=2)-r[i:i+128,None])>-1e-8

def test_gpu_stars():
    rng=np.random.default_rng(193);points=rng.normal(size=(64,3)).astype(np.float32);check(points,triangulate_gpu(points,verbose=True))
    with np.load(ROOT/'tests/fixtures/reference-small.npz') as data:points=data['dt_fcc_points'].astype(np.float32)
    check(points,triangulate_gpu(points,verbose=True),False)


def test_gpu_larger_random_and_degenerate_fcc():
    points=np.random.default_rng(19).normal(size=(1000,3)).astype(np.float32)
    check(points,triangulate_gpu(points))
    points=np.array([(i,j,k) for i in range(-3,4) for j in range(-3,4) for k in range(-3,4) if (i+j+k)%2==0],np.float32)
    check(points,triangulate_gpu(points),False)
def test_gpu_gao_fast_pass_and_cooperative_completion():
    points=np.random.default_rng(19).normal(size=(1000,3)).astype(np.float32)
    check(points,triangulate_gpu(points,local=True,approximate=True))
    points=np.array([(i,j,k) for i in range(-3,4) for j in range(-3,4) for k in range(-3,4) if (i+j+k)%2==0],np.float32)
    check(points,triangulate_gpu(points,local=True,approximate=True),False)

if __name__=='__main__':test_gpu_stars()
