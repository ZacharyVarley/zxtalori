"""Source-guided Gao main pass: completed stars and explicit deferred meshes.

CPU bucket packing here is fixture preparation, not the production scheduler.
Qhull validates each completed star independently; no CPU DT fills its output.
"""
from pathlib import Path
import numpy as np
import pytest
import wgpu
from scipy.spatial import Delaunay

ROOT=Path(__file__).resolve().parents[1]


def main_pass(points,*,post=False,post_lanes=32,grow=False,filtered=True):
    points=np.asarray(points,np.float32);n=len(points)
    dim=max(1,round((n/4)**(1/3)))
    lower=points.min(axis=0);upper=points.max(axis=0)
    scale=np.full(3,dim/max(float(np.ptp(points,axis=0).max()),1e-5),np.float32)
    cell=np.clip(np.floor((points-lower)*scale).astype(int),0,dim-1)
    key=cell[:,0]+dim*(cell[:,1]+dim*cell[:,2]);order=np.argsort(key,kind='stable')
    offsets=np.r_[0,np.cumsum(np.bincount(key,minlength=dim**3))].astype(np.uint32)
    coords=np.zeros((n+4,4),np.float32);coords[:n,:3]=points[order]
    bound=np.float32(max(1,float(np.max(np.abs(points))))*1e8)
    coords[n:,:3]=[[-bound,-bound,-bound],[bound,-bound,-bound],[0,bound,-bound],[0,0,bound]]
    coords[:,3]=np.r_[order,np.arange(n,n+4)].astype(np.uint32).view(np.float32)
    params=np.zeros(16,np.uint32);params[:2]=n,dim;params[11]=n  # every site is a seed
    params[4:7]=lower.view(np.uint32);params[8:11]=upper.view(np.uint32);params[12:15]=scale.view(np.uint32)
    code='const GAO_POST=false;const GAO_MAX_P=32u;const GAO_MAX_T=64u;\n'+''.join((ROOT/'shaders'/name).read_text() for name in ['gao_storage.wgsl','gao_topology.wgsl','gao_geometry.wgsl','gao_main.wgsl'])
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync()
    module=device.create_shader_module(code=code)
    pipeline=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'gao_main'})
    usage=wgpu.BufferUsage
    buffers=[device.create_buffer_with_data(data=params,usage=usage.UNIFORM),device.create_buffer_with_data(data=coords,usage=usage.STORAGE),device.create_buffer_with_data(data=offsets,usage=usage.STORAGE)]
    for size in [n*128*4,n*64*4,n*16]:buffers.append(device.create_buffer(size=size,usage=usage.STORAGE|usage.COPY_SRC|usage.COPY_DST))
    retired=[]
    bind=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{'binding':i,'resource':{'buffer':b}} for i,b in enumerate(buffers)])
    encoder=device.create_command_encoder();work=encoder.begin_compute_pass();work.set_pipeline(pipeline);work.set_bind_group(0,bind);work.dispatch_workgroups((n+127)//128);work.end();device.queue.submit([encoder.finish()])
    if post:
        from test_gpu_predicates import primes_table
        buffers.append(device.create_buffer_with_data(data=primes_table(),usage=usage.STORAGE))
        buffers.append(device.create_buffer(size=32,usage=usage.STORAGE|usage.COPY_SRC))
        buffers.append(device.create_buffer(size=64*32,usage=usage.STORAGE|usage.COPY_SRC))
        def dispatch_post(capacity,lanes,prime_count,filter_only=False):
            prefix=f'const GAO_FILTERED_ONLY=false;const GAO_INTEGER_PRIMES={prime_count}u;const GAO_POST=true;const GAO_MAX_P={capacity}u;const GAO_MAX_T={capacity*2}u;const POST_LANES={lanes}u;const NONE=0xffffffffu;struct Cache{{ids:vec4<u32>,more:vec4<u32>}}\n'
            if filter_only:prefix=prefix.replace('GAO_FILTERED_ONLY=false','GAO_FILTERED_ONLY=true')
            source=prefix+''.join((ROOT/'shaders'/name).read_text() for name in ['gao_storage.wgsl','gao_topology.wgsl','gao_geometry.wgsl','dt_math.wgsl','dt_double.wgsl','dt_integer.wgsl','gao_predicates.wgsl','gao_double.wgsl','gao_post.wgsl'])
            module=device.create_shader_module(code=source)
            pipeline=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'gao_post'})
            bind=device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[{'binding':i,'resource':{'buffer':b}} for i,b in enumerate(buffers)])
            encoder=device.create_command_encoder();work=encoder.begin_compute_pass();work.set_pipeline(pipeline);work.set_bind_group(0,bind);work.dispatch_workgroups(n);work.end();device.queue.submit([encoder.finish()])
        if filtered:dispatch_post(64,post_lanes,32,True)
        dispatch_post(64,post_lanes,32)
        if grow:
            for capacity in [128,256]:
                states=np.frombuffer(device.queue.read_buffer(buffers[5]),np.uint32).reshape(n,4)
                if not np.any(states[:,0]==1):break
                encoder=device.create_command_encoder()
                for slot,words in [(3,capacity*2),(4,capacity)]:
                    larger=device.create_buffer(size=n*words*4,usage=usage.STORAGE|usage.COPY_SRC|usage.COPY_DST)
                    encoder.copy_buffer_to_buffer(buffers[slot],0,larger,0,buffers[slot].size)
                    retired.append(buffers[slot]);buffers[slot]=larger
                device.queue.submit([encoder.finish()])
                dispatch_post(capacity,32 if capacity==128 else 16,96)
    triangles=np.frombuffer(device.queue.read_buffer(buffers[3]),np.uint32).reshape(-1,n).T.copy()
    indices=np.frombuffer(device.queue.read_buffer(buffers[4]),np.uint32).reshape(-1,n).T.copy()
    states=np.frombuffer(device.queue.read_buffer(buffers[5]),np.uint32).reshape(n,4).copy()
    for b in buffers+retired:b.destroy()
    device.destroy()
    return coords,order,states,triangles,indices


def validate_main(points):
    coords,order,states,triangles,indices=main_pass(points);n=len(points)
    # Ownership from add_triangle: exactly one other vertex is lexicographically
    # greater than the seed. Every nondegenerate Qhull tet has one such owner.
    ranks=np.empty(n,int);ranks[np.lexsort((np.arange(n),coords[:n,2],coords[:n,1],coords[:n,0]))]=np.arange(n)
    expected=[set() for _ in range(n)]
    for t in Delaunay(coords[:n,:3].astype(float)).simplices:
        seed=sorted(t,key=lambda i:ranks[i])[-2]
        expected[seed].add(tuple(sorted(t)))
    completed=0;actual=set()
    for seed,(status,nv,nt,cycles) in enumerate(states):
        assert status in [0,1,2,3,4] and 4<=nv<=32 and nt<=64 and cycles<=40
        assert status!=2,('inconsistent boundary',seed,states[seed])
        local=triangles[seed,:nt];faces=np.stack([local&255,(local>>8)&255,(local>>16)&255],axis=1)
        assert np.all(faces<nv)
        assert np.all(indices[seed,faces]&0x7fffffff<n+4)
        # A star keeps all its triangles; the tetrahedra it outputs carry GAO_OUTPUT (4).
        owned=((local>>24)&4)!=0
        global_faces=indices[seed,faces[owned]]&0x7fffffff
        if status:continue
        completed+=1
        assert np.all(global_faces<n)
        assert not np.any(indices[seed,:nv]&0x80000000)
        emitted={tuple(sorted([seed,*map(int,f)])) for f in global_faces}
        assert emitted==expected[seed],(seed,len(emitted),len(expected[seed]),emitted^expected[seed])
        assert len(emitted)==owned.sum()
        for f in global_faces:
            v=coords[[*f,seed],:3].astype(float)
            assert np.linalg.det(v[1:]-v[:1])>0
        assert not (actual&emitted)
        actual|=emitted
    assert completed>n//2,dict(zip(*np.unique(states[:,0],return_counts=True)))
    return completed,states


def test_gao_main_random_completed_stars():
    for n in [64,1000]:
        points=np.random.default_rng(193+n).uniform(0.05,0.95,(n,3)).astype(np.float32)
        completed,states=validate_main(points)
        print(n,'sites;',completed,'certified main stars;',dict(zip(*np.unique(states[:,0],return_counts=True))))


def test_gao_main_degenerate_handoff():
    points=np.array([(i,j,k) for i in range(-3,4) for j in range(-3,4) for k in range(-3,4) if (i+j+k)%2==0],np.float32)
    _,_,states,triangles,indices=main_pass(points)
    assert (states[:,0]!=0).any(),'cospherical stars must be left for exact processing'
    assert not (states[:,0]==2).any()
    for seed,(status,nv,nt,cycles) in enumerate(states):
        assert nv<=32 and nt<=64
        if status==4:
            assert np.any(triangles[seed,:nt]>>24&2),'exact-processing status must preserve uncertain triangles'
            assert np.any(indices[seed,:nv]&0x80000000),'unfinished mesh keeps ordering bits'


def completed_tets(points,post_lanes=32,*,grow=False,expected_capacity=None,filtered=True):
    coords,order,states,triangles,indices=main_pass(points,post=True,post_lanes=post_lanes,grow=grow,filtered=filtered)
    if expected_capacity is not None:assert indices.shape[1]==expected_capacity
    assert np.all(states[:,0]==0),dict(zip(*np.unique(states[:,0],return_counts=True)))
    tets=[]
    for seed,(_,nv,nt,cycles) in enumerate(states):
        local=triangles[seed,:nt];local=local[((local>>24)&4)!=0];faces=np.stack([local&255,(local>>8)&255,(local>>16)&255],axis=1)
        assert np.all(faces<nv)
        for f in indices[seed,faces]:
            assert np.all(f<len(points))
            tets.append(order[[*map(int,f),seed]])
    return np.asarray(tets,np.uint32)


@pytest.mark.parametrize('post_lanes',[32,128])
def test_gao_main_and_cooperative_random(post_lanes):
    from test_gpu_delaunay import check
    points=np.random.default_rng(1193).uniform(0.05,0.95,(1000,3)).astype(np.float32)
    check(points,completed_tets(points,post_lanes))


def test_gao_main_and_cooperative_fcc():
    from test_gpu_delaunay import check
    points=np.array([(i,j,k) for i in range(-3,4) for j in range(-3,4) for k in range(-3,4) if (i+j+k)%2==0],np.float32)
    tets=completed_tets(points);reference=completed_tets(points,128)
    check(points,tets,False)
    assert set(map(tuple,np.sort(tets,axis=1)))==set(map(tuple,np.sort(reference,axis=1)))


@pytest.mark.parametrize('neighbors',[80,180])
@pytest.mark.parametrize('filtered',[False,True])
def test_gao_gpu_capacity_growth_preserves_pending_face(neighbors,filtered):
    from test_gpu_delaunay import check
    rng=np.random.default_rng(319+neighbors)
    angle=(np.arange(neighbors)+rng.uniform(-.08,.08,neighbors))*2*np.pi/neighbors
    x=np.where(np.arange(neighbors)%2,.03,-.03)
    radius=np.sqrt(1-x*x)
    sphere=np.vstack([np.column_stack([x,radius*np.cos(angle),radius*np.sin(angle)]),[[1,0,0],[-1,0,0]]])
    points=np.vstack([np.zeros((1,3)),sphere]).astype(np.float32)
    if neighbors==80:
        _,_,states,_,_=main_pass(points,post=True,filtered=filtered)
        assert np.any(states[:,0]==1),'fixture must exhaust the 64-vertex local mesh'
    tets=completed_tets(points,grow=True,expected_capacity=128 if neighbors==80 else 256,filtered=filtered);check(points,tets)
    incident=tets[np.any(tets==0,axis=1)]
    assert set(np.unique(incident))==set(range(len(points)))
