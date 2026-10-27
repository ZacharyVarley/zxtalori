"""Direct Gao local-mesh operations executed in WGSL, with independent topology checks."""
from pathlib import Path
from collections import Counter
import numpy as np
import wgpu
ROOT=Path(__file__).resolve().parents[1]


def test_gao_packed_boundary_and_vertex_compaction():
    points=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    faces=[]
    for x in [0,1]:
        for y in [2,3]:
            for z in [4,5]:
                f=[x,y,z]
                if np.linalg.det(points[f])<0:f[1],f[2]=f[2],f[1]
                faces.append(f)
    masks=[];expected=[]
    for mask in range(1,255):
        edges=Counter()
        for i,f in enumerate(faces):
            if mask>>i&1:
                for j in range(3):
                    edge=(f[j],f[(j+1)%3]);reverse=edge[::-1]
                    if edges[reverse]:edges[reverse]-=1
                    else:edges[edge]+=1
        boundary={edge for edge,n in edges.items() if n}
        successor={a:b for a,b in boundary}
        if len(successor)!=len(boundary) or len({b for _,b in boundary})!=len(boundary):continue
        start=next(iter(successor));at=start;visited=set()
        while at not in visited:visited.add(at);at=successor[at]
        if at!=start or len(visited)!=len(boundary):continue
        masks.append(mask);expected.append(boundary)
    packed=[f[0]|f[1]<<8|f[2]<<16 for f in faces]
    prefix='''const GAO_POST=false; const GAO_MAX_P=32u; const GAO_MAX_T=64u;
@group(0) @binding(0) var<storage,read> masks:array<u32>;
@group(0) @binding(1) var<storage,read_write> triangles:array<u32>;
@group(0) @binding(2) var<storage,read_write> indices:array<u32>;
@group(0) @binding(3) var<storage,read_write> output:array<u32>;
var<private> case_id:u32;
fn gao_triangle(i:u32)->vec4<u32>{let t=triangles[case_id*64u+i];return vec4<u32>(t&255u,(t>>8u)&255u,(t>>16u)&255u,t>>24u);}
fn gao_write_triangle(i:u32,t:vec4<u32>){triangles[case_id*64u+i]=t.x|(t.y<<8u)|(t.z<<16u)|(t.w<<24u);}
fn gao_index(i:u32)->u32{return indices[case_id*32u+i];}
fn gao_write_index(i:u32,value:u32){indices[case_id*32u+i]=value;}
'''
    prefix+='''
fn gao_conflict(encoded_id:u32,t:vec4<u32>)->vec2<bool>{
 for(var i=0u;i<8u;i++){let packed=t.x|(t.y<<8u)|(t.z<<16u);if(packed==initial[i]){return vec2<bool>((masks[case_id]&(1u<<i))!=0u,encoded_id!=0xfffffffeu);}}
 return vec2<bool>(false,true);
}
'''
    suffix='const initial=array<u32,8>('+','.join(str(x)+'u' for x in packed)+');\n'+'''
@compute @workgroup_size(64)
fn boundary_check(@builtin(global_invocation_id) id:vec3<u32>){
 case_id=id.x;if(case_id>=arrayLength(&masks)){return;}gao_nv=6u;let mask=masks[case_id];
 for(var i=0u;i<8u;i++){if((mask&(1u<<i))==0u){triangles[case_id*64u+gao_nt]=initial[i];gao_nt++;}}
 for(var i=0u;i<8u;i++){if((mask&(1u<<i))!=0u){triangles[case_id*64u+gao_nt+gao_nr]=initial[i];gao_nr++;}}
 let first=gao_compute_boundary();output[case_id*34u]=gao_status;output[case_id*34u+1u]=first;
 for(var i=0u;i<32u;i++){output[case_id*34u+2u+i]=gao_boundary_next[i];}
}
@compute @workgroup_size(64)
fn clip_check(@builtin(global_invocation_id) id:vec3<u32>){
 case_id=id.x;if(case_id>=arrayLength(&masks)){return;}gao_nv=6u;gao_nt=8u;
 for(var i=0u;i<6u;i++){gao_write_index(i,i);}for(var i=0u;i<8u;i++){triangles[case_id*64u+i]=initial[i];}
 let changed=gao_halfspace_intersection(777u,1000u);output[case_id*34u]=gao_status;output[case_id*34u+1u]=gao_nt;output[case_id*34u+2u]=gao_nv;output[case_id*34u+3u]=u32(changed);
}
@compute @workgroup_size(1)
fn rollback_check(){
 case_id=0u;gao_nv=6u;gao_nt=8u;
 for(var i=0u;i<6u;i++){gao_write_index(i,i);}for(var i=0u;i<8u;i++){triangles[i]=initial[i];}
 let changed=gao_halfspace_intersection(0xfffffffeu,1000u);output[0]=gao_status;output[1]=gao_nt;output[2]=gao_nv;output[3]=gao_nr;output[4]=u32(changed);
}
@compute @workgroup_size(1)
fn compact_check(){
 case_id=0u;gao_nv=32u;gao_nt=4u;for(var i=0u;i<32u;i++){gao_write_index(i,100u+i);}
 gao_write_triangle(0u,vec4<u32>(0u,8u,3u,0u));gao_write_triangle(1u,vec4<u32>(0u,3u,31u,0u));
 gao_write_triangle(2u,vec4<u32>(0u,31u,8u,0u));gao_write_triangle(3u,vec4<u32>(8u,31u,3u,0u));
 let slot=gao_add_vertex(777u);output[0]=gao_status;output[1]=gao_nv;output[2]=slot;
}
@compute @workgroup_size(1)
fn overflow_check(){
 case_id=0u;gao_nv=32u;gao_nt=31u;
 for(var i=0u;i<32u;i++){gao_write_index(i,i);}
 for(var i=1u;i<31u;i++){gao_write_triangle(i-1u,vec4<u32>(0u,i,i+1u,0u));}
 gao_write_triangle(30u,vec4<u32>(0u,31u,1u,0u));
 gao_remove_triangle(0u);let changed=gao_halfspace_intersection(777u,1000u);
 output[0]=gao_status;output[1]=gao_nv;output[2]=gao_nt;output[3]=gao_nr;output[4]=u32(changed);
}
'''
    device=wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync()
    module=device.create_shader_module(code=prefix+(ROOT/'shaders/gao_topology.wgsl').read_text()+suffix)
    u=wgpu.BufferUsage;n=len(masks)
    buffers={0:device.create_buffer_with_data(data=np.array(masks,np.uint32),usage=u.STORAGE),1:device.create_buffer(size=n*64*4,usage=u.STORAGE|u.COPY_SRC),2:device.create_buffer(size=n*32*4,usage=u.STORAGE|u.COPY_SRC),3:device.create_buffer(size=n*34*4,usage=u.STORAGE|u.COPY_SRC)}
    def run(entry,groups,slots):
        pipe=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':entry})
        bind=device.create_bind_group(layout=pipe.get_bind_group_layout(0),entries=[{'binding':i,'resource':{'buffer':buffers[i]}} for i in slots])
        encoder=device.create_command_encoder();work=encoder.begin_compute_pass();work.set_pipeline(pipe);work.set_bind_group(0,bind);work.dispatch_workgroups(groups);work.end();device.queue.submit([encoder.finish()])
    run('boundary_check',(n+63)//64,[0,1,3])
    result=np.frombuffer(device.queue.read_buffer(buffers[3]),np.uint32).reshape(n,34)
    for row,boundary in zip(result,expected):
        assert row[0]==0
        start=at=int(row[1]);actual=set()
        for _ in range(len(boundary)):
            nxt=int(row[2+at]);actual.add((at,nxt));at=nxt
        assert at==start and actual==boundary
    run('clip_check',(n+63)//64,[0,1,2,3])
    rows=np.frombuffer(device.queue.read_buffer(buffers[3]),np.uint32).reshape(n,34)
    triangles=np.frombuffer(device.queue.read_buffer(buffers[1]),np.uint32).reshape(n,64)
    def cyclic(f):return min(tuple(f[i:]+f[:i]) for i in range(3))
    for mask,boundary,row,tri in zip(masks,expected,rows,triangles):
        assert row[0]==0 and row[2]==7 and row[3]==1
        wanted={cyclic(f) for i,f in enumerate(faces) if not (mask>>i&1)}|{cyclic([6,a,b]) for a,b in boundary}
        actual={cyclic([(int(t)>>shift)&255 for shift in [0,8,16]]) for t in tri[:row[1]]}
        assert actual==wanted
    run('rollback_check',1,[0,1,2,3])
    row=np.frombuffer(device.queue.read_buffer(buffers[3]),np.uint32)
    assert list(row[:5])==[0,8,6,0,0]
    tri=np.frombuffer(device.queue.read_buffer(buffers[1]),np.uint32)[:8]
    assert set(map(int,tri))==set(packed)
    run('compact_check',1,[1,2,3]);row=np.frombuffer(device.queue.read_buffer(buffers[3]),np.uint32)
    assert row[0]==0 and row[1]==5
    index=np.frombuffer(device.queue.read_buffer(buffers[2]),np.uint32)[:5]
    assert index[row[2]]==777
    tri=np.frombuffer(device.queue.read_buffer(buffers[1]),np.uint32)[:4]
    actual={tuple(sorted(int(index[(int(t)>>shift)&255]) for shift in [0,8,16])) for t in tri}
    expected={tuple(sorted(100+i for i in f)) for f in [(0,8,3),(0,3,31),(0,31,8),(8,31,3)]}
    assert actual==expected
    run('overflow_check',1,[0,1,2,3]);row=np.frombuffer(device.queue.read_buffer(buffers[3]),np.uint32)
    assert list(row[:5])==[1,32,31,0,0], 'capacity handoff must restore the pre-removed active triangle'
    print(n,'independent cavity boundaries and local vertex compaction passed')
    for buffer in buffers.values():buffer.destroy()
    device.destroy()
