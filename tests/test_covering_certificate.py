"""Independent domain-neighborhood checks for pruned covering certificates."""
import json
from pathlib import Path

import numpy as np
import pytest
import wgpu

ROOT = Path(__file__).resolve().parents[1]
OPS = json.loads((ROOT/'tests/fixtures/plot-ops.json').read_text())


def fold(q, ops):
    w = q[:, None, :1]*ops[None, :, :1] - np.sum(q[:, None, 1:]*ops[None, :, 1:], axis=-1, keepdims=True)
    v = q[:, None, :1]*ops[None, :, 1:] + q[:, None, 1:]*ops[None, :, :1] + np.cross(q[:, None, 1:], ops[None, :, 1:])
    images = np.concatenate([w, v], axis=-1)
    best = np.argmax(np.abs(images[:, :, 0]), axis=1)
    result = images[np.arange(len(q)), best]
    result *= np.where(result[:, :1] < 0, -1, 1)
    return result/np.linalg.norm(result, axis=1, keepdims=True)


@pytest.mark.parametrize('group', OPS)
def test_pruned_chart_vertices_still_cover_fz(group):
    ops = np.asarray(OPS[group], float)
    ops /= np.linalg.norm(ops, axis=1, keepdims=True)
    normals = np.concatenate([np.array([[1., 0, 0, 0]])-ops[1:], np.array([[1., 0, 0, 0]])+ops])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # Match represented normals, then use f64 for the independent decision.
    normals = normals.astype(np.float32).astype(float)
    m = 24
    side = m+1
    n = 4*side**3
    params = np.zeros(16, np.uint32)
    params[3] = m
    params[14] = np.float32(np.sqrt(3)/m).view(np.uint32)
    code = ''.join((ROOT/'shaders'/name).read_text() for name in ['dt_common.wgsl', 'dt_math.wgsl', 'dt_double.wgsl', 'covering.wgsl']) + '''
@group(0) @binding(16) var<storage,read_write> decisions:array<u32>;
@compute @workgroup_size(128)
fn verify_probes(@builtin(global_invocation_id) id:vec3<u32>){
 if(id.x>=arrayLength(&decisions)){return;}
 decisions[id.x]=u32(certificate_probe_near_zone(certificate_chart_probe(id.x)));
}
'''
    device = wgpu.gpu.request_adapter_sync(power_preference='high-performance').request_device_sync()
    usage = wgpu.BufferUsage
    module = device.create_shader_module(code=code)
    pipe = device.create_compute_pipeline(layout='auto', compute={'module': module, 'entry_point': 'verify_probes'})
    buffers = {
        0: device.create_buffer_with_data(data=params, usage=usage.UNIFORM),
        11: device.create_buffer_with_data(data=normals.astype(np.float32), usage=usage.STORAGE),
        16: device.create_buffer(size=n*4, usage=usage.STORAGE | usage.COPY_SRC),
    }
    bind = device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[{'binding': k, 'resource': {'buffer': b}} for k, b in buffers.items()])
    encoder = device.create_command_encoder()
    work = encoder.begin_compute_pass()
    work.set_pipeline(pipe)
    work.set_bind_group(0, bind)
    work.dispatch_workgroups((n+127)//128)
    work.end()
    device.queue.submit([encoder.finish()])
    retained = np.frombuffer(device.queue.read_buffer(buffers[16]), np.uint32).astype(bool)
    # A direct f64 halfspace test must never be excluded. Test all chart points.
    indices = np.arange(n)
    chart = indices//side**3
    local = indices % side**3
    xyz = np.stack([local % side, (local//side) % side, local//side**2], axis=1)
    chart_points = np.empty((n, 4))
    for k in range(4):
        rows = chart == k
        chart_points[rows, k] = 1
        chart_points[np.ix_(rows, np.arange(4) != k)] = 2*xyz[rows]/m-1
    chart_points /= np.linalg.norm(chart_points, axis=1, keepdims=True)
    sides = chart_points @ normals.T
    threshold = np.sin(np.sqrt(3)/m)
    required = np.all(sides >= -threshold, axis=1) | np.all(sides <= threshold, axis=1)
    assert retained[required].all()
    assert np.sum(retained) < n
    # Independently quantize each FZ rotation in its dominant chart. Its nearest
    # parameter vertex must survive, even at equatorial seams and antipodes.
    rng = np.random.default_rng(862)
    points = rng.normal(size=(20000, 4))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points = fold(points, ops)
    points = np.concatenate([points, fold(np.eye(4), ops), fold(-np.eye(4), ops)])
    major = np.argmax(np.abs(points), axis=1)
    parameters = points/points[np.arange(len(points)), major, None]
    for k in range(4):
        rows = major == k
        if not np.any(rows):
            continue
        at = np.clip(np.rint((parameters[rows][:, np.arange(4) != k]+1)*m/2), 0, m).astype(int)
        nearest = k*side**3 + at[:, 0] + side*(at[:, 1]+side*at[:, 2])
        assert retained[nearest].all(), 'A nearest FZ chart vertex was pruned'
        distance = np.arccos(np.clip(np.abs(np.sum(points[rows]*chart_points[nearest], axis=1)), 0, 1))
        assert np.max(distance) <= np.sqrt(3)/m + 1e-14
    for b in buffers.values():
        b.destroy()
    device.destroy()
