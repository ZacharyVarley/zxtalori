"""Covering radius against a full-orbit 4D Qhull oracle (tests only)."""
import numpy as np
import pytest
from scipy.spatial import ConvexHull
import zxtalori


def full_orbit_covering(q,operations):
    q=q.astype(np.float64);q/=np.linalg.norm(q,axis=1,keepdims=True)
    ops=operations.astype(np.float64);ops/=np.linalg.norm(ops,axis=1,keepdims=True)
    w=q[:,None,:1]*ops[None,:,:1]-np.sum(q[:,None,1:]*ops[None,:,1:],axis=-1,keepdims=True)
    v=q[:,None,:1]*ops[None,:,1:]+q[:,None,1:]*ops[None,:,:1]+np.cross(q[:,None,1:],ops[None,:,1:])
    cloud=np.concatenate((w,v),axis=-1).reshape(-1,4);cloud/=np.linalg.norm(cloud,axis=1,keepdims=True)
    hull=ConvexHull(np.concatenate((cloud,-cloud)))
    return 2*np.degrees(np.arccos(np.clip(np.min(-hull.equations[:,-1]),-1,1)))


@pytest.mark.parametrize('group',zxtalori.GROUPS)
def test_gpu_covering_full_orbit_and_stale_state(group):
    with zxtalori.generate(group,h=2,count=100) as grid:
        assert grid.stats['covering_radius_deg'] is None
        for stage in range(2):
            if stage:
                previous=grid.stats['covering_radius_deg'];grid.relax(5)
                assert grid.stats['covering_radius_deg']==previous
                assert not grid.stats['covering_current']
            measured=grid.covering()
            expected=full_orbit_covering(grid.quaternions,grid.operations)
            assert abs(measured['covering_radius_deg']-expected)<2e-4
            count=2*len(grid.operations)*len(grid)
            expected_bound=sum(c*count**(-(2*i+1)/3) for i,c in enumerate([1.889832101454757,.573930763512449,.581069217539205,.859789414116394,1.490131881944242,2.791922498626376,5.481645765756282,11.12163594341394]))*360/np.pi
            assert abs(measured['simplex_bound_deg']-expected_bound)<1e-12
            assert abs(measured['covering_excess']-(measured['covering_radius_deg']/measured['simplex_bound_deg']-1))<1e-12
            if stage:assert abs(measured['mesh_ratio']-measured['covering_radius_deg']/measured['separation_deg'])<1e-12
            assert measured['covering_current']
            assert measured['covering_iteration']==grid.stats['iteration']
            assert measured['triangulation_backend']=='wgsl'
            assert measured['covering_certificate_probes']>0
            assert measured['covering_domain_bound_deg']>=measured['covering_radius_deg']
            assert np.isfinite(measured['covering_witness']).all()


def test_imported_grid_covering_and_regeneration_invalidates():
    q=np.random.default_rng(197).normal(size=(65,4))
    with zxtalori.from_quaternions(q,'I') as grid:
        result=grid.covering();assert abs(result['covering_radius_deg']-full_orbit_covering(grid.quaternions,grid.operations))<2e-4
        grid.set(q,'O');assert grid.stats['covering_radius_deg'] is None
