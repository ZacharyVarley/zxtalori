"""Regressions for zone seams, identity constraints and exact-count augmentation."""
import numpy as np
import pytest
import zxtalori
from scipy.spatial import cKDTree

@pytest.mark.parametrize('group',zxtalori.GROUPS)
def test_exact_count_identity_and_zone_after_relaxation(group):
    with zxtalori.generate(group,count=1000,exact=True,fixed_identity=True) as grid:
        assert len(grid)==1000
        assert grid.stats['base_count']+grid.stats['inserted_count']==1000
        before=grid.quaternions.copy()
        # Initial augmentation must not cluster nearly co-spherical cell centres.
        q=before.astype(np.float64);ops=grid.operations.astype(np.float64)
        w=q[:,None,:1]*ops[None,:,:1]-np.sum(q[:,None,1:]*ops[None,:,1:],axis=-1,keepdims=True)
        v=q[:,None,:1]*ops[None,:,1:]+q[:,None,1:]*ops[None,:,:1]+np.cross(q[:,None,1:],ops[None,:,1:])
        cloud=np.concatenate((w,v),axis=-1).reshape(-1,4)
        cloud=np.concatenate((cloud,-cloud))
        assert cKDTree(cloud).query(q,k=2)[0][:,1].min()>0.01
        identity=grid.stats['identity_index']
        assert identity is not None
        grid.relax(12)
        q=grid.quaternions.astype(np.float64)
        ops=grid.operations.astype(np.float64)
        assert np.max(np.abs(q@ops.T)-q[:,:1])<1e-6
        np.testing.assert_array_equal(q[identity],[1,0,0,0])
        assert np.max(np.abs(q-before))>1e-5
        assert grid.stats['separation_deg']>0
        grid.identity(fixed=False)
        assert not grid.stats['fixed_identity']


def test_add_identity_is_idempotent_and_preserves_generator():
    with zxtalori.generate('C4',count=500) as grid:
        n=len(grid);name=grid.stats['generator']
        grid.identity(add=True,fixed=True)
        assert len(grid)==n+1
        grid.identity(add=True,fixed=True)
        assert len(grid)==n+1
        assert grid.stats['generator']==name


@pytest.mark.parametrize('group,count',[('C6',84),('D3',62),('D3',84),('D4',84)])
def test_exact_count_thin_shell_hull(group,count):
    with zxtalori.generate(group,count=count,exact=True,identity=True) as grid:
        assert len(grid)==count
