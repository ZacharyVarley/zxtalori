"""GPU covering optimization: lowers the radius, keeps the zone and the best sample."""
import numpy as np
import pytest
import zxtalori
from test_covering import full_orbit_covering


def in_zone(grid):
    q=grid.quaternions.astype(np.float64);ops=grid.operations.astype(np.float64)
    return np.max(np.abs(q@ops.T)-q[:,:1])<1e-6


@pytest.mark.parametrize('group,mode',[('O','tet'),('D4','face'),('I','face')])
def test_optimizer_lowers_covering_and_keeps_zone(group,mode):
    with zxtalori.haar(group,count=300,seed=11) as grid:
        grid.relax(20)
        before=grid.covering()['covering_radius_deg']
        grid.optimize_covering(250,mode=mode)
        stats=grid.stats
        assert len(grid)==300
        assert stats['covering_current']
        assert stats['covering_radius_deg']<before
        # The engine keeps the best sample it saw, so the result is that sample.
        assert stats['covering_radius_deg']==pytest.approx(stats['covering_optimizer_best_deg'],abs=1e-5)
        assert np.max(np.abs(np.linalg.norm(grid.quaternions,axis=1)-1))<1e-5
        assert in_zone(grid)
        # Independent full-orbit oracle for the optimized sample.
        assert full_orbit_covering(grid.quaternions,grid.operations)==pytest.approx(stats['covering_radius_deg'],abs=2e-3)


def test_optimizer_holds_fixed_identity_and_stops_on_best():
    with zxtalori.generate('O',count=400,exact=True,fixed_identity=True) as grid:
        index=grid.stats['identity_index']
        calls=[]
        grid.optimize_covering(600,batch=50,callback=lambda g,s:calls.append(s['covering_optimizer_steps']) or len(calls)<3)
        stats=grid.stats
        assert len(calls)==3 and stats['covering_optimizer_steps']==150
        np.testing.assert_array_equal(grid.quaternions[index],[1,0,0,0])
        assert stats['covering_radius_deg']==pytest.approx(stats['covering_optimizer_best_deg'],abs=1e-5)
        with pytest.raises(ValueError):
            grid.optimize_covering(1,mode='everything')


def test_local_retessellation_matches_exact_pass():
    with zxtalori.haar('O',count=400,seed=5) as grid:
        grid.relax(20)
        grid.covering()
        grid.optimize_covering(40,check=True)
        stats=grid.stats
        # Every job's local maximum, from its own GPU re-tessellation, agrees with
        # the exact maximum over the same caps from a full pass after the move.
        assert stats['covering_optimizer_mismatches']==0
        assert stats['covering_optimizer_local_error']<1e-5
        assert stats['covering_optimizer_local_rounds']>0
        assert stats['covering_optimizer_skipped']<stats['covering_optimizer_steps']*stats['covering_optimizer_jobs']


def test_full_pass_variant_still_optimizes():
    with zxtalori.haar('T',count=300,seed=3) as grid:
        grid.relax(20)
        before=grid.covering()['covering_radius_deg']
        grid.optimize_covering(150,local=False)
        assert grid.stats['covering_radius_deg']<before and in_zone(grid)


def test_separation_floor_holds():
    with zxtalori.haar('D4',count=400,seed=9) as grid:
        grid.relax(30)
        before=grid.covering()['covering_radius_deg']
        floor=0.95*grid.measure()['separation_deg']
        grid.optimize_covering(200,min_separation_deg=floor)
        assert grid.stats['covering_optimizer_full_steps']==0
        assert grid.stats['covering_radius_deg']<before
        assert grid.measure()['separation_deg']>=floor*(1-1e-5)
        with pytest.raises(ValueError):
            grid.optimize_covering(1,min_separation_deg=200)
