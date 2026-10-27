import ctypes as C
import numpy as np
import pytest
import zxtalori as zo

def with_scale(g, scale):
    lib = g._lib
    lib.zo_pair_budget.argtypes, lib.zo_pair_budget.restype = [C.c_void_p, C.c_float], None
    lib.zo_pair_budget(g._handle, scale)

@pytest.mark.parametrize("group,count", [("T", 700), ("O", 1500), ("D4", 400)])
def test_chunked_pair_sums_match_single_dispatch(group, count):
    """Splitting the all-pairs sums over operator and site ranges changes only rounding."""
    g = zo.haar(group, count=count, seed=5)
    with_scale(g, 1)
    whole = dict(g.measure())
    g.relax(1); moved_whole = g.quaternions.copy()
    g.set(zo.haar(group, count=count, seed=5).quaternions, group)
    with_scale(g, 1e-5)  # chunks of 64 sites, one operator per dispatch
    chunked = dict(g.measure())
    g.relax(1); moved_chunked = g.quaternions.copy()
    for key in ("quey_energy", "riesz_1", "riesz_2", "riesz_3", "green_ratio", "log_ratio", "separation_deg"):
        assert chunked[key] == pytest.approx(whole[key], rel=2e-5, abs=2e-6), key
    assert np.max(np.abs(moved_chunked - moved_whole)) < 2e-5
    g.close()
