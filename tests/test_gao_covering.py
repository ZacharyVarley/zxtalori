"""Native Zig Gao scheduling against an independent full-orbit covering oracle."""
import ctypes as C
import numpy as np
import pytest
import zxtalori as zo
from test_covering import full_orbit_covering


@pytest.mark.parametrize('group',zo.GROUPS)
def test_source_guided_gao_covering(group):
    with zo.generate(group,h=2,count=100) as grid:
        grid._lib.zo_delaunay_mode.argtypes=[C.c_void_p,C.c_uint32]
        grid._lib.zo_delaunay_mode(grid._handle,32)
        for step in [0,3]:
            if step:grid.relax(step)
            measured=grid.covering()
            expected=full_orbit_covering(grid.quaternions,grid.operations)
            assert abs(measured['covering_radius_deg']-expected)<2e-4
            assert measured['covering_current']
            assert measured['triangulation_backend']=='wgsl'
            assert np.isfinite(measured['covering_witness']).all()
