import ctypes as C
import numpy as np
import zxtalori as zo

def test_haar_repeatable_and_invariant_uniform():
    with zo.haar('C2',count=20000,seed=48) as g:
        q=g.quaternions
        assert np.max(abs(np.linalg.norm(q,axis=1)-1))<3e-6
        invariant=q[:,0]**2+q[:,3]**2
        assert abs(float(invariant.mean())-.5)<.01
        assert np.max(abs(np.sort(invariant)-np.arange(len(q))/len(q)))<.015
        assert np.max(np.abs(q@g.operations.T)-q[:,:1])<1e-6
    with zo.haar('C2',count=20000,seed=48) as g:np.testing.assert_array_equal(g.quaternions,q)

def test_actual_shell_mesh_export():
    with zo.generate('O',h=2) as g:
        mesh=g.triangulation()
        assert len(mesh['stereographic'])>=len(g)
        assert len(mesh['tetrahedra'])>0
        assert mesh['tetrahedra'].max()<len(mesh['stereographic'])
        assert g.stats['covering_current']
