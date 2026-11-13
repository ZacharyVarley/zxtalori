import ctypes as C
import numpy as np
import zxtalori as zo

def test_haar_repeatable_and_invariant_uniform():
    with zo.haar('C2',count=1000,seed=48) as g:
        q=g.quaternions;n=len(q)
        assert np.max(abs(np.linalg.norm(q,axis=1)-1))<3e-6
        # Uniform on [0, 1] under Haar measure: its mean within four standard errors, and its
        # empirical distribution within the 0.1% Kolmogorov-Smirnov bound.
        invariant=q[:,0]**2+q[:,3]**2
        assert abs(float(invariant.mean())-.5)<4/np.sqrt(12*n)
        assert np.max(abs(np.sort(invariant)-np.arange(n)/n))<1.95/np.sqrt(n)
        assert np.max(np.abs(q@g.operations.T)-q[:,:1])<1e-6
    with zo.haar('C2',count=1000,seed=48) as g:np.testing.assert_array_equal(g.quaternions,q)

def test_actual_shell_mesh_export():
    with zo.generate('O',h=2,kind='transport') as g:
        mesh=g.triangulation()
        assert len(mesh['stereographic'])>=len(g)
        assert len(mesh['tetrahedra'])>0
        assert mesh['tetrahedra'].max()<len(mesh['stereographic'])
        assert g.stats['covering_current']
