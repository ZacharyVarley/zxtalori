"""End-to-end native engine measures, quotient invariance and Quey descent.

This compares the installed Zig/WGSL library through the public Python API
against an independent dense NumPy f64 model, not another engine path.
"""
from pathlib import Path
import json
import math
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
sys.path.insert(0,str(ROOT/'tests'))
import zxtalori as zo
from test_pair_kernels import reference


def receipt(root, name):
    """Validation receipts stay local, in the ignored research/validation folder."""
    folder = root / "research/validation"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def expected_measures(q,ops):
    rows=reference(q,ops)
    order=len(ops);n=len(q);count=2*order*n
    sums=2*order*rows[:,:5].sum(0)
    cgreen=3**(5/3)/(5*(2*math.pi**2)**(1/3)*(4*math.pi)**(2/3))
    delta=(sums[3]+count*.25+count*math.log(count)/3)/count
    return {
        'riesz_1':sums[0]/(count*count*8/(3*math.pi)),
        'riesz_2':sums[1]/(count*count),
        'riesz_3':sums[2]/(count*count*2/(3*math.pi)*math.log(count)),
        'green_ratio':sums[4]/(-cgreen*count**(4/3)),
        'log_ratio':delta/(-5/12),
        'quey_energy':rows[:,5].sum()/(n*order*max(n-1,1)),
        'separation_deg':4*math.asin(math.sqrt(rows[:,6].min())/2)*180/math.pi,
        'force_rms':float(np.sqrt(np.square(rows[:,8:11]).sum(axis=1).mean())),
    }


def main():
    rng=np.random.default_rng(312)
    q=rng.normal(size=(29,4));q/=np.linalg.norm(q,axis=-1,keepdims=True)
    report={}
    for group in zo.GROUPS:
        with zo.from_quaternions(q,group) as grid:
            current=grid.quaternions
            actual=grid.measure()
            expected=expected_measures(current,grid.operations)
            for key,value in expected.items():
                np.testing.assert_allclose(actual[key],value,rtol=4e-4,atol=4e-4,err_msg=f'{group} {key}')
            signed=current.copy();signed[::2]*=-1
            grid.set(signed,group)
            invariant=grid.measure()
            for key in expected:
                np.testing.assert_allclose(invariant[key],actual[key],rtol=4e-4,atol=4e-4,err_msg=f'{group} antipodes {key}')
            report[group]={'random_measures':True,'antipodal_invariance':True,'max_measure_absolute_error':float(max(abs(actual[k]-v) for k,v in expected.items()))}
        with zo.generate(group,h=3,count=256) as grid:
            initial=grid.measure()['quey_energy']
            grid.relax(steps=3)
            quick=grid.stats
            assert quick['energy_ratios_current'] is False
            assert quick['riesz_1'] is None and quick['log_ratio'] is None
            final=grid.measure()['quey_energy']
            np.testing.assert_allclose(quick['quey_energy'],final,rtol=2e-6,atol=2e-6)
            assert final<=initial+1e-6,(group,initial,final)
            np.testing.assert_allclose(np.linalg.norm(grid.quaternions,axis=-1),1,rtol=3e-7,atol=3e-7)
            assert grid.stats['iteration']==3
            report[group].update({'generated_count':len(grid),'initial_quey':initial,'final_quey':final,'unit_norm_after_relax':True})
    # Distances obtained from chords must resolve pairs closer than dot-product precision.
    near=np.array([[1,0,0,0],[math.cos(5e-5),math.sin(5e-5),0,0]],np.float64)
    with zo.from_quaternions(near,'C2') as grid:
        actual=grid.measure()
        expected=expected_measures(grid.quaternions,grid.operations)
        np.testing.assert_allclose(actual['separation_deg'],expected['separation_deg'],rtol=1e-5)
        assert actual['separation_deg']>0
        report['_near_pair']={'separation_deg':actual['separation_deg']}
    with zo.from_quaternions(np.array([[1,0,0,0],[1,0,0,0]],np.float32),'C2') as grid:
        try:grid.measure()
        except RuntimeError as error:assert 'DuplicateOrientations' in str(error)
        else:raise AssertionError('Exact duplicate was not rejected')
    report['_duplicates_rejected']=True
    print(json.dumps(report,indent=2))
    (receipt(ROOT, 'native-measure-validation.json')).write_text(json.dumps(report,indent=2)+'\n')


def test_native_measures():
    main()


def test_import_scale_invariance():
    q=np.random.default_rng(198).normal(size=(29,4))
    q/=np.linalg.norm(q,axis=1,keepdims=True)
    with zo.from_quaternions(q,'I') as grid:
        expected=grid.quaternions
        # All rows are finite and nonzero. Direct normalize(dot(q,q)) would
        # overflow on the large rows despite their valid orientations.
        grid.set(q*np.geomspace(1e-8,3e38,len(q))[:,None])
        actual=grid.quaternions
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual,expected,rtol=0,atol=4e-7)
        np.testing.assert_allclose(np.linalg.norm(actual,axis=1),1,rtol=0,atol=3e-7)
        extreme=np.array([[1,1,1,1],[1,.1,.3,.4]],np.float32)
        grid.set(extreme,'C2')
        expected=grid.quaternions
        grid.set(extreme*np.float32(3e38),'C2')
        np.testing.assert_allclose(grid.quaternions,expected,rtol=0,atol=4e-7)


if __name__=='__main__':main()
