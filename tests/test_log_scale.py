"""Assess log-index error at the current ~1k-site target against dense f64."""
from pathlib import Path
import sys,json,math,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
sys.path.insert(0,str(ROOT/'tests'))
import zxtalori as zo
from test_pair_kernels import multiply


def receipt(root, name):
    """Validation receipts stay local, in the ignored research/validation folder."""
    folder = root / "research/validation"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def reference_log(q,ops):
    q=np.asarray(q,dtype=np.float64);q/=np.linalg.norm(q,axis=1,keepdims=True)
    ops=np.asarray(ops,dtype=np.float64);ops/=np.linalg.norm(ops,axis=1,keepdims=True)
    n=len(q);total=0.
    for k,op in enumerate(ops):
        y=multiply(q,op);y/=np.linalg.norm(y,axis=1,keepdims=True)
        for first in range(0,n,128):
            x=q[first:first+128]
            dm=((x[:,None]-y[None])**2).sum(-1)
            dp=((x[:,None]+y[None])**2).sum(-1)
            if k==0:
                row=np.arange(len(x));col=first+row
                dm[row,col]=1.;dp[row,col]=1.
            values=-.5*np.log(dm*dp)+.5
            if k==0:values[row,col]=0.
            total+=values.sum()
    total+=n*(-math.log(2)+.25)
    full=2*len(ops)*n
    return -12/5*(total/n+.25+math.log(full)/3)


def main():
    report={}
    random=np.random.default_rng(936).normal(size=(1000,4));random/=np.linalg.norm(random,axis=1,keepdims=True)
    cases=[('T',5,None),('O',6,None),('I',6,None),('O',None,random),('I',None,random)]
    for group,h,points in cases:
        with zo.generate(group,h=h) if points is None else zo.from_quaternions(points,group) as grid:
            t0=time.perf_counter();actual=grid.measure()['log_ratio'];gpu_seconds=time.perf_counter()-t0
            expected=reference_log(grid.quaternions,grid.operations)
            error=abs(actual-expected)
            key=group if points is None else group+'_random'
            report[key]={'count':len(grid),'gpu_log_ratio':actual,'f64_log_ratio':expected,'absolute_error':error,'measure_wall_seconds':gpu_seconds}
            assert error<1e-5,(group,error)
    print(json.dumps(report,indent=2))
    (receipt(ROOT, 'log-scale-validation.json')).write_text(json.dumps(report,indent=2)+'\n')


def test_log_scale():
    main()


if __name__=='__main__':main()
