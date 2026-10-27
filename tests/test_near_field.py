"""Near-field relaxation (src/near.zig): the GPU sums against a dense f64 reference, and
an outcome close to all pairs."""
import math
import numpy as np
import pytest
import zxtalori


def multiply(a, b):
    return np.concatenate([(a[..., :1]*b[..., :1]-(a[..., 1:]*b[..., 1:]).sum(-1,keepdims=True)),
        a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]+np.cross(a[..., 1:],b[..., 1:])],axis=-1)


def mean_radius(count):
    target=math.pi**2/count;lo,hi=0.0,math.pi
    for _ in range(60):
        middle=0.5*(lo+hi)
        if math.pi*(middle-math.sin(middle))<target: lo=middle
        else: hi=middle
    return 0.25*(lo+hi)


def near_reference(q, ops, delta):
    """Windowed near-field Quey energy (per pair, as stats['energy']) and separation."""
    q=q.astype(np.float64);ops=ops.astype(np.float64);n=len(q);order=len(ops)
    r=min(1.5,delta*mean_radius(n*order));cut=(2*math.sin(r/2))**2
    images=np.concatenate([multiply(q,np.broadcast_to(op,q.shape)) for op in ops])
    images/=np.linalg.norm(images,axis=1,keepdims=True)
    source=np.tile(np.arange(n),order)
    energy=0.0;nearest=4.0
    for i in range(n):
        dm2=((images-q[i])**2).sum(1);dp2=((images+q[i])**2).sum(1)
        near=np.minimum(dm2,dp2);far=np.maximum(dm2,dp2)
        own=source==i;own[i]=False  # the identity image of q[i] is q[i]
        pick=(near<=cut)&(source!=i)
        a=2*np.arctan(np.sqrt(near[pick]/far[pick]))
        s=np.clip((a-0.5*r)/(0.5*r),0,1)
        energy+=(math.pi/(a*(math.pi-a))*(1-s*s*(3-2*s))).sum()
        nearest=min(nearest,near[pick].min(initial=4.0),near[own].min(initial=4.0))
    return energy/(n*order*(n-1)),4*math.degrees(math.asin(math.sqrt(nearest)/2))


@pytest.mark.parametrize("group,count",[("O",1500),("D6",1200),("C2",800)])
def test_near_sums_match_dense_reference(group, count):
    with zxtalori.haar(group,count=count,seed=11) as grid:
        grid.relax(20)
        grid.relax(1,neighbourhood=20)
        stats=grid.stats
        assert stats["energy_neighbourhood"]==20 and stats["relax_neighbourhood"]==20
        energy,separation=near_reference(grid.quaternions,grid.operations,20)
        assert stats["energy"]==pytest.approx(energy,rel=2e-6)
        assert stats["separation_deg"]==pytest.approx(separation,rel=1e-4)
        grid.relax(1,neighbourhood=0)
        assert grid.stats["energy_neighbourhood"] is None


def test_near_relaxation_approaches_all_pairs():
    outcomes={}
    for delta in (0,20):
        with zxtalori.haar("O",count=4000,seed=7) as grid:
            grid.relax(400,neighbourhood=delta)
            grid.relax(0,neighbourhood=0);grid.measure()
            outcomes[delta]=grid.stats
    assert outcomes[20]["energy"]==pytest.approx(outcomes[0]["energy"],rel=1e-5)
    assert outcomes[20]["separation_deg"]>0.95*outcomes[0]["separation_deg"]


def test_neighbourhood_range():
    with zxtalori.haar("O",count=200,seed=3) as grid:
        with pytest.raises(RuntimeError):
            grid.relax(1,neighbourhood=1)
