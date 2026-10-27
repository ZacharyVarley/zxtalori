"""Per-point display values against independent references: the Delaunay triangulation
of the orbit on S3 is its convex hull in R4 (scipy), and separations come from a k-d tree."""
import math
import numpy as np
import pytest
from scipy.spatial import ConvexHull, cKDTree
import zxtalori


def multiply(a, b):
    return np.concatenate([(a[..., :1]*b[..., :1]-(a[..., 1:]*b[..., 1:]).sum(-1,keepdims=True)),
        a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]+np.cross(a[..., 1:],b[..., 1:])],axis=-1)


def orbit(q, ops):
    q=q.astype(np.float64);ops=ops.astype(np.float64)
    images=np.concatenate([multiply(q,np.broadcast_to(op,q.shape)) for op in ops])
    images/=np.linalg.norm(images,axis=1,keepdims=True)
    return np.concatenate([images,-images]),np.tile(np.arange(len(q)),2*len(ops))


@pytest.mark.parametrize("group,count",[("O",300),("D6",250),("C2",120)])
def test_point_values_match_hull(group, count):
    with zxtalori.haar(group,count=count,seed=5) as grid:
        grid.relax(30)
        grid.covering()
        q=grid.quaternions;ops=grid.operations
        points,source=orbit(q,ops)
        hull=ConvexHull(points)
        # Identity images are the first n points; their neighbours are distinct hull vertices.
        neighbours=[set() for _ in range(count)]
        largest=np.zeros(count)
        for facet,(normal,offset) in zip(hull.simplices,zip(hull.equations[:,:4],hull.equations[:,4])):
            radius=math.degrees(2*math.acos(min(1,-offset/np.linalg.norm(normal))))
            for v in facet:
                if v<count:neighbours[v].update(int(w) for w in facet if w!=v)
                largest[source[v]]=max(largest[source[v]],radius)
        faces=grid.point_values("faces");covering=grid.point_values("covering")
        assert faces is not None and covering is not None
        assert np.array_equal(faces,[len(s) for s in neighbours])
        assert np.allclose(covering,largest,atol=2e-3)
        assert covering.max()==pytest.approx(grid.stats["covering_radius_deg"],abs=1e-3)
        grid.measure()
        separation=grid.point_values("separation")
        distance,index=cKDTree(points).query(points[:count],k=3)
        expected=[next(4*math.degrees(math.asin(d/2)) for d,j in zip(row,ids) if j!=i) for i,(row,ids) in enumerate(zip(distance,index))]
        assert np.allclose(separation,expected,atol=2e-3)
        assert grid.stats["point_separation_iteration"]==grid.stats["iteration"]


def test_point_values_track_runs():
    with zxtalori.haar("O",count=2000,seed=9) as grid:
        assert grid.point_values("covering") is None
        grid.relax(10)
        assert grid.point_values("separation") is not None
        assert grid.stats["point_separation_iteration"]==grid.stats["iteration"]
        grid.optimize_covering(20,batch=10)
        assert grid.stats["point_covering_iteration"]==grid.stats["iteration"]
        values=grid.point_values("covering")
        assert values.max()==pytest.approx(grid.stats["covering_radius_deg"],abs=1e-3)
