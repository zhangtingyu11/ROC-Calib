"""Exact distance to the union of foreground pixel cells."""
import numpy as np
from scipy.spatial import cKDTree

class PixelRegion:
    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)
        padded = np.pad(self.mask, 1)
        mids, half = [], []
        for neighbor, dx, dy in (
            (padded[1:-1, :-2], -.5, 0), (padded[1:-1, 2:], .5, 0),
            (padded[:-2, 1:-1], 0, -.5), (padded[2:, 1:-1], 0, .5),
        ):
            y, x = np.nonzero(self.mask & ~neighbor)
            mids.append(np.column_stack((x + dx, y + dy)))
            half.append(np.tile([0., .5] if dx else [.5, 0.], (len(x), 1)))
        self.midpoints = np.concatenate(mids)
        self.half_extents = np.concatenate(half)
        if not len(self.midpoints):
            raise ValueError('Empty mask has no finite region distance')
        self.tree = cKDTree(self.midpoints)

    def boundary_distance(self, uv):
        uv = np.asarray(uv, dtype=float)
        if not len(uv):
            return np.empty(0)
        if not np.isfinite(uv).all():
            raise ValueError('Distance queries must be finite')
        _, nearest = self.tree.query(uv)
        delta = np.maximum(np.abs(uv-self.midpoints[nearest])-self.half_extents[nearest], 0.)
        upper = np.linalg.norm(delta, axis=1)
        # Every point of a unit segment is <= .5 from its midpoint. Triangle
        # inequality therefore guarantees inclusion within upper + .5, also
        # when the closest point is an endpoint (not an orthogonal foot).
        radius = np.nextafter(upper+.5, np.inf)
        neighbors = self.tree.query_ball_point(uv, radius)
        counts = np.fromiter((len(x) for x in neighbors), dtype=int, count=len(uv))
        source = np.repeat(np.arange(len(uv)), counts)
        target = np.concatenate(neighbors).astype(int)
        delta = np.maximum(np.abs(uv[source]-self.midpoints[target])-self.half_extents[target], 0.)
        distances = np.linalg.norm(delta, axis=1)
        result = upper.copy()
        np.minimum.at(result, source, distances)
        return result

    def outside_distance(self, uv):
        uv = np.asarray(uv, dtype=float)
        if not len(uv):return np.empty(0)
        if not np.isfinite(uv).all():raise ValueError('Distance queries must be finite')
        xy = np.rint(uv).astype(np.int64)
        h,w = self.mask.shape
        valid = (xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        inside = np.zeros(len(uv), bool)
        inside[valid] = self.mask[xy[valid,1],xy[valid,0]]
        result = np.zeros(len(uv))
        result[~inside] = self.boundary_distance(uv[~inside])
        return result
