"""Later depth-ray witnesses for recovering untyped stale geometry."""
import numpy as np
from scipy.spatial import cKDTree

class RayEvidence:
    def __init__(self,packet,ground_mask=None):
        self.camera=packet['camera']
        points=packet['points'][(packet['positive'] if ground_mask is None else ground_mask)&~packet['negative']]
        rays=points-self.camera;length=np.linalg.norm(rays,axis=1)
        valid=np.isfinite(rays).all(1)&(length>0)
        self.distance=length[valid];self.direction=rays[valid]/self.distance[:,None]
        self.tree=cKDTree(self.direction)
        self.lower=self.direction.min(0) if len(self.direction) else np.zeros(3)
        self.upper=self.direction.max(0) if len(self.direction) else np.zeros(3)

    def clears(self,points,margin):
        """A nearby observed ray must pass the old point and end farther away.

        Ground labels alone cannot clear occupied volume. No explicit negative
        samples are passed here; callers also require two newer observations.
        """
        points=np.asarray(points);out=np.zeros(len(points),bool)
        if not len(self.distance) or not len(points):return out
        delta=points-self.camera;length=np.linalg.norm(delta,axis=1)
        valid=np.isfinite(delta).all(1)&(length>margin);rows=np.flatnonzero(valid)
        unit=delta[rows]/length[rows,None]
        # Older terrain behind/outside the latest observed ray fan cannot be
        # cleared by these observations. Avoid expensive nearest-ray queries.
        padding=margin/np.maximum(length[rows],margin)
        inside=((unit>=self.lower-padding[:,None])&(unit<=self.upper+padding[:,None])).all(1)
        rows=rows[inside]
        for start in range(0,len(rows),8192):
            selected=rows[start:start+8192];q=delta[selected];r=length[selected]
            _,nearest=self.tree.query(q/r[:,None],k=min(4,len(self.distance)))
            if nearest.ndim==1:nearest=nearest[:,None]
            along=np.einsum('bi,bki->bk',q,self.direction[nearest])
            perpendicular=np.maximum(0,r[:,None]**2-along**2)
            clear=(along>0)&(perpendicular<=(margin/2)**2)&(self.distance[nearest]>along+2*margin)
            out[selected]=clear.any(1)
        return out
