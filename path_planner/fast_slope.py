"""Sparse same-frame slope screening, without per-point KD trees or refits."""
import numpy as np

class SourceSlope:
    """Estimate only requested source rows using bounded image neighborhoods."""
    def __init__(self,packet,reference,radius,ground_mask=None):
        self.packet=packet;self.reference=reference;self.radius=radius
        h,w=map(int,packet['shape']);self.shape=(h,w)
        self.image=np.full((h,w,3),np.nan,np.float32)
        good=(packet['positive'] if ground_mask is None else ground_mask)&~packet['negative']
        self.image.reshape(-1,3)[packet['pixels'][good]]=packet['points'][good]
        self.image=np.pad(self.image,((4,4),(4,4),(0,0)),constant_values=np.nan)
        self.values=np.full(len(packet['points']),np.nan);self.computed=np.zeros(len(self.values),bool)

    def _fit(self,rows,half):
        p=self.packet;points=p['points'][rows].astype(float);y,x=np.divmod(p['pixels'][rows],self.shape[1])
        patches=np.stack([self.image[y+4+dy,x+4+dx] for dy in range(-half,half+1)
            for dx in range(-half,half+1)],axis=1).astype(float)
        delta=patches-points[:,None]
        valid=np.isfinite(delta).all(2)&(np.linalg.norm(delta,axis=2)<=self.radius)
        count=valid.sum(1);delta=np.nan_to_num(delta)
        centre=(delta*valid[...,None]).sum(1)/np.maximum(count[:,None],1)
        delta=(delta-centre[:,None])*valid[...,None]
        covariance=np.einsum('bki,bkj->bij',delta,delta)/np.maximum(count[:,None,None],1)
        values,vectors=np.linalg.eigh(covariance)
        # Identifiability, not a flatness requirement. Rough ground may still
        # have a defined low incline. Unknown remains unknown, never zero.
        good=(count>=6)&(values[:,2]>0)&(values[:,1]-values[:,0]>1e-6*values[:,2])
        angles=np.arccos(np.clip(abs(vectors[:,:,0]@self.reference),0,1));angles[~good]=np.nan
        return angles

    def slopes(self,rows):
        rows=np.asarray(rows,np.int64);missing=np.unique(rows[~self.computed[rows]])
        for start in range(0,len(missing),2048):
            selected=missing[start:start+2048];angles=self._fit(selected,2)
            unknown=~np.isfinite(angles)
            if unknown.any():angles[unknown]=self._fit(selected[unknown],4)
            self.values[selected]=angles;self.computed[selected]=True
        return self.values[rows]

def prepare_slopes(grid,packet,reference,radius,enabled):
    """Attach one inexpensive source-frame angle to each ground representative."""
    rows=grid['point_rows'];angles=np.full(len(rows),np.nan);selected=np.flatnonzero(rows>=0)
    screen=None
    if enabled:
        screen=SourceSlope(packet,reference,radius,grid.get('selected_ground'));angles[selected]=screen.slopes(rows[selected])
    else:angles[selected]=0.
    grid['slopes']=angles
    return screen

def screen_grid(grid,enabled):
    from .terrain_reasons import UNKNOWN_SLOPE,STEEP
    angles=grid['slopes'].copy();eligible=np.isfinite(grid['cost'].ravel())
    if enabled:
        unknown=eligible&~np.isfinite(angles);steep=eligible&(angles>=np.pi/6)
        grid['reasons'].ravel()[unknown]|=UNKNOWN_SLOPE;grid['reasons'].ravel()[steep]|=STEEP
        grid['cost'].ravel()[unknown|steep]=np.inf
    # Inclination is an exclusion screen, not a dense slope penalty map.
    return angles

def screen_sources(grid,history,reference,radius,enabled):
    """Final maps screen retained representatives once, not every old packet."""
    values=np.full(grid['cost'].size,np.nan);eligible=np.isfinite(grid['cost'].ravel())
    if enabled:
        for slot in history:
            ids=np.flatnonzero(eligible&(grid['source_ticks']==slot['tick']))
            if not len(ids):continue
            screen=SourceSlope(slot['p'],reference,radius,slot.get('ground_mask'))
            values[ids]=screen.slopes(grid['point_rows'][ids])
    else:values[eligible]=0.
    grid['slopes']=values
    return screen_grid(grid,enabled)
