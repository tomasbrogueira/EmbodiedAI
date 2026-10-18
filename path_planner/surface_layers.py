"""Observed ground-layer selection and filtering; original coordinates stay intact."""
import numpy as np

def column_keys(xy):
    xy=np.asarray(xy,dtype=np.int64)
    return (xy[:,0]<<32)^(xy[:,1]&0xffffffff)

class GroundAtlas:
    """Bounded confirmed ground outside the rolling window, for map display.

    This is not planning support: raw source witnesses still expire normally.
    Store actual observed heights; never fill holes or average stacked layers.
    """
    def __init__(self,capacity=250000):
        self.columns={};self.capacity=capacity

    def update(self,grid,camera):
        w=len(grid['height']);ids=np.flatnonzero(grid['layer_confirmed'].ravel()&np.isfinite(grid['height'].ravel()))
        xy=np.c_[ids%w,ids//w]+grid['origin'];keys=column_keys(xy)
        camera_height=float(camera@grid['basis'][:,2]);tol=grid['layer_tolerance']
        for key,height in zip(keys,grid['height'].ravel()[ids]):
            key=int(key);old=self.columns.get(key)
            if old is None or height>=old-tol or camera_height<old:
                self.columns.pop(key,None);self.columns[key]=float(height)
        while len(self.columns)>self.capacity:self.columns.pop(next(iter(self.columns)))

    def overlay(self):
        keys=np.array(sorted(self.columns),np.int64)
        return dict(layer_atlas_keys=keys,layer_atlas_height=np.array([self.columns[int(k)] for k in keys]))

def surface_heights(points,surface):
    """Current confirmed columns take precedence; atlas covers older windows."""
    heights=np.full(len(points),np.nan)
    if not surface.get('layer_filter',False):return heights
    xyz=points@surface['basis'];xy=np.floor(xyz[:,:2]/surface['cell']).astype(np.int64)
    keys=surface.get('atlas_keys',np.empty(0,np.int64))
    if len(keys):
        queries=column_keys(xy);locations=np.searchsorted(keys,queries);rows=np.flatnonzero(locations<len(keys))
        rows=rows[keys[locations[rows]]==queries[rows]]
        heights[rows]=surface['atlas_height'][locations[rows]]
    local=xy-surface['origin'];w=len(surface['height']);rows=np.flatnonzero(((local>=0)&(local<w)).all(1))
    cells=local[rows,1]*w+local[rows,0];ground=surface['height'].ravel()[cells]
    take=surface['layer_confirmed'].ravel()[cells]&np.isfinite(ground)
    heights[rows[take]]=ground[take]
    return heights

def select_ground_layers(grid,p,radius):
    """Pick a supported upper ground cluster, not an average between layers.

    Only SAM-positive clusters with >=3 witnesses and >=25% of the largest
    cluster's support qualify. Layers above the camera cannot displace observed
    ground below it (e.g. when looking underneath an overpass).
    """
    w=len(grid['cost']);xyz=p['points']@grid['basis'];xy=np.floor(xyz[:,:2]/grid['cell']).astype(int)-grid['origin']
    inside=((xy>=0)&(xy<w)).all(1);rows=np.flatnonzero(inside&p['positive']&~p['negative'])
    active=(p['positive']&~p['negative']).copy();active[inside]=False
    heights=np.full(w*w,np.nan);selected=np.full(w*w,-1,np.int64);cost=np.full(w*w,np.inf)
    if len(rows):
        ids=xy[rows,1]*w+xy[rows,0];z=xyz[rows,2];order=np.lexsort((z,ids));rows,ids,z=rows[order],ids[order],z[order]
        starts=np.r_[0,np.flatnonzero((ids[1:]!=ids[:-1])|(np.diff(z)>radius/4))+1]
        ends=np.r_[starts[1:],len(rows)];count=ends-starts;cell_ids=ids[starts]
        means=np.add.reduceat(z,starts)/count
        largest=np.zeros(w*w,int);np.maximum.at(largest,cell_ids,count)
        eligible=(count>=3)&(count>=largest[cell_ids]/4)&(z[ends-1]-z[starts]<=radius/2)
        camera_height=float(p['camera']@grid['basis'][:,2]);below=means<camera_height
        has_below=np.zeros(w*w,bool);np.logical_or.at(has_below,cell_ids,eligible&below)
        eligible &= below|~has_below[cell_ids]
        top=np.full(w*w,-np.inf);np.maximum.at(top,cell_ids[eligible],means[eligible])
        chosen=eligible&(means==top[cell_ids]);groups=np.repeat(np.arange(len(starts)),count);keep=chosen[groups]
        active[rows[keep]]=True
        centres=np.c_[(ids%w+grid['origin'][0]+.5)*grid['cell'],(ids//w+grid['origin'][1]+.5)*grid['cell'],means[groups]]
        error=np.sum((xyz[rows]-centres)**2,axis=1);best=np.full(w*w,np.inf);np.minimum.at(best,ids[keep],error[keep])
        take=keep&(error==best[ids]);chosen_rows=np.full(w*w,len(p['points']),np.int64);np.minimum.at(chosen_rows,ids[take],rows[take])
        good=chosen_rows<len(p['points']);selected[good]=chosen_rows[good];heights[good]=xyz[chosen_rows[good],2];cost[good]=1.
    points=np.full((w*w,3),np.nan);good=selected>=0;points[good]=p['points'][selected[good]]
    grid.update(cost=cost.reshape(w,w),height=heights.reshape(w,w),points=points,point_rows=selected,
        selected_ground=active,layer_filter=True,layer_tolerance=radius/4)
    return grid

def obsolete_lower(points,surface):
    """Remove only points below an independently confirmed selected surface."""
    out=np.zeros(len(points),bool)
    if not surface.get('layer_filter',False):return out
    ground=surface_heights(points,surface)
    return np.isfinite(ground)&((points@surface['basis'][:,2])<ground-surface['layer_tolerance'])

def obsolete_ground(points,positive,surface):
    """Typed ground copies above or below the chosen ground; objects stay."""
    ground=surface_heights(points,surface)
    return positive&np.isfinite(ground)&(abs(points@surface['basis'][:,2]-ground)>surface['layer_tolerance'])

def overlay_surface(grid):
    return dict(layer_filter=np.array(bool(grid.get('layer_filter',False))),layer_height=grid['height'],
        layer_confirmed=grid.get('layer_confirmed',np.zeros_like(grid['cost'],bool)),
        layer_basis=grid['basis'],layer_origin=grid['origin'],layer_cell=np.array(grid['cell']),
        layer_tolerance=np.array(grid.get('layer_tolerance',0.)))

def surface_from_overlay(data):
    if data is None or 'layer_height' not in data:return dict(layer_filter=False)
    return dict(layer_filter=bool(data['layer_filter']),height=data['layer_height'],layer_confirmed=data['layer_confirmed'],
        basis=data['layer_basis'],origin=data['layer_origin'],cell=float(data['layer_cell']),layer_tolerance=float(data['layer_tolerance']),
        atlas_keys=data.get('layer_atlas_keys',np.empty(0,np.int64)),atlas_height=data.get('layer_atlas_height',np.empty(0)))
