"""Bounded local terrain costs and native routing; independent of models/viewer."""
import ctypes
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from .cost_map import CostMap, _settings
from .slope import _fit_patches

def milliseconds(start): return (time.perf_counter()-start)*1000

def native_library():
    """Compile a source-keyed local library once; reuse it across viewer runs."""
    import hashlib
    import os
    import subprocess
    import tempfile
    source=Path(__file__).with_name('local_grid_search.cpp')
    key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory=Path.home()/'.cache/embodiedai/native'
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/f'local_grid_{key}.so'
    if not target.exists():
        fd,name=tempfile.mkstemp(suffix='.so',dir=directory); os.close(fd)
        try:
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',name],
                           check=True,capture_output=True,text=True)
            os.replace(name,target)
        finally:
            Path(name).unlink(missing_ok=True)
    return target

def basis_for(normal):
    axis=np.eye(3)[np.argmin(abs(normal))]
    x=axis-(axis@normal)*normal; x/=np.linalg.norm(x)
    return np.column_stack((x,np.cross(normal,x),normal))

def organized(points,pixels,shape,reference,radius,method):
    """Small same-image patches, without KD neighbour search or robust refit."""
    h,w=shape
    image=np.full((h,w,3),np.nan); image.reshape(-1,3)[pixels]=points
    y,x=np.divmod(pixels,w)
    pad=np.pad(image,((1,1),(1,1),(0,0)),constant_values=np.nan)
    if method=='cross':
        right,left=pad[y+1,x+2],pad[y+1,x]
        down,up=pad[y+2,x+1],pad[y,x+1]
        normals=np.cross(right-left,down-up)
        length=np.linalg.norm(normals,axis=1)
        valid=np.isfinite(normals).all(1)&(length>1e-12)
        for neighbour in (right,left,down,up):
            valid &= np.linalg.norm(neighbour-points,axis=1)<=radius
        normals/=np.maximum(length[:,None],1e-12)
        quality=valid.astype(float)
    else:
        patches=np.stack([pad[y+1+dy,x+1+dx] for dy in (-1,0,1) for dx in (-1,0,1)],axis=1)
        valid=np.isfinite(patches).all(2)&(np.linalg.norm(patches-points[:,None],axis=2)<=radius)
        count=valid.sum(1)
        patches=np.nan_to_num(patches-points[:,None])
        centre=(patches*valid[...,None]).sum(1)/np.maximum(count[:,None],1)
        delta=(patches-centre[:,None])*valid[...,None]
        covariance=np.einsum('bki,bkj->bij',delta,delta)/np.maximum(count[:,None,None],1)
        values,vectors=np.linalg.eigh(covariance)
        normals=vectors[:,:,0]
        good=(count>=6)&(values[:,2]>0)&(values[:,1]-values[:,0]>1e-6*values[:,2])
        good &= values[:,0]<=.05*np.maximum(values[:,1],1e-20)
        quality=np.where(good,1.,0.)
    slope=np.arccos(np.clip(abs(normals@reference),0,1))
    slope[quality==0]=np.nan
    return slope,quality

class Search:
    def __init__(self):
        self.library=ctypes.CDLL(str(native_library()))
        self.function=self.library.grid_search
        self.function.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_double]*2+[
            ctypes.c_int,ctypes.c_double,ctypes.c_int]+[ctypes.c_void_p]*4
        self.function.restype=ctypes.c_int
    def __call__(self,cost,height,start,cell,enforce,budget=40.,*,max_stride=1):
        width=cost.shape[0]; cost=np.ascontiguousarray(cost,dtype='float64')
        height=np.ascontiguousarray(height,dtype='float64')
        distance=np.empty(cost.size); previous=np.empty(cost.size,np.int32); settled=np.empty(cost.size,np.uint8)
        expanded=np.zeros(1,np.int32)
        status=self.function(cost.ctypes.data,height.ctypes.data,width,int(start),cell,np.pi/6,
            int(enforce),budget,int(max_stride),distance.ctypes.data,previous.ctypes.data,settled.ctypes.data,expanded.ctypes.data)
        return status,distance,previous,settled.astype(bool),int(expanded[0])

def terrain(points,positive,negative,slopes,camera,basis,radius,height,width=200,divisor=4,enforce=True):
    """Conservative aggregation: no unknown filling and no negative averaging."""
    cell=radius/divisor
    xyz=points@basis
    origin=np.floor((camera@basis)[:2]/cell).astype(int)-width//2
    xy=np.floor(xyz[:,:2]/cell).astype(int)-origin
    inside=((xy>=0)&(xy<width)).all(1)
    source=np.flatnonzero(inside)
    xy=xy[inside]; z=xyz[inside,2]
    ids=xy[:,1]*width+xy[:,0]; total=width*width
    pos=positive[inside]&~negative[inside]
    # Unknown slope never becomes a zero-slope observation.
    usable=pos&(np.isfinite(slopes[inside])&(slopes[inside]<np.pi/6) if enforce else True)
    count=np.bincount(ids[usable],minlength=total)
    sums=np.bincount(ids[usable],weights=z[usable],minlength=total)
    surface=np.divide(sums,count,out=np.full(total,np.nan),where=count>0)
    low,high=np.full(total,np.inf),np.full(total,-np.inf)
    np.minimum.at(low,ids[usable],z[usable]); np.maximum.at(high,ids[usable],z[usable])
    # Reject mixed/uncertain surfaces rather than flattening two levels.
    observed=(count>=3)&(high-low<=radius/2)
    veto=np.bincount(ids[negative[inside]],minlength=total)>0
    non_support=(~positive[inside])|negative[inside]
    near_body=np.isfinite(surface[ids])&(z>=surface[ids]-radius/20)&(z<=surface[ids]+height+radius/20)
    obstacle=np.bincount(ids[non_support&near_body],minlength=total)>0
    blocked=veto|obstacle|~observed
    margin=radius/20+cell/np.sqrt(2)
    clearance=distance_transform_edt((~blocked).reshape(width,width),sampling=cell).ravel()
    eligible=observed&~blocked&(clearance>margin)
    node_slope=np.zeros(total)
    if enforce:
        np.maximum.at(node_slope,ids[usable],slopes[inside][usable])
    cost=np.full(total,np.inf)
    cost[eligible]=1+(node_slope[eligible]/(np.pi/6))**2*(1 if enforce else 0)+np.exp(-np.minimum(clearance[eligible],radius)/radius)
    yy,xx=np.indices((width,width)); locations=np.c_[
        ((xx.ravel()+origin[0])+.5)*cell,((yy.ravel()+origin[1])+.5)*cell,surface]@basis.T
    negative_preserved=bool(not np.isfinite(cost[veto]).any())
    return dict(cost=cost.reshape(width,width),height=surface.reshape(width,width),points=locations,
        cell=cell,origin=origin,veto_cells=int(veto.sum()),negative_preserved=negative_preserved,
        eligible_cells=int(eligible.sum()),source_rows=source,units='reconstruction units',basis=basis)

def certify(points,positive,negative,slopes,route,normal,radius,height,enforce,diagnostics=None):
    """Use original raw support-gap and swept-obstacle predicates on each edge."""
    if len(route)<2: return False,0.
    t=time.perf_counter()
    eligible=positive&~negative
    if enforce: eligible &= np.isfinite(slopes)&(slopes<np.pi/6)
    if not eligible.any(): return False,milliseconds(t)
    c=CostMap.__new__(CostMap)
    c.points=np.vstack((points,route)); c.reference=normal
    c.config=_settings(dict(radius=radius,clearance_height=height,slope_policy='enforce' if enforce else 'disabled'))
    c.obstacle_rows=np.flatnonzero(~positive|negative)
    c.obstacles=cKDTree(points[c.obstacle_rows]); c.support=cKDTree(np.unique(points[eligible],axis=0))
    c._index_overhead()
    node_valid=c.support.query(route,k=[c.config['min_support']],distance_upper_bound=radius)[0][:,0]<=radius
    if diagnostics is None:
        valid=node_valid.all() and all(c._connected(len(points)+i,len(points)+i+1) for i in range(len(route)-1))
    else:
        edges=np.asarray([c._connected(len(points)+i,len(points)+i+1) for i in range(len(route)-1)])
        diagnostics.update(failed_nodes=np.flatnonzero(~node_valid).tolist(),failed_edges=np.flatnonzero(~edges).tolist())
        valid=node_valid.all() and edges.all()
    return bool(valid),milliseconds(t)
def representatives(grid,p):
    cell=grid['cell']; width=grid['cost'].shape[0]
    xy=np.floor((p['points']@grid['basis'])[:,:2]/cell).astype(int)-grid['origin']
    take=((xy>=0)&(xy<width)).all(1)&p['positive']&~p['negative']
    rows=np.flatnonzero(take); ids=xy[rows,1]*width+xy[rows,0]
    keep=np.isfinite(grid['cost'].ravel()[ids]); rows,ids=rows[keep],ids[keep]
    errors=np.einsum('ij,ij->i',p['points'][rows]-grid['points'][ids],p['points'][rows]-grid['points'][ids])
    best=np.full(grid['cost'].size,np.inf); np.minimum.at(best,ids,errors)
    selected=np.full(grid['cost'].size,len(p['points']),np.int64)
    pick=errors==best[ids]; np.minimum.at(selected,ids[pick],rows[pick])
    present=selected<len(p['points'])
    grid['points'][present]=p['points'][selected[present]]
    grid['height'].ravel()[present]=grid['points'][present]@grid['basis'][:,2]
    selected[~present]=-1; grid['point_rows']=selected
    return grid
def grid_pca(grid,normal,radius):
    valid=np.flatnonzero(np.isfinite(grid['cost'].ravel()))
    surface_points=grid['points'][valid]
    slope,_=organized(surface_points,valid,grid['cost'].shape,normal,radius,'pca')
    full=np.full(grid['cost'].size,np.nan); full[valid]=slope
    reject=~np.isfinite(slope)|(slope>=np.pi/6)
    grid['cost'].ravel()[valid[reject]]=np.inf
    good=valid[~reject]
    grid['cost'].ravel()[good]+=(full[good]/(np.pi/6))**2
    grid['eligible_cells']=int(np.isfinite(grid['cost']).sum())
    return full

def raw_node_slopes(points,grid,full):
    xy=np.floor((points@grid['basis'])[:,:2]/grid['cell']).astype(int)-grid['origin']
    keep=((xy>=0)&(xy<grid['cost'].shape[0])).all(1)
    values=np.full(len(points),np.nan)
    values[keep]=full[xy[keep,1]*grid['cost'].shape[0]+xy[keep,0]]
    return values
class Refiner:
    def __init__(self,points,normal,radius):
        self.points=np.asarray(points,dtype=float); self.normal=normal; self.radius=radius
        self.tree=cKDTree(self.points); self.canonical=False; self.cache={}
    def slopes(self,rows):
        missing=[int(row) for row in rows if int(row) not in self.cache]
        if missing:
            q=self.points[missing]
            k=min(128,len(self.tree.data))
            distance,index=self.tree.query(q,k=k,distance_upper_bound=np.nextafter(self.radius,np.inf))
            valid=np.isfinite(distance)
            neighbour=self.tree.data[np.minimum(index,len(self.tree.data)-1)]
            if not self.canonical and any(len(np.unique(p[v],axis=0))<v.sum() for p,v in zip(neighbour,valid)):
                self.tree=cKDTree(np.unique(self.points,axis=0)); self.canonical=True; self.cache={}
                return self.slopes(rows)
            patch=(neighbour-q[:,None])/self.radius; patch[~valid]=0
            normals,quality=_fit_patches(patch,valid,6)
            angles=np.arccos(np.clip(abs(normals@self.normal),0,1))
            self.cache.update(zip(missing,angles.tolist()))
        return np.asarray([self.cache[int(row)] for row in rows])
def fresh_ground(p,camera,basis,radius,height,*,width=200,divisor=8,layer_filter=False):
    """Retain observed support before clearance inflation, applied once later."""
    grid=terrain(p['points'],p['positive'],p['negative'],np.zeros(len(p['points'])),
        camera,basis,radius,height,width=width,divisor=divisor,enforce=False)
    w=grid['cost'].shape[0]
    xyz=p['points']@basis; xy=np.floor(xyz[:,:2]/grid['cell']).astype(int)-grid['origin']
    take=((xy>=0)&(xy<w)).all(1)&p['positive']&~p['negative']
    ids=xy[take,1]*w+xy[take,0]; z=xyz[take,2]
    count=np.bincount(ids,minlength=w*w)
    lo=np.full(w*w,np.inf); hi=np.full(w*w,-np.inf)
    np.minimum.at(lo,ids,z); np.maximum.at(hi,ids,z)
    grid['cost']=np.where((count>=3)&(hi-lo<=radius/2),1.,np.inf).reshape(w,w)
    representatives(grid,p)
    grid['radius']=radius
    if layer_filter:
        from .surface_layers import select_ground_layers
        select_ground_layers(grid,p,radius)
    return grid

class PathState:
    def __init__(self): self.path=None; self.committed=0; self.extensions=0; self.replans=0
    def splice(self, anchor, suffix):
        if self.path is None:
            self.path=suffix.copy(); return 'initial'
        if anchor<self.committed or not np.array_equal(self.path[anchor],suffix[0]):
            raise ValueError('Missing shared anchor or attempted history rewrite')
        old=self.path.copy(); self.path=np.vstack((old[:anchor],suffix))
        if not np.array_equal(self.path[:anchor+1],old[:anchor+1]):
            raise RuntimeError('Connected prefix changed during splice')
        self.committed=anchor
        if anchor==len(old)-1: self.extensions+=1; return 'extension'
        self.replans+=1; return 'suffix_replan'
