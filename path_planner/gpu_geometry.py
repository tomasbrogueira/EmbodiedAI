"""Optional GPU support/clearance validator retaining original FP64 geometry."""
from pathlib import Path
import time
import numpy as np
import cupy as cp

HERE=Path(__file__).resolve().parent
CODE=(HERE/'gpu_geometry.cu').read_text()
OPTIONS=('--std=c++17','--fmad=false')
SUPPORT=cp.RawKernel(CODE,'support_check',options=OPTIONS)
OBSTACLE=cp.RawKernel(CODE,'obstacle_check',options=OPTIONS)

def sync(): cp.cuda.get_current_stream().synchronize()
def elapsed(t): sync(); return (time.perf_counter()-t)*1000

class GPUIndex:
    def __init__(self,points,positive,negative,slopes,radius,enforce):
        self.radius=radius; self.cell=radius/4
        t=time.perf_counter()
        p=cp.asarray(points,dtype=cp.float64)
        positive=cp.asarray(positive); negative=cp.asarray(negative); slope=cp.asarray(slopes)
        labels=(positive&~negative&(cp.isfinite(slope)&(slope<np.pi/6) if enforce else True)).astype(cp.uint8)
        labels|=((~positive)|negative).astype(cp.uint8)*2
        self.upload_ms=elapsed(t); t=time.perf_counter()
        self.n=len(points)
        self.origin=cp.floor(p.min(axis=0)/self.cell)*self.cell if self.n else cp.zeros(3,dtype=cp.float64)
        xyz=cp.floor((p-self.origin)/self.cell).astype(cp.int64)
        self.dims=xyz.max(axis=0)+1 if self.n else cp.ones(3,dtype=cp.int64)
        keys=(xyz[:,0]*self.dims[1]+xyz[:,1])*self.dims[2]+xyz[:,2]
        order=cp.argsort(keys)
        self.keys=keys[order]; self.points=p[order]; self.labels=labels[order]
        self.build_ms=elapsed(t)
        self.bytes=sum(a.nbytes for a in (self.keys,self.points,self.labels,self.origin,self.dims))
        self.base=(self.points,self.labels,self.keys,np.int64(self.n),self.origin,self.dims,np.float64(self.cell))

    def check(self,route,normal,height,enforce):
        if len(route)<2: return False,dict(total_ms=0.,device_ms=0.)
        t=time.perf_counter(); route=np.asarray(route,dtype=np.float64); normal=np.asarray(normal,dtype=np.float64)
        delta=np.diff(route,axis=0); length=np.linalg.norm(delta,axis=1)
        if enforce:
            vertical=delta@normal; horizontal=np.linalg.norm(delta-vertical[:,None]*normal,axis=1)
            if ((length>0)&(abs(vertical)>=horizontal*np.tan(np.pi/6))).any():
                bad=(length>0)&(abs(vertical)>=horizontal*np.tan(np.pi/6))
                return False,dict(total_ms=(time.perf_counter()-t)*1000,device_ms=0.,grade_rejected=True,failed_edges=np.flatnonzero(bad).tolist(),failed_nodes=[])
        queries=[]; radii=[]; query_edges=[]
        for edge,(a,d,l) in enumerate(zip(route[:-1],delta,length)):
            if l==0: continue
            steps=max(1,int(np.ceil(l/(self.radius/16))))
            padding=l/(2*steps)
            q=a+(np.arange(steps+1)/steps)[:,None]*d
            queries.append(q); radii.append(np.full(len(q),self.radius/8-padding))
            query_edges.append(np.full(len(q),edge,np.int64))
        queries=np.concatenate(queries) if queries else np.empty((0,3))
        radii=np.concatenate(radii) if radii else np.empty(0)
        gpu_route=cp.asarray(route); gpu_queries=cp.asarray(queries); gpu_radii=cp.asarray(radii)
        node_radii=cp.full(len(route),self.radius,dtype=cp.float64); gpu_normal=cp.asarray(normal)
        nodes=cp.empty(len(route),dtype=cp.uint8); gaps=cp.empty(len(queries),dtype=cp.uint8); edges=cp.empty(len(route)-1,dtype=cp.uint8)
        prepare_ms=elapsed(t); t=time.perf_counter(); begin=cp.cuda.Event(); end=cp.cuda.Event(); begin.record()
        SUPPORT(((len(route)+127)//128,),(128,),self.base+(gpu_route,node_radii,np.int32(len(route)),np.int32(6),nodes))
        if len(queries): SUPPORT(((len(queries)+127)//128,),(128,),self.base+(gpu_queries,gpu_radii,np.int32(len(queries)),np.int32(1),gaps))
        OBSTACLE(((len(edges)+127)//128,),(128,),self.base+(gpu_route,np.int32(len(edges)),gpu_normal,np.float64(height),np.float64(self.radius/20),edges))
        answer=cp.all(nodes)&cp.all(gaps)&cp.all(edges); end.record(); valid=bool(answer.get())
        kernel_wall_ms=elapsed(t); device_ms=cp.cuda.get_elapsed_time(begin,end)
        metrics=dict(total_ms=prepare_ms+kernel_wall_ms,prepare_ms=prepare_ms,
            kernel_wall_ms=kernel_wall_ms,device_ms=device_ms,route_points=len(route),gap_queries=len(queries))
        if not valid:
            failed_edges=np.flatnonzero(~edges.get().astype(bool))
            if query_edges:
                edge_ids=np.concatenate(query_edges); failed_edges=np.r_[failed_edges,edge_ids[~gaps.get().astype(bool)]]
            metrics.update(failed_nodes=np.flatnonzero(~nodes.get().astype(bool)).tolist(),failed_edges=np.unique(failed_edges).tolist())
        return valid,metrics
