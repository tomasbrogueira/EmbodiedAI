"""Source-indexed bounded temporal evidence; no model initialization."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from orchestrator.visibility import RayEvidence
from path_planner.terrain_reasons import SEMANTIC,GEOMETRY,CLEARANCE,HEIGHT_CONFLICT,UNKNOWN_GROUND

class PacketIndex:
    def __init__(self,p,basis,bin_size):
        self.p=p; self.basis=basis; self.bin_size=bin_size
        xy=np.floor((p['points']@basis)[:,:2]/bin_size).astype(np.int64)
        keys=(xy[:,0]<<32)^(xy[:,1]&0xffffffff)
        self.order=np.argsort(keys,kind='stable')
        sorted_keys=keys[self.order]
        self.keys,self.starts=np.unique(sorted_keys,return_index=True)
        self.ends=np.r_[self.starts[1:],len(keys)]
    def query(self,route,normal,radius,height):
        sweep=np.vstack((route,route+height*normal))
        lower=sweep.min(0)-radius; upper=sweep.max(0)+radius
        xy=sweep@self.basis[:,:2]
        lo=np.floor((xy.min(0)-radius)/self.bin_size).astype(np.int64)
        hi=np.floor((xy.max(0)+radius)/self.bin_size).astype(np.int64)
        keys=np.array([(x<<32)^(y&0xffffffff) for x in range(int(lo[0]),int(hi[0])+1)
            for y in range(int(lo[1]),int(hi[1])+1)],dtype=np.int64)
        locations=np.searchsorted(self.keys,keys)
        keep=locations<len(self.keys); locations,keys=locations[keep],keys[keep]
        locations=locations[self.keys[locations]==keys]
        rows=np.concatenate([self.order[self.starts[i]:self.ends[i]] for i in locations]) if len(locations) else np.array([],dtype=np.int64)
        points=self.p['points'][rows]
        rows=rows[((points>=lower)&(points<=upper)).all(1)]
        return np.sort(rows)
class StableTerrain:
    """Retain actual source points, with separate costs, vetoes and source IDs.

    All source witnesses expire after 20 packets. Negative observations persist while their cells remain in the rolling window;
    positive-only observations never clear a negative veto. Untyped occupancy
    needs two newer depth rays through its old location before recovery.
    """
    def __init__(self, capacity=20,layer_filter=False):
        self.capacity=capacity; self.state=None;self.ray=None;self.layer_filter=layer_filter

    def update(self, fresh, p, tick, radius, height):
        w=fresh['cost'].shape[0]; shape=(w,w)
        if self.state is None:
            state=dict(cost=np.full(shape,np.inf), height=np.full(shape,np.nan),
                points=np.full((w,w,3),np.nan), rows=np.full(shape,-1,np.int64),
                sources=np.full(shape,-1,np.int64), veto=np.full(shape,-100000,np.int64),
                occupied=np.full(shape,-100000,np.int64),conflict=np.full(shape,-100000,np.int64),
                confirmed=np.full(shape,-1,np.int64),slopes=np.full(shape,np.nan),
                frames=np.full(shape,-1,np.int64),occupied_points=np.full((w,w,3),np.nan),
                free_count=np.zeros(shape,np.uint8),free_last=np.full(shape,-1,np.int64),cleared_at=np.full(shape,-1,np.int64),
                layer_count=np.zeros(shape,np.uint8),layer_last=np.full(shape,-1,np.int64))
        else:
            old=self.state; dx,dy=map(int,fresh['origin']-old['origin'])
            state={}
            for key,fill in [('cost',np.inf),('height',np.nan),('points',np.nan),
                             ('rows',-1),('sources',-1),('veto',-100000),('occupied',-100000),('conflict',-100000),('confirmed',-1),
                             ('slopes',np.nan),('frames',-1),('occupied_points',np.nan),('free_count',0),('free_last',-1),('cleared_at',-1),('layer_count',0),('layer_last',-1)]:
                a=np.roll(old[key],(-dy,-dx),axis=(0,1)).copy()
                if abs(dx)>=w or abs(dy)>=w: a[:]=fill
                else:
                    if dy>0: a[-dy:]=fill
                    elif dy<0: a[:-dy]=fill
                    if dx>0: a[:,-dx:]=fill
                    elif dx<0: a[:,:-dx]=fill
                state[key]=a
        expired=state['sources']<=tick-self.capacity
        state['cost'][expired]=np.inf; state['height'][expired]=np.nan
        state['points'][expired]=np.nan; state['rows'][expired]=-1; state['sources'][expired]=-1
        state['slopes'][expired]=np.nan;state['frames'][expired]=-1
        state['layer_count'][expired]=0;state['layer_last'][expired]=-1
        new=np.isfinite(fresh['cost']); old=np.isfinite(state['cost'])
        conflict=new&old&(abs(fresh['height']-state['height'])>radius/2)
        lower_ignored=np.zeros(shape,bool)
        frame=int(p.get('source_index',tick));tolerance=fresh.get('layer_tolerance',radius/4)
        if self.layer_filter:
            # Do not let a lower duplicate overwrite a still-observed upper
            # surface. Looking from under that surface permits lower ground.
            lower_ignored=new&old&(fresh['height']<state['height']-tolerance)&(float(p['camera']@fresh['basis'][:,2])>state['height'])
            new[lower_ignored]=False
            changed=new&old&(abs(fresh['height']-state['height'])>tolerance)
            distinct=new&(state['layer_last']!=frame)
            state['layer_count'][new&(~old|changed)]=0
            state['layer_count'][distinct]=np.minimum(state['layer_count'][distinct]+1,2)
            state['layer_last'][distinct]=frame
            conflict=changed|(new&old&(state['layer_count']<2))
        # Refresh all compatible observations together. Keeping the first point
        # forever mixed old and new frame heights into apparent steep steps.
        take=new
        state['cost'][take]=fresh['cost'][take]; state['height'][take]=fresh['height'][take]
        state['points'][take]=fresh['points'].reshape(w,w,3)[take]
        state['rows'][take]=fresh['point_rows'].reshape(shape)[take]; state['sources'][take]=tick
        state['slopes'][take]=fresh.get('slopes',np.zeros(w*w)).reshape(shape)[take]
        state['frames'][take]=int(p.get('source_index',tick))
        state['conflict'][conflict]=tick
        state['confirmed'][new&~conflict]=tick
        xy=np.floor((p['points']@fresh['basis'])[:,:2]/fresh['cell']).astype(int)-fresh['origin']
        inside=((xy>=0)&(xy<w)).all(1); ids=xy[inside,1]*w+xy[inside,0]
        z=(p['points'][inside]@fresh['basis'])[:,2]; ground=state['height'].ravel()[ids]
        body=np.isfinite(ground)&(z>=ground-radius/20)&(z<=ground+height+radius/20)
        state['veto'].ravel()[np.unique(ids[p['negative'][inside]])]=tick
        # Floor-level classification and height conflicts are transient;
        # explicit negatives and elevated geometry remain conservative vetoes.
        ground_mask=fresh.get('selected_ground',p['positive'])
        if self.layer_filter:
            ground_mask=ground_mask.copy();source_rows=np.flatnonzero(inside)
            mismatch=np.isfinite(ground)&(abs(z-ground)>tolerance)
            ground_mask[source_rows[mismatch]]=False
        out_ground_mask=ground_mask
        elevated=(~ground_mask[inside])&body&(z>ground+radius/20)
        occupied_ids=np.unique(ids[elevated]);state['occupied'].ravel()[occupied_ids]=tick
        if len(occupied_ids):
            highest=np.full(w*w,-np.inf);np.maximum.at(highest,ids[elevated],z[elevated])
            source_rows=np.flatnonzero(inside);selected=np.full(w*w,len(p['points']),np.int64)
            match=elevated&(z==highest[ids]);np.minimum.at(selected,ids[match],source_rows[match])
            state['occupied_points'].reshape(-1,3)[occupied_ids]=p['points'][selected[occupied_ids]]
            state['free_count'].ravel()[occupied_ids]=0;state['free_last'].ravel()[occupied_ids]=-1
            state['cleared_at'].ravel()[occupied_ids]=-1
        self.ray=None
        candidates=np.flatnonzero(((state['occupied']>=0)&(state['occupied']<tick)&new&~conflict).ravel())
        witnessed=np.zeros(w*w,bool)
        if len(candidates):
            self.ray=RayEvidence(p,ground_mask)
            witnessed[candidates]=self.ray.clears(state['occupied_points'].reshape(-1,3)[candidates],radius/20)
        frame=int(p.get('source_index',tick));last=state['free_last'].ravel()
        distinct=witnessed&(last!=frame)
        prior=state['free_count'].ravel();prior[~witnessed]=0;prior[distinct]=np.minimum(prior[distinct]+1,2)
        last[distinct]=frame
        recovered=np.flatnonzero(witnessed&(prior>=2))
        state['occupied'].ravel()[recovered]=-100000;state['cleared_at'].ravel()[recovered]=tick
        floor_obstacle=np.bincount(ids[(~ground_mask[inside])&body&~elevated],minlength=w*w)>0
        ambiguous=(fresh.get('layer_filter',False)&~np.isfinite(fresh['cost'])&old)
        # A currently ambiguous column may not resurrect its previous lower
        # representative as usable ground.
        mixed_rows=np.flatnonzero(inside&p['positive']&~p['negative'])
        mixed_ids=np.floor((p['points'][mixed_rows]@fresh['basis'])[:,:2]/fresh['cell']).astype(int)-fresh['origin']
        observed_now=np.zeros(w*w,bool);observed_now[mixed_ids[:,1]*w+mixed_ids[:,0]]=True
        ambiguous &= observed_now.reshape(shape)
        blocked=(state['veto']>=0)|(state['occupied']>=0)|floor_obstacle.reshape(shape)|~np.isfinite(state['cost'])|(state['conflict']==tick)|ambiguous
        clearance=distance_transform_edt(~blocked,sampling=fresh['cell'])
        valid=~blocked&(clearance>radius/20+fresh['cell']/np.sqrt(2))
        reasons=np.zeros(shape,np.uint8)
        reasons[state['veto']>=0]|=SEMANTIC
        reasons[(state['occupied']>=0)|floor_obstacle.reshape(shape)]|=GEOMETRY
        reasons[(state['conflict']==tick)|ambiguous]|=HEIGHT_CONFLICT
        reasons[~np.isfinite(state['cost'])]|=UNKNOWN_GROUND
        reasons[~blocked&~valid]|=CLEARANCE
        out=dict(fresh)
        out['cost']=np.where(valid,1+np.exp(-np.minimum(clearance,radius)/radius),np.inf)
        out['height']=state['height'].copy(); out['points']=state['points'].reshape(-1,3).copy()
        out['hard_veto']=((state['veto']>=0)|(state['occupied']>=0)).copy()
        out['point_rows']=state['rows'].ravel().copy(); out['source_ticks']=state['sources'].ravel().copy()
        out['confirmed_ticks']=state['confirmed'].ravel().copy()
        out['slopes']=state['slopes'].ravel().copy();out['source_frames']=state['frames'].ravel().copy()
        out['cleared_at']=state['cleared_at'].ravel().copy();out['reasons']=reasons
        out['search_stride']=max(1,min(4,int(radius/(2*fresh['cell']))))
        out['layer_filter']=self.layer_filter;out['layer_tolerance']=tolerance
        out['layer_confirmed']=state['layer_count']>=2
        out['lower_layer_updates_ignored']=int(lower_ignored.sum())
        out['selected_ground']=out_ground_mask
        out['goal_observed']=(new&~conflict).ravel()
        state['origin']=fresh['origin'].copy(); self.state=state
        assert not np.isfinite(out['cost'][state['veto']>=0]).any()
        return out
