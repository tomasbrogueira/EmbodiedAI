"""Live cost, slope, goal and connected-route overlays; no planning logic."""
import numpy as np
from path_planner.terrain_reasons import (SEMANTIC,GEOMETRY,CLEARANCE,HEIGHT_CONFLICT,
    UNKNOWN_SLOPE,STEEP,RETRY,UNKNOWN_GROUND,LABELS)
from path_planner.surface_layers import obsolete_lower,surface_from_overlay

def terrain_colors(data,colour):
    finite=np.isfinite(data['costs']);slopes=data['slopes']
    reasons=data.get('reasons')
    if reasons is None:
        reasons=np.where(finite,0,GEOMETRY).astype(np.uint8)
        if bool(data['slopes_enabled']):
            reasons[~finite&~np.isfinite(slopes)]=UNKNOWN_SLOPE
            reasons[~finite&(slopes>=np.pi/6)]=STEEP
    colours=np.full((len(finite),3),(135,145,155),np.uint8)
    colours[finite]=(30,210,90)
    if colour=='Local costs':
        t=np.clip(data['costs'][finite]-1,0,1)
        colours[finite]=np.c_[30+190*t,210-60*t,90-30*t].astype(np.uint8)
    elif colour=='Local slopes':
        good=finite&np.isfinite(slopes);t=np.clip(slopes[good]/(np.pi/6),0,1)
        colours[good]=np.c_[30+190*t,210-60*t,90-30*t].astype(np.uint8)
    colours[(reasons&CLEARANCE)!=0]=(195,110,105)
    colours[(reasons&STEEP)!=0]=(235,155,35)
    colours[(reasons&RETRY)!=0]=(135,85,125)
    colours[(reasons&(SEMANTIC|GEOMETRY))!=0]=(230,50,50)
    return colours

def route_message(metadata,vertices=0,has_history=False):
    status=metadata.get('status','waiting')
    if metadata.get('active_valid'):
        partial=status in ('partial','partial_ready')
        return f"**Validated {'partial 'if partial else ''}path: {vertices} points.**"+(' Stops before an unresolved connection.' if partial else '')+(
            f" Ground layer corrected; new planning epoch {metadata.get('path_epoch',0)}." if metadata.get('layer_corrected') else '')+(
            f" New local segment (epoch {metadata.get('path_epoch',0)}); previous connection unavailable." if metadata.get('continuity_restart') else '')
    explanations={
        'waiting_for_reference':'finding a stable ground reference',
        'geometry_rejected':'the candidate fails support, clearance or grade checks',
        'connection_lost':'observed ground no longer connects to the retained path',
        'no_goal_ahead':'no reachable forward goal in the observed ground',
        'no_useful_forward_component':'no useful forward ground within the planning window',
        'no_observed_ground':'no usable ground observations',
        'no_current_ground':'ground directly below the camera has not been observed',
        'current_ground_blocked':'ground below the camera is excluded',
        'updating':'the newest frame is being planned',
        'incomplete_required_semantics':'required traversability evidence is incomplete',
    }
    text='**No current path:** '+explanations.get(status,str(status).replace('_',' '))+'.'
    if status=='waiting_for_reference':
        frames=metadata.get('reference_diagnostics',{}).get('frames',[])
        if frames:text+=f" {sum(f.get('reason')=='accepted' for f in frames)}/3 suitable frames."
    if has_history:
        text+=' Grey shows the previous validated trace.'
        if status=='updating' and metadata.get('planning_lag_frames'):
            text+=f" Planning is {metadata['planning_lag_frames']} mapped frame(s) behind capture."
    return text

class LocalOverlay:
    def __init__(self,server,detail=None):
        self.server=server; self.handles={}; self.data=None; self.mtime=None
        self.detail=detail;self.extent=1.;self.active=False;self.normal=np.array([0.,0.,1.])
        if detail is not None:server.scene.on_click(modifier='cmd/ctrl')(self.inspect)

    def inspect(self,event):
        data=self.data
        if data is None or not len(data['points']) or not self.handles.get('grid') or not self.handles['grid'].visible:return
        direction=np.asarray(event.ray_direction);direction/=np.linalg.norm(direction)
        delta=data['points']-np.asarray(event.ray_origin);along=delta@direction
        distance=np.linalg.norm(delta-along[:,None]*direction,axis=1);distance[along<=0]=np.inf
        row=int(np.argmin(distance))
        if distance[row]>self.extent*.01:return
        bits=int(data.get('reasons',np.zeros(len(distance),np.uint8))[row])
        reason=', '.join(label for flag,label in LABELS.items() if bits&flag)
        available=np.isfinite(data['costs'][row]);angle=data['slopes'][row]
        slope=f'{np.degrees(angle):.1f}°' if bool(data['slopes_enabled']) and np.isfinite(angle) else ('unknown' if bool(data['slopes_enabled']) else 'disabled')
        frame=int(data.get('source_frames',np.full(len(distance),-1))[row])
        self.detail.content=f"Selected terrain: **{'eligible' if available else reason or 'excluded'}** · slope {slope} · source frame {frame}."

    def clear(self):
        for handle in self.handles.values(): handle.remove()
        self.handles={}; self.data=None; self.mtime=None
        if self.detail is not None:self.detail.content='Ctrl/Cmd-click terrain to inspect its decision.'

    def focus(self,client):
        if self.data is None:return False
        keys=('route','last_route','history') if self.active else ('last_route','history')
        points=None
        for key in keys:
            if key not in self.data or len(self.data[key])<2:continue
            candidate=self.data[key];keep=~obsolete_lower(candidate,surface_from_overlay(self.data))
            if keep.sum()>1:points=candidate[keep];break
        if points is None:return False
        centre=(points.min(0)+points.max(0))/2
        scale=max(np.linalg.norm(np.ptp(points,axis=0)),self.extent*.05,1e-3)
        axis=np.eye(3)[np.argmin(abs(self.normal))];side=axis-(axis@self.normal)*self.normal;side/=np.linalg.norm(side)
        client.camera.position=centre+scale*(1.4*self.normal+.9*side)
        client.camera.look_at=centre;client.camera.up_direction=self.normal
        return True

    def visibility(self,show_grid,show_route,active):
        for name,handle in self.handles.items():
            if name=='grid':handle.visible=show_grid
            elif name=='last_route':handle.visible=show_route and not active
            else:handle.visible=show_route and (active or name=='history')

    def update(self,directory,metadata,colour,extent,show_grid,show_route):
        path=directory/'local_planning/display.npz'
        if path.exists() and path.stat().st_mtime_ns!=self.mtime:
            with np.load(path,allow_pickle=False) as data: self.data={key:data[key].copy() for key in data.files}
            self.mtime=path.stat().st_mtime_ns
            self.normal=np.asarray(metadata.get('reference',[0.,0.,1.]),float)
            self.normal/=max(np.linalg.norm(self.normal),1e-12)
            lift=self.normal*max(extent*.0008,float(metadata.get('radius',0))*.03)
            for name in ('route','history','last_route','goal'):
                if name in self.handles: self.handles.pop(name).remove()
            for name,colour_value in (('route',(240,0,220)),('history',(150,150,150)),('last_route',(165,170,180))):
                points=self.data.get(name,np.empty((0,3)))
                if len(points)>1:
                    keep=~obsolete_lower(points,surface_from_overlay(self.data))
                    edges=keep[:-1]&keep[1:]
                    if not edges.any():continue
                    self.handles[name]=self.server.scene.add_line_segments('/local_'+name,
                        points=(np.stack((points[:-1],points[1:]),axis=1)[edges]+lift).astype(np.float32),colors=colour_value,
                        line_width=6. if name=='route' else 3.)
            if self.data['goal'].size:
                self.handles['goal']=self.server.scene.add_point_cloud('/local_goal',
                    points=(self.data['goal'].reshape(1,3)+lift).astype(np.float32),colors=(255,210,0),
                    point_size=extent*.014,point_shape='circle')
        if self.data is None: return []
        data=self.data;colours=terrain_colors(data,colour);self.extent=extent
        if len(data['points']):
            if 'grid' in self.handles: self.handles['grid'].colors=colours
            else:
                self.handles['grid']=self.server.scene.add_point_cloud('/local_grid',points=data['points'].astype(np.float32),
                    colors=colours,point_size=extent*.003,point_shape='circle')
            # Geometry changes between revisions, not only colours.
            self.handles['grid'].points=data['points'].astype(np.float32)
        active=metadata.get('active_valid',False) and int(data['revision'])==metadata.get('latest_revision',metadata.get('revision'))
        self.active=bool(active)
        self.visibility(show_grid,show_route,active)
        modes=['Traversability']
        if bool(data['costs_enabled']): modes.append('Local costs')
        if bool(data['slopes_enabled']): modes.append('Local slopes')
        return modes
