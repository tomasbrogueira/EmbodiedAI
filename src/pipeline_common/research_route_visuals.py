"""Native-map direction marks on the exact saved route and assumed ground up."""
import numpy as np


def mission_display_pose(pose, plan):
    """Fit an oblique view above the assumed floor; preserve every scene point."""
    pose=dict(pose)
    points=np.asarray(plan['path_points'],float)
    up=np.asarray(plan['assumed_up_vector'],float);up/=np.linalg.norm(up)
    forward=points[-1]-points[0];forward-=up*(forward@up);forward/=np.linalg.norm(forward)
    direction=-.55*forward+.835*up;direction/=np.linalg.norm(direction)
    center=np.asarray(pose['look_at'])
    distance=pose['distance_native_units']*1.4
    pose.update(position=(center+direction*distance).tolist(),up_direction=up.tolist(),
                method='forward_mission_oblique_assumed_floor_display_only',
                distance_native_units=distance)
    return pose


def route_arrow_segments(points, up, factor, spacing_m=.75, length_m=.18, width_m=.09):
    points=np.asarray(points,float);up=np.asarray(up,float);up/=np.linalg.norm(up)
    if len(points)<2:return np.empty((0,2,3))
    spans=np.linalg.norm(np.diff(points,axis=0),axis=1);total=float(spans.sum())
    output=[];cumulative=np.r_[0.,np.cumsum(spans)]
    for distance in np.arange(min(.35/factor,total/2),total,spacing_m/factor):
        j=min(int(np.searchsorted(cumulative,distance,side='right')-1),len(spans)-1)
        if spans[j]<length_m/factor:continue
        forward=(points[j+1]-points[j])/spans[j]
        side=np.cross(forward,up);side/=max(np.linalg.norm(side),1e-12)
        tip=points[j]+forward*(distance-cumulative[j])
        tail=tip-forward*(length_m/factor)
        left=tail+side*width_m/(2*factor);right=tail-side*width_m/(2*factor)
        output.extend(((left,tip),(right,tip)))
    return np.asarray(output,float).reshape(-1,2,3)


def mission_overlays_class(base):
    """Extend the frozen replay adapter without changing its lifetime machinery."""
    class MissionOverlays(base):
        def line(self,name,points,color,width):
            super().line(name,points,color,width)
            if 'research_route' not in name or len(points)<2:return
            plan=self.data['research_plan'];factor=plan['assumptions']['metres_per_native_unit']
            arrows=route_arrow_segments(points,plan['assumed_up_vector'],factor)
            if len(arrows):
                self.handles.append(self.viewer.server.scene.add_line_segments(name+'/direction',points=arrows.astype(np.float32),
                    colors=np.broadcast_to(np.array(color,np.uint8),arrows.shape).copy(),line_width=width))
            for label,point in [('Start',points[0]),('Goal',points[-1])]:
                self.handles.append(self.viewer.server.scene.add_label(name+'/'+label,text=label,position=point))
    return MissionOverlays
