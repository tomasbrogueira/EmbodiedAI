"""Accumulated reconstruction survives current-frame and terrain updates."""
import unittest
import numpy as np
from visualizer.display import planning_view

class MapDisplayTests(unittest.TestCase):
    def surface(self,height=.1):
        return dict(layer_filter=True,basis=np.eye(3),height=np.full((2,2),height),
            layer_confirmed=np.ones((2,2),bool),cell=1.,origin=np.zeros(2,int),layer_tolerance=.02)

    def test_full_map_preserves_original_rows_even_with_a_different_current_frame(self):
        points=np.array([[.2,.2,0.],[.2,.2,.1],[.2,.2,.2],[.5,.5,-.3]])
        colors=np.array([[30,210,90],[30,210,90],[80,90,100],[255,0,0]],np.uint8)
        for current in [points[:1],points[2:]]:
            data=dict(points=points,colors=colors,current_points=current,current_colors=colors[:len(current)])
            shown,rgb=planning_view(data,self.surface(),True)
            np.testing.assert_array_equal(shown,points);np.testing.assert_array_equal(rgb,colors)

    def test_optional_filter_preserves_objects_upper_ground_and_semantic_negatives(self):
        points=np.array([[.2,.2,0.],[.2,.2,.1],[.2,.2,.2],[.5,.5,-.3],[.5,.5,-.2]])
        colors=np.array([[30,210,90],[30,210,90],[30,210,90],[255,0,0],[80,90,100]],np.uint8)
        shown,rgb=planning_view(dict(points=points,colors=colors),self.surface())
        np.testing.assert_array_equal(shown,points[1:]);np.testing.assert_array_equal(rgb,colors[1:])

    def test_optional_filter_keeps_history_where_ground_is_unconfirmed(self):
        points=np.array([[.2,.2,0.],[.5,.5,-.2]])
        colors=np.array([[30,210,90],[80,90,100]],np.uint8)
        surface=self.surface();surface['layer_confirmed'][:]=False
        shown,rgb=planning_view(dict(points=points,colors=colors),surface)
        np.testing.assert_array_equal(shown,points);np.testing.assert_array_equal(rgb,colors)

if __name__=='__main__':unittest.main()
