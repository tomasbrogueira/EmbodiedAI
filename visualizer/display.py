"""Display reservoir; never used to reduce the planning graph."""
import numpy as np

class DisplayBuffer:
    """Uniform random-priority reservoir over the whole accumulated map."""
    def __init__(self, max_points=200_000):
        if max_points <= 0:
            raise ValueError('Display point limit must be positive')
        self.max_points = max_points
        self.points = np.empty((0, 3), dtype=np.float32)
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self.priorities = np.empty(0)
        self.rng = np.random.default_rng(42)
        self.count = 0

    def append(self, points, colors):
        if not len(points):
            return
        joined_points = np.concatenate((self.points, points))
        joined_colors = np.concatenate((self.colors, colors))
        priorities = np.concatenate((self.priorities, self.rng.random(len(points))))
        self.count += len(points)
        if len(priorities) > self.max_points:
            indices = np.argpartition(priorities, self.max_points - 1)[:self.max_points]
        else:
            indices = np.arange(len(priorities))
        self.points = joined_points[indices]
        self.colors = joined_colors[indices]
        self.priorities = priorities[indices]

    def arrays(self):
        return self.points, self.colors
