"""Bounded prompt and display state; streaming PLY output for incremental maps."""
from pathlib import Path

import numpy as np


class PromptState:
    """Keep recent concepts, never carry a pixel mask between camera frames."""
    def __init__(self, ttl=5., max_prompts=8):
        self.ttl, self.max_prompts = ttl, max_prompts
        self.seen = {}

    def observe(self, prompts, source_time):
        for prompt in prompts:
            # These equivalent descriptions generated duplicate SAM queries
            # in the development videos. Keep one object concept online.
            prompt = {'fallen log': 'log', 'detached log': 'log'}.get(prompt, prompt)
            self.seen[prompt] = source_time

    def active(self, source_time):
        self.seen = {p: t for p, t in self.seen.items() if source_time - t <= self.ttl}
        return [p for p, _ in sorted(self.seen.items(), key=lambda pair: pair[1], reverse=True)][:self.max_prompts]


class StreamingPly:
    """Append all observations to disk with a fixed-width updatable vertex count."""
    dtype = np.dtype([(c, '<f4') for c in 'xyz'] + [(c, 'u1') for c in ('red', 'green', 'blue')])

    def __init__(self, path):
        self.path, self.count = Path(path), 0
        self.file = self.path.open('xb+')
        prefix = b'ply\nformat binary_little_endian 1.0\nelement vertex '
        self.count_offset = len(prefix)
        self.file.write(prefix + b'00000000000000000000\n' +
            b'property float x\nproperty float y\nproperty float z\n'
            b'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')

    def append(self, points, colors):
        if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
            raise ValueError('Expected matching N x 3 points and colors')
        data = np.empty(len(points), dtype=self.dtype)
        for i, key in enumerate('xyz'):
            data[key] = points[:, i]
        for i, key in enumerate(('red', 'green', 'blue')):
            data[key] = colors[:, i]
        self.file.seek(0, 2)
        self.file.write(data.tobytes())
        self.file.flush()
        self.count += len(points)
        self.file.seek(self.count_offset)
        self.file.write(f'{self.count:020d}'.encode('ascii'))
        self.file.flush()

    def close(self):
        self.file.close()


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
