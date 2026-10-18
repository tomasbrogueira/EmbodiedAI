"""Bounded prompt and display state; streaming PLY output for incremental maps."""
from pathlib import Path

import numpy as np


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



from vlm.state import PromptState
from visualizer.display import DisplayBuffer
