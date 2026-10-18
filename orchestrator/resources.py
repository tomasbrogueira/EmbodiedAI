"""One cross-process GPU owner for batch stages and persistent live sessions."""
from contextlib import contextmanager
from pathlib import Path
import os


@contextmanager
def gpu_allocation():
    import fcntl
    # A user-wide lock also coordinates independent checkouts of this repository.
    directory = Path(os.environ.get('EMBODIEDAI_LOCK_DIR', Path.home() / '.cache/embodiedai'))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'gpu.lock').open('a') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('GPU is owned by another EmbodiedAI pipeline/session; unload or stop it first') from None
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)
