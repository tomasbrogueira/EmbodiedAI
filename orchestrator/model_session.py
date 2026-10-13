"""Own reusable model workers across independent video runs."""
import json
from pathlib import Path
import threading
from uuid import uuid4

ROLES = ('lingbot', 'vlm', 'sam')


def required_roles(config):
    roles = ['lingbot']
    if config.get('proposal_mode') != 'manual' and not config.get('no_vlm', False):
        roles.append('vlm')
    if not config.get('no_sam', False):
        roles.append('sam')
    return tuple(roles)


MODEL_SETTINGS = ('lingbot_python', 'vlm_python', 'sam_python', 'lingbot_checkpoint',
    'sam_checkpoint', 'vlm_model', 'max_new_tokens', 'image_size', 'camera_iterations',
    'max_frame_num', 'kv_cache_window', 'worker_threads', 'sam_text_cache', 'sam_text_cache_size')


class ModelSession:
    def __init__(self, worker_factory=None):
        self.manage_allocation = worker_factory is None
        self.allocation = None
        if worker_factory is None:
            from orchestrator.streaming import Worker
            worker_factory = Worker
        self.factory = worker_factory
        self.workers = {}
        self.signature = None
        self.active = False
        self.lock = threading.Lock()
        self.id = uuid4().hex

    @property
    def loaded(self):
        with self.lock:
            return bool(self.workers) and all(
                w.process.poll() is None for w in self.workers.values())

    @property
    def loaded_roles(self):
        with self.lock:
            return tuple(role for role, worker in self.workers.items() if worker.process.poll() is None)

    def _close_workers(self):
        with self.lock:
            workers, self.workers = self.workers, {}
            self.signature = None
        for worker in reversed(list(workers.values())):
            worker.close()
        if self.allocation is not None:
            self.allocation.__exit__(None, None, None)
            self.allocation = None

    def begin(self, config_path, directory, on_loading, on_ready):
        config = json.loads(Path(config_path).read_text())
        signature = tuple(config.get(key) for key in MODEL_SETTINGS)
        with self.lock:
            if self.active:
                raise RuntimeError('Model session already has an active video')
            self.active = True
        try:
            if not self.loaded or signature != self.signature:
                self._close_workers()
            roles = required_roles(config)
            for role in tuple(self.workers):
                if role not in roles:
                    self.workers.pop(role).close()
            if self.manage_allocation and self.allocation is None:
                from orchestrator.resources import gpu_allocation
                allocation = gpu_allocation()
                allocation.__enter__()
                self.allocation = allocation
            for role in roles:
                on_loading(role)
                worker = self.workers.get(role)
                reused = worker is not None
                if reused:
                    result = worker.call(op='reset', config=str(config_path))
                    result.update(load_seconds=0., original_load_seconds=worker.ready['load_seconds'])
                else:
                    worker = self.factory(role, config[role+'_python'], config_path, directory)
                    with self.lock:
                        self.workers[role] = worker
                    result = dict(worker.ready)
                result.update(reused=reused, process_id=worker.process.pid, session_id=self.id)
                on_ready(role, worker, result)
            self.signature = signature
            return [self.workers[role] for role in roles]
        except BaseException:
            self._close_workers()
            with self.lock:
                self.active = False
            raise

    def finish(self, healthy=True):
        """Drain outstanding work and drop sequence tensors, retaining weights."""
        results = {}
        try:
            if not healthy:
                self._close_workers()
                return results
            for role, worker in self.workers.items():
                if worker.pending:
                    worker.receive()
                results[role] = worker.call(op='release_sequence')
            return results
        except BaseException:
            self._close_workers()
            raise
        finally:
            with self.lock:
                self.active = False

    def close(self):
        with self.lock:
            if self.active:
                raise RuntimeError('Stop the active video before unloading models')
        self._close_workers()
