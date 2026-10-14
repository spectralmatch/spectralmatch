"""Small, JSON-only progress protocol for nested phases and individual jobs."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic
from uuid import uuid4

_parent = ContextVar('progress_substep_parent', default=None)


class Substep:
    def __init__(self, callback, name, total=None, unit='jobs', kind='phase', reused=0):
        self.callback = callback
        self.started = monotonic()
        self.done = 0
        self.details = dict(id=uuid4().hex, parent=_parent.get(), name=name,
                            total=total, unit=unit, kind=kind, reused=reused)

    def emit(self, event, **values):
        if self.callback is not None:
            elapsed = monotonic() - self.started
            self.callback(n=1 if event == 'end' and self.details['kind'] == 'group' else self.done,
                          total=1 if event == 'end' and self.details['kind'] == 'group' else self.details['total'], prefix=self.details['name'],
                          unit=self.details['unit'], elapsed=elapsed,
                          rate=self.done / elapsed if elapsed and self.done else None,
                          operation=self.details['id'],
                          status='failed' if event == 'failed' else 'completed' if event == 'end' else 'running',
                          substep={**self.details, 'event': event, 'done': self.done, **values})

    def commit(self, job=None):
        self.done += 1
        self.emit('advance', job=job)

    def reporter(self, callback, job, scene):
        return JobReporter(callback, self.details, str(job), str(scene)) if callback is not None else None


@contextmanager
def substep(callback, name, total=None, unit='jobs', kind='phase', reused=0):
    node = Substep(callback, name, total, unit, kind, reused)
    token = _parent.set(node.details['id'])
    try:
        node.emit('start')
        yield node
    except BaseException:
        node.emit('failed')
        raise
    else:
        node.emit('end')
    finally:
        _parent.reset(token)


@dataclass
class JobReporter:
    callback: object
    details: dict
    job: str
    scene: str

    @property
    def identity(self):
        return self.details['id'] + ':' + self.job

    def __call__(self, **stats):
        child = stats.get('substep')
        if child is not None and child.get('parent') is None:
            stats['substep'] = {**child, 'parent': self.details['id']}
        self.callback(**{'scene': self.scene, 'job_id': self.identity,
                         'parent_substep': self.details['id'], **stats})

    def lifecycle(self, event):
        self(n=0, total=None, prefix=self.details['name'], unit=self.details['unit'],
             elapsed=0, rate=None, operation=self.identity,
             substep={**self.details, 'event': event, 'job': self.job, 'scene': self.scene})

    def message(self, text):
        if hasattr(self.callback, 'message'):
            self.callback.message(text)
