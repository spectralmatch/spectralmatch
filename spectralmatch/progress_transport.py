"""Forward worker callbacks without pickling application callbacks or renderers."""

from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from queue import Queue
from threading import Thread
from uuid import uuid4


@dataclass
class WorkerReporter:
    queue: object = None
    topic: str | None = None

    def _send(self, event):
        if self.topic is not None:
            from distributed import get_worker
            get_worker().log_event(self.topic, event)
        else:
            self.queue.put(event)

    def __call__(self, **stats):
        keys = ("n", "total", "prefix", "unit", "elapsed", "rate", "operation", "status", "scene")
        self._send({"stats": {key: stats[key] for key in keys if key in stats}})

    def message(self, text):
        self._send({"message": text})


def _deliver(callback, event):
    if "stats" in event:
        callback(**event["stats"])
    elif hasattr(callback, "message"):
        callback.message(event["message"])


@contextmanager
def local_worker_progress(callback, *, processes=False):
    if callback is None:
        yield None
        return
    manager = get_context("spawn").Manager() if processes else None
    queue = manager.Queue() if manager is not None else Queue()
    errors = []

    def consume():
        while True:
            event = queue.get()
            if event is None:
                return
            try:
                _deliver(callback, event)
            except BaseException as exc:
                errors.append(exc)

    reader = Thread(target=consume, name="spectralmatch-progress", daemon=True)
    reader.start()
    try:
        yield WorkerReporter(queue=queue)
    finally:
        queue.put(None)
        reader.join()
        if manager is not None:
            manager.shutdown()
    if errors:
        raise errors[0]


@contextmanager
def dask_worker_progress(callback, client):
    if callback is None:
        yield None
        return
    topic = f"spectralmatch-progress-{uuid4().hex}"
    client.subscribe_topic(topic, lambda event: _deliver(callback, event[1]))
    try:
        yield WorkerReporter(topic=topic)
    finally:
        client.unsubscribe_topic(topic)
