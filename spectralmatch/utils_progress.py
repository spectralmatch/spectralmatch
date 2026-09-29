"""Renderer-independent progress callbacks using tqdm's ``format_dict`` fields."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import Parameter, signature
from time import monotonic
from uuid import uuid4

from tqdm import tqdm


_callback = ContextVar("progress_callback", default=None)
_phase = ContextVar("progress_phase", default="Processing images")


def current_callback():
    return _callback.get()


class CallbackTqdm(tqdm):
    """Count and estimate rates with tqdm, forwarding snapshots without printing."""

    def __init__(self, *args, callback, **kwargs):
        self.callback = callback
        self.operation = uuid4().hex
        kwargs.update(gui=True, disable=False)
        super().__init__(*args, **kwargs)
        self.display()

    def display(self, *args, **kwargs):
        self.callback(**self.format_dict, operation=self.operation)

    def close(self):
        if not self.disable:
            self.display()
        super().close()


def progress(iterable=None, *, callback=None, **kwargs):
    """Use an inherited callback, or a regular (silent by default) tqdm bar.

    Callbacks receive keyword fields from ``tqdm.format_dict``: in particular
    ``n``, ``total``, ``prefix``, ``unit``, ``elapsed`` and ``rate``. ``operation``
    identifies a particular bar, allowing its count to restart for a new phase.
    """
    callback = callback if callback is not None else current_callback()
    if callback is not None:
        return CallbackTqdm(iterable, callback=callback, **kwargs)
    kwargs.setdefault("disable", True)
    return tqdm(iterable, **kwargs)


@contextmanager
def progress_context(callback=None):
    token = _callback.set(callback)
    phase_token = _phase.set(_phase.get())
    try:
        yield
    finally:
        _phase.reset(phase_token)
        _callback.reset(token)


def phase():
    return _phase.get()


def report_phase(description):
    _phase.set(description)
    callback = current_callback()
    if callback is not None:
        callback(n=0, total=None, prefix=description, unit="it", elapsed=0,
                 rate=None, operation=uuid4().hex)


@contextmanager
def operation(description, callback=None):
    """Report lifecycle for opaque operations without inventing a percentage."""
    callback = callback if callback is not None else current_callback()
    if callback is None:
        yield
        return
    identity, started = uuid4().hex, monotonic()
    fields = dict(prefix=description, unit="call", operation=identity, rate=None)
    callback(n=0, total=None, elapsed=0, **fields)
    try:
        yield
    except BaseException:
        callback(n=0, total=None, elapsed=monotonic() - started, status="failed", **fields)
        raise
    else:
        callback(n=1, total=1, elapsed=monotonic() - started, **fields)


def reports_progress(function=None, *, worker_progress=False):
    """Add an optional Python callback, preserving the function's public API."""
    if function is None:
        return lambda target: reports_progress(target, worker_progress=worker_progress)
    if getattr(function, "__reports_progress__", False):
        return function
    parameters = signature(function)
    accepts_callback = "progress_callback" in parameters.parameters

    @wraps(function)
    def wrapped(*args, progress_callback=None, **kwargs):
        callback = progress_callback if progress_callback is not None else current_callback()
        if callback is not None and not callable(callback):
            raise TypeError("progress_callback must be callable or None")
        token = _callback.set(callback)
        phase_token = _phase.set(function.__name__)
        try:
            with operation(function.__name__, callback):
                if accepts_callback:
                    kwargs["progress_callback"] = callback
                return function(*args, **kwargs)
        finally:
            _phase.reset(phase_token)
            _callback.reset(token)

    if not accepts_callback:
        items = list(parameters.parameters.values())
        position = next((i for i, p in enumerate(items) if p.kind == p.VAR_KEYWORD), len(items))
        items.insert(position, Parameter("progress_callback", Parameter.KEYWORD_ONLY, default=None))
        wrapped.__signature__ = parameters.replace(parameters=items)
    wrapped.__reports_progress__ = True
    wrapped.__worker_progress__ = worker_progress or accepts_callback
    return wrapped


def gdal_progress(description):
    """Adapt GDAL's fractional callback to the same tqdm field contract."""
    callback = current_callback()
    if callback is None:
        return None
    started, identity = monotonic(), uuid4().hex
    last = [-1.0]

    def report(fraction, message, data):
        elapsed = monotonic() - started
        if fraction in (0, 1) or elapsed - last[0] >= 0.1:
            last[0] = elapsed
            callback(n=fraction * 100, total=100, prefix=description, unit="%",
                     elapsed=elapsed, rate=fraction * 100 / elapsed if elapsed else None,
                     operation=identity)
        return 1

    return report
