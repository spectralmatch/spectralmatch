"""Write temporary destinations only after functions have decided against reuse."""
import os
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature

_POSTFIX = ContextVar("spectralmatch_writing_postfix", default=None)
_FINALIZER = ContextVar("spectralmatch_output_finalizer", default=None)


def current_output_context():
    return _POSTFIX.get(), _FINALIZER.get()


@contextmanager
def output_context(postfix, finalizer=None):
    a, b = _POSTFIX.set(postfix), _FINALIZER.set(finalizer)
    try:
        yield
    finally:
        _FINALIZER.reset(b)
        _POSTFIX.reset(a)


@contextmanager
def output_finalizer(finalizer):
    """Internal runner hook for its existing validation/overview settings."""
    token = _FINALIZER.set(finalizer)
    try:
        yield
    finally:
        _FINALIZER.reset(token)


def uses_writing_postfix(function):
    parameters = signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        postfix = bound.arguments.pop("validity_writing_postfix", None)
        if postfix is not None and (not isinstance(postfix, str) or '/' in postfix or '\\' in postfix):
            raise ValueError("validity_writing_postfix must be a filename suffix or None")
        with output_context(postfix, _FINALIZER.get()):
            return function(*bound.args, **bound.kwargs)
    if wrapped.__doc__:
        wrapped.__doc__ += "\n\n    validity_writing_postfix (str | None): Optional temporary filename suffix, used only after checking final outputs for reuse. None disables it."
    return wrapped


def _companions(path):
    if Path(path).suffix.lower() not in {'.tif', '.tiff', '.dat', '.img', '.vrt'}:
        return []
    result = [path + suffix for suffix in ('.aux.xml', '.msk', '.ovr', '.msk.ovr')]
    if Path(path).suffix.lower() in {'.dat', '.img'}:
        result.append(str(Path(path).with_suffix('.hdr')))
    return result


def _publish(outputs, mapping):
    for final in outputs:
        if not os.path.isfile(mapping.get(final, final)):
            raise RuntimeError(f"Output was not created: {final}")
    for final, work in mapping.items():
        if os.path.exists(work + '-wal') and os.path.getsize(work + '-wal'):
            raise RuntimeError(f"Output database must be closed before publishing: {work}")
        for source, destination in zip(_companions(work), _companions(final)):
            if os.path.isfile(source):
                os.replace(source, destination)
            elif os.path.isfile(destination):
                os.unlink(destination)
        os.replace(work, final)
        if os.path.isfile(final + '.incomplete'):
            os.unlink(final + '.incomplete')


@contextmanager
def writing_outputs(*outputs):
    """Use inside the writer AFTER checking original paths for native reuse.

    Successful blocks publish each completed image; failed blocks leave the
    original file intact. A runner may retain its existing pre-publication checks
    through an internal hook, without adding function parameters for them.
    """
    postfix, finalizer = current_output_context()
    mapping = {p: str(Path(p).with_name(Path(p).stem + postfix + Path(p).suffix))
               for p in outputs if not os.path.isdir(p)} if postfix else {}
    if set(mapping.values()) & set(outputs):
        raise ValueError("Writing filenames collide with declared final outputs")
    for work in mapping.values():
        for candidate in (work, *_companions(work), work + '.incomplete', work + '-wal', work + '-shm', work + '-journal'):
            if os.path.isfile(candidate):
                os.unlink(candidate)
    yield tuple(mapping.get(p, p) for p in outputs)
    if finalizer is not None:
        finalizer(outputs, mapping)
    elif mapping:
        _publish(outputs, mapping)
