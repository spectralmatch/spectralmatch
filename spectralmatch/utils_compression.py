"""Per-call raster compression, propagated to image workers without global patches."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature

_COMPRESSION = ContextVar("spectralmatch_compression", default=True)

def current_compression():
    return _COMPRESSION.get()

@contextmanager
def compression_context(value):
    if not isinstance(value, bool):
        raise ValueError("compression must be true or false")
    token = _COMPRESSION.set(value)
    try:
        yield
    finally:
        _COMPRESSION.reset(token)

def uses_compression(function):
    parameters = signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        value = bound.arguments.pop("compression", current_compression())
        with compression_context(value):
            return function(*bound.args, **bound.kwargs)
    if wrapped.__doc__:
        wrapped.__doc__ += "\n\n    compression (bool): Default True. False disables TIFF/COG compression for this call, including temporary rasters and overviews. Nested calls inherit the setting unless explicitly overridden. Existing reused outputs are not rewritten."
    return wrapped

def raster_creation_options(options, driver="GTiff"):
    options = list(options or [])
    if driver not in ("GTiff", "COG"):
        return options
    if current_compression():
        return options if any(o.upper().startswith("COMPRESS=") for o in options) else options + ["COMPRESS=DEFLATE"]
    keys = {"COMPRESS", "PREDICTOR", "ZLEVEL", "ZSTD_LEVEL", "LEVEL", "JPEG_QUALITY", "QUALITY", "OVERVIEW_COMPRESS", "OVERVIEW_PREDICTOR", "OVERVIEW_QUALITY"}
    return [o for o in options if o.split("=", 1)[0].upper() not in keys] + ["COMPRESS=NONE"] + (["OVERVIEW_COMPRESS=NONE"] if driver == "COG" else [])
