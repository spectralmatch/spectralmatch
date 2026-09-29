import os
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from .utils_progress import current_callback, progress_context, report_phase


_IMAGE_RESULTS = ContextVar("spectralmatch_image_results", default=None)


@dataclass
class _SceneReporter:
    callback: object
    scene: str

    def __call__(self, **stats):
        self.callback(**{**stats, "scene": self.scene})

    def message(self, text):
        if hasattr(self.callback, "message"):
            self.callback.message(text)


def _print_line(message):
    """Write each progress line and its newline together so parallel messages cannot split between them."""
    callback = current_callback()
    if callback is None:
        sys.stdout.write(message + "\n")
        sys.stdout.flush()
    elif hasattr(callback, "message"):
        callback.message(message)


def _print_step_start(step):
    """Print a step announcement immediately, including when debug logging is disabled."""
    report_phase(step)
    _print_line(f"START {step.upper()}:")


def _format_paths(paths):
    """Format one or more file paths for a progress message."""
    if paths is None:
        return "memory"
    if isinstance(paths, (str, os.PathLike)):
        return os.fspath(paths)
    return ", ".join(os.fspath(path) for path in paths)


def _image_id(paths):
    """Use input filenames as image identifiers, joining filenames for pair or mosaic tasks."""
    if isinstance(paths, (str, os.PathLike)):
        return os.path.basename(os.fspath(paths).rstrip(os.sep))
    return ", ".join(_image_id(path) for path in paths)


def _print_image_start(input_paths, output_paths, *, image_id=None):
    """Announce an image before its processing begins, with every input and output path."""
    identifier = _image_id(input_paths) if image_id is None else image_id
    _print_line(f"[{identifier}] Start | in={_format_paths(input_paths)} | out={_format_paths(output_paths)}")


def _print_image_completed(input_paths, completed, total, *, image_id=None, details=None):
    """Report a successfully completed image using the parent's completion count."""
    identifier = _image_id(input_paths) if image_id is None else image_id
    summary = "".join(f" | {key}: {value}" for key, value in (details or {}).items())
    _print_line(f"[{identifier}] Completed {completed}/{total}{summary}")


def _report_image_result(key, value):
    """Collect a concise key-value result for a loop task, or print it separately outside a task."""
    key = " ".join(str(key).split())
    value = " ".join(str(value).split())
    results = _IMAGE_RESULTS.get()
    if results is None:
        _print_line(f"{key}: {value}")
    else:
        results[key] = value


def _run_logged_image(function, input_paths, output_paths, args, progress_callback=None):
    """Run an image in an isolated reporting context and return its original result with completion details."""
    details = {}
    token = _IMAGE_RESULTS.set(details)
    try:
        callback = _SceneReporter(progress_callback, _image_id(input_paths)) if progress_callback is not None else None
        with progress_context(callback):
            _print_image_start(input_paths, output_paths)
            result = function(*args)
        return result, details
    finally:
        _IMAGE_RESULTS.reset(token)
