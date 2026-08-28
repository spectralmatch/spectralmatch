import multiprocessing as mp
import os
import sys

from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
from typing import Tuple, Literal, Callable, Optional

from .types_and_validation import (
    Universal,
    _validate_dask_scheduler,
    _validate_image_processing_config,
)


def _parse_dask_scheduler(value: Universal.DaskScheduler):
    """Validate and normalize a Dask scheduler connection tuple."""
    _validate_dask_scheduler(value)
    if value is None:
        return None
    kind, target = value
    target = (
        os.path.abspath(os.path.expanduser(target))
        if kind == "file"
        else target.strip()
    )
    if kind == "file" and not os.path.isfile(target):
        raise ValueError(f"Dask scheduler file does not exist: {target}")
    return kind, target


def _choose_context(prefer_fork: bool = True) -> mp.context.BaseContext:
    """
    Chooses the most appropriate multiprocessing context based on platform and preference.

    Args:
        prefer_fork (bool): If True, prefers "fork" context where available; default is True.

    Returns:
        mp.context.BaseContext: Selected multiprocessing context ("fork", "forkserver", or "spawn").
    """

    if prefer_fork and sys.platform.startswith("linux"):
        return mp.get_context("fork")
    if prefer_fork and sys.platform == "darwin":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    try:
        return mp.get_context("forkserver")
    except ValueError:
        return mp.get_context("spawn")


def _resolve_parallel_config(
    workers: Literal["cpu"] | int | None,
    image_processing_backend: Universal.ImageProcessingBackend = "local",
    dask_scheduler: Universal.DaskScheduler = None,
) -> Tuple[bool, Optional[int]]:
    """
    Parses a parallel worker config into execution flags and worker count.

    Args:
        workers ("cpu" | int | None): Number of workers.
            - "cpu" → use os.cpu_count()
            - int   → use that many workers
            - None  → disables parallelism

    Returns:
        Tuple[bool, Optional[int]]:
            - Whether to run in parallel,
            - Number of workers.
    """
    _validate_image_processing_config(
        image_processing_backend, dask_scheduler, workers
    )
    if image_processing_backend == "dask":
        return True, None
    if workers is None:
        return False, 1
    max_workers = os.cpu_count() if workers == "cpu" else int(workers)
    return True, max_workers


def _get_executor(
    backend: str,
    max_workers: Optional[int],
    initializer: Optional[Callable] = None,
    initargs: Optional[tuple] = None,
    image_processing_backend: Universal.ImageProcessingBackend = "local",
    dask_scheduler: Universal.DaskScheduler = None,
):
    """
    Creates a parallel executor (process or thread) with optional initialization logic.

    Args:
        backend (str): Execution backend, either "process" or "thread".
        max_workers (int): Maximum number of worker processes or threads.
        initializer (Callable, optional): Function to initialize worker context.
        initargs (tuple, optional): Arguments to pass to the initializer.
        image_processing_backend: Execution backend for image-level tasks.
        dask_scheduler: Existing scheduler connection tuple required by Dask mode.

    Returns:
        Executor: An instance of ThreadPoolExecutor or ProcessPoolExecutor.

    Raises:
        ValueError: If the backend is not "process" or "thread".
    """

    _validate_image_processing_config(
        image_processing_backend, dask_scheduler, None
    )
    if image_processing_backend == "dask":
        return _DaskExecutor(dask_scheduler)

    if backend == "process":
        return ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initializer,
            initargs=initargs or (),
            mp_context=_choose_context(),
        )

    elif backend == "thread":
        if initializer is not None:
            # Run initializer immediately in the main thread for all worker threads to share context
            initializer(*(initargs or ()))
        return ThreadPoolExecutor(max_workers=max_workers)

    else:
        raise ValueError(f"Unsupported backend: {backend}")


class _DaskExecutor:
    """Expose a Dask client through the concurrent-futures interface used here."""

    def __init__(self, scheduler: Universal.DaskScheduler):
        scheduler = _parse_dask_scheduler(scheduler)
        try:
            from dask.distributed import Client
        except ImportError as exc:
            raise ImportError(
                "Dask execution requires dask.distributed; install spectralmatch[dask]."
            ) from exc
        kind, target = scheduler
        self._client = Client(scheduler_file=target) if kind == "file" else Client(target)
        self._futures = []

    def submit(self, function, /, *args, **kwargs):
        dask_future = self._client.submit(function, *args, pure=False, **kwargs)
        concurrent_future = Future()

        def complete(future):
            try:
                concurrent_future.set_result(future.result())
            except BaseException as exc:
                concurrent_future.set_exception(exc)

        dask_future.add_done_callback(complete)
        self._futures.append(dask_future)
        return concurrent_future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        if cancel_futures:
            self._client.cancel(self._futures)
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown(wait=True, cancel_futures=exc_type is not None)
