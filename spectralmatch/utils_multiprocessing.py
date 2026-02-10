import multiprocessing as mp
import os
import sys

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Tuple, Literal, Callable, Optional
from multiprocessing import Lock

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
    if workers is None:
        return False, 1
    max_workers = os.cpu_count() if workers == "cpu" else int(workers)
    return True, max_workers


def _get_executor(
    backend: str,
    max_workers: int,
    initializer: Optional[Callable] = None,
    initargs: Optional[tuple] = None,
):
    """
    Creates a parallel executor (process or thread) with optional initialization logic.

    Args:
        backend (str): Execution backend, either "process" or "thread".
        max_workers (int): Maximum number of worker processes or threads.
        initializer (Callable, optional): Function to initialize worker context.
        initargs (tuple, optional): Arguments to pass to the initializer.

    Returns:
        Executor: An instance of ThreadPoolExecutor or ProcessPoolExecutor.

    Raises:
        ValueError: If the backend is not "process" or "thread".
    """

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