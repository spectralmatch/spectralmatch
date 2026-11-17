"""Optional caller-owned thread budgets; no workflow dependency."""
from contextlib import contextmanager
from contextvars import ContextVar

_BUDGET = ContextVar("spectralmatch_tile_thread_budget", default=None)


@contextmanager
def tile_thread_budget(budget):
    token = _BUDGET.set(budget if budget is not None else _BUDGET.get())
    try:
        yield
    finally:
        _BUDGET.reset(token)


def current_tile_threads(value):
    """Refresh allocation before a native tile operation, never during one."""
    budget = _BUDGET.get()
    return budget.tile_threads_for_jobs(value) if budget is not None else value
