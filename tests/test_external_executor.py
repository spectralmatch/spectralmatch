"""Borrowed executors preserve ordering, ownership and progress across phases."""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import get_context
import os
import pytest

from spectralmatch.utils_multiprocessing import _run_image_tasks, uses_executor, current_executor


def worker(value):
    if value < 0:
        raise ValueError('negative image')
    return value * 2, os.getpid()


@uses_executor
def phase(values, *, executor=None):
    return _run_image_tasks(worker, [(v,) for v in values], input_paths=['image'] * len(values),
                            output_paths=[None] * len(values))


@pytest.mark.parametrize('kind', ['thread', 'process'])
def test_external_executor_is_borrowed_and_nested_phases_reuse_it(kind):
    pool = (ProcessPoolExecutor(2, mp_context=get_context('spawn')) if kind == 'process' else ThreadPoolExecutor(2))
    with pool:
        result = phase([3, 1, 2], executor=pool)
        assert [v for v, _ in result] == [6, 2, 4]
        if kind == 'process':
            assert all(pid != os.getpid() for _, pid in result)
        assert phase([5], executor=pool)[0][0] == 10
        with pytest.raises(ValueError, match='negative image'):
            phase([-1, 4], executor=pool)
        assert pool.submit(abs, -5).result(timeout=5) == 5
    assert current_executor() is None


def test_parent_result_callback_order_and_no_private_pool():
    commits = []
    with ThreadPoolExecutor(2) as pool:
        result = _run_image_tasks(worker, [(2,), (3,)], input_paths=['a','b'], output_paths=[None,None],
                                  executor=pool, result_callback=lambda i, value: commits.append((i, value[0])),
                                  executor_factory=lambda *a, **kw: pytest.fail('private pool created'))
    assert sorted(commits) == [(0,4),(1,6)]
    assert [v for v, _ in result] == [4,6]


def test_nested_coordinator_restores_executor_context():
    @uses_executor
    def outer(*, executor=None):
        assert current_executor() is pool
        return phase([2])
    with ThreadPoolExecutor(1) as pool:
        assert outer(executor=pool)[0][0] == 4
    assert current_executor() is None


@pytest.mark.parametrize('method', ['global_regression', 'local_block_adjustment'])
def test_matching_phases_use_external_process_pool(tmp_path, method):
    from spectralmatch import Match
    from .test_match import _make_two_test_rasters
    paths, outputs, _, _ = _make_two_test_rasters(tmp_path, [('A', 100), ('B', 120)])
    seen = []
    events = []
    kwargs = {'pif_method': 'entire'} if method == 'global_regression' else {'number_of_blocks': (2, 2)}
    with ProcessPoolExecutor(2, mp_context=get_context('spawn')) as pool:
        original = pool.submit
        def submit(function, *args, **kwargs):
            seen.append(function.__name__)
            return original(function, *args, **kwargs)
        pool.submit = submit
        getattr(Match, method)(paths, outputs, executor=pool, tile_threads=1,
                               progress_callback=lambda **stats: events.append(stats), **kwargs)
        assert pool.submit(abs, -2).result(timeout=5) == 2
    assert all(os.path.isfile(path) for path in outputs)
    assert seen.count('_run_logged_image') >= 4
    assert any(e.get('unit') == 'images' and e.get('n') == e.get('total') == 2 for e in events)


def test_tiled_merge_borrows_image_executor(tmp_path):
    from spectralmatch import merge_rasters
    from .utils_test import create_dummy_raster
    source = tmp_path / 'source.tif'
    create_dummy_raster(source, 32, 32, count=1, fill_value=100)
    output = tmp_path / 'tiles'
    seen = []
    with ProcessPoolExecutor(2, mp_context=get_context('spawn')) as pool:
        original = pool.submit
        def submit(function, *args, **kwargs):
            seen.append(function.__name__)
            return original(function, *args, **kwargs)
        pool.submit = submit
        merge_rasters([str(source)], str(output), output_tiles=True, window_size=16,
                      tile_threads=1, executor=pool)
        assert pool.submit(abs, -1).result(timeout=5) == 1
    assert len(list(output.glob('*.tif'))) == 4
    assert seen.count('_run_logged_image') == 4


def test_worker_does_not_inherit_coordinator_executor():
    from spectralmatch.utils_logging import _run_logged_image
    with ThreadPoolExecutor(1) as pool:
        @uses_executor
        def coordinator(*, executor=None):
            assert current_executor() is pool
            value, _ = _run_logged_image(current_executor, 'image', None, ())
            assert value is None
            assert current_executor() is pool
        coordinator(executor=pool)
