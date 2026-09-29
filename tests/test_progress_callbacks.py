"""Callbacks work in the parent and inside the existing worker backends."""

from pathlib import Path
from time import monotonic, sleep
import inspect

import pytest

from spectralmatch import align_rasters, global_regression, markov_triangles, merge_rasters
from spectralmatch.utils_multiprocessing import _run_image_tasks
from spectralmatch.utils_progress import current_callback, progress, reports_progress
from .utils_test import create_dummy_raster


def _tiles(value, receipt):
    with progress(total=3, desc="worker tiles", unit="tiles", mininterval=0) as bar:
        bar.update(1)
        deadline = monotonic() + 10
        while not Path(receipt).exists() and monotonic() < deadline:
            sleep(0.02)
        assert Path(receipt).exists(), "Callback did not arrive until after the worker returned"
        bar.update(2)
    return value * 2


@pytest.mark.parametrize("backend", ["serial", "thread", "process", "dask"])
def test_parent_and_worker_callbacks_arrive_during_work(tmp_path, backend, capsys):
    snapshots = []
    receipt = tmp_path / "received"

    # A local closure is deliberately not pickleable by multiprocessing.
    def callback(**stats):
        snapshots.append(stats)
        if stats["prefix"] == "worker tiles":
            receipt.touch()

    def run(**kwargs):
        return _run_image_tasks(
            _tiles, [(2, str(receipt)), (4, str(receipt))],
            input_paths=["a.tif", "b.tif"], output_paths=["out_a.tif", "out_b.tif"],
            parallel=backend != "serial", backend="process" if backend == "process" else "thread",
            workers=2, progress_callback=callback, **kwargs,
        )

    if backend == "dask":
        distributed = pytest.importorskip("distributed")
        with distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=False,
                                      dashboard_address=None) as cluster:
            result = run(concurrent_processing_backend="dask", dask_scheduler=("address", cluster.scheduler_address))
    else:
        result = run()
    assert result == [4, 8]
    assert snapshots[0]["n"] == 0 and snapshots[0]["total"] == 2
    assert snapshots[-1]["n"] == snapshots[-1]["total"] == 2
    assert any(s["prefix"] == "worker tiles" and s["n"] >= 1 for s in snapshots)
    assert {"n", "total", "prefix", "unit", "elapsed", "rate", "operation"} <= snapshots[-1].keys()
    assert current_callback() is None
    assert not capsys.readouterr().out


def _fail():
    raise RuntimeError("failed tile")


def test_failed_task_does_not_advance_completion():
    snapshots = []
    with pytest.raises(RuntimeError, match="failed tile"):
        _run_image_tasks(_fail, [()], input_paths=["a"], output_paths=["out"],
                         parallel=True, workers=1, progress_callback=lambda **s: snapshots.append(s))
    assert snapshots[-1]["n"] == 0
    assert current_callback() is None


def test_aggregate_raster_api_reports_gdal_and_image_progress(tmp_path):
    sources = [tmp_path / "a.tif", tmp_path / "b.tif"]
    outputs = [tmp_path / "out_a.tif", tmp_path / "out_b.tif"]
    for filename in sources:
        create_dummy_raster(filename, count=1)
    snapshots = []
    align_rasters([str(p) for p in sources], [str(p) for p in outputs], image_threads=2, resolution=2,
                  progress_callback=lambda **s: snapshots.append(s))
    assert all(p.exists() for p in outputs)
    assert any(s["unit"] == "images" and s["n"] == s["total"] == 2 for s in snapshots)
    assert any(s["total"] == 100 and s["prefix"] == "Warping raster" for s in snapshots)
    for function in (align_rasters, global_regression, markov_triangles):
        assert "progress_callback" in inspect.signature(function).parameters
        assert function.__worker_progress__


def test_lifecycle_reports_failure_and_restores_nested_context():
    snapshots = []

    @reports_progress
    def fail():
        raise RuntimeError("opaque failure")

    with pytest.raises(RuntimeError, match="opaque failure"):
        fail(progress_callback=lambda **s: snapshots.append(s))
    assert snapshots[-1]["status"] == "failed"
    assert current_callback() is None


def test_aggregate_tile_processes_report_after_parent_gdal_threads(tmp_path):
    source = tmp_path / "source.tif"
    create_dummy_raster(source, count=1)
    # Initialize native GDAL threads before starting the tile workers.
    align_rasters([str(source)], [str(tmp_path / "aligned.tif")], tile_threads=2, resolution=2)
    snapshots = []
    folder = tmp_path / "tiles"
    merge_rasters([str(source)], str(folder), output_tiles=True, image_threads=2,
                  window_size=32, progress_callback=lambda **s: snapshots.append(s))
    assert list(folder.glob("*.tif"))
    assert any(s["unit"] == "images" and s["n"] == s["total"] and s["n"] > 0 for s in snapshots)
