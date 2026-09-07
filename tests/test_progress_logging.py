from concurrent.futures import ThreadPoolExecutor
import importlib
from threading import Event

import pytest

from spectralmatch import handlers, utils, utils_multiprocessing
from spectralmatch.utils import align_rasters, compute_overviews
from spectralmatch.utils_logging import _print_image_start, _print_step_start, _report_image_result
from spectralmatch.utils_multiprocessing import _run_image_tasks

from .utils_test import create_dummy_raster


def test_gdal_setup_helpers_are_silent_but_apply_settings(tmp_path, monkeypatch, capsys):
    source = tmp_path / "image.tif"
    create_dummy_raster(source, dtype="int16", count=1)
    cache_values, thread_values = [], []
    monkeypatch.setattr(utils.gdal, "SetCacheMax", cache_values.append)
    monkeypatch.setattr(utils.gdal, "SetConfigOption", lambda name, value: thread_values.append((name, value)))
    utils._set_gdal_cache(0.8, True)
    utils._set_gdal_workers(3, True)
    assert utils._resolve_gdal_dtype(None, str(source), True) == "Int16"
    assert utils._resolve_window_size(1024, str(source), True) == 1024
    assert capsys.readouterr().out == ""
    assert cache_values == [int(0.8 * 1024 ** 3)]
    assert thread_values == [("GDAL_NUM_THREADS", "3")]


@pytest.mark.parametrize("step", ["global_regression", "local_block_adjustment"])
def test_masked_vrts_have_standard_substep_progress(tmp_path, capsys, step):
    paths = [tmp_path / "a.tif", tmp_path / "b.tif"]
    for path in paths:
        create_dummy_raster(path, count=1)
    result = utils._create_masked_vrts(
        ["a", "b"], [str(path) for path in paths], step_name=step, out_dir=str(tmp_path), debug_logs=True,
    )
    assert result == [str(tmp_path / "vrt_a.vrt"), str(tmp_path / "vrt_b.vrt")]
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"START {step.upper()} MASKING:",
        f"[a.tif] Start | in={paths[0]} | out={result[0]}",
        "[a.tif] Completed 1/2",
        f"[b.tif] Start | in={paths[1]} | out={result[1]}",
        "[b.tif] Completed 2/2",
    ]
    for path in result:
        with utils.gdal.Open(path) as dataset:
            assert dataset.GetRasterBand(1).ReadAsArray().min() == 100


def _echo(value):
    return value


def _fail():
    _report_image_result("Partial count", 7)
    raise RuntimeError("image failed")


def _reporting_echo(value):
    _report_image_result("Tie points kept", value)
    _report_image_result("PIF pixels", value * 2)
    return value


def test_tie_point_summary_appears_on_completion(capsys):
    coregistration = importlib.import_module("spectralmatch.joint_coregistration.joint_coregistration")
    points = [(i % 37, i // 37, i % 37, i // 37) for i in range(1157)]
    result = _run_image_tasks(
        coregistration._filter_point_pairs, [(points, True)],
        input_paths=[("Worldview_20160923.tif", "Worldview_20160930.tif")], output_paths=[None],
    )
    assert len(result[0]) == 1157
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[1] == "[Worldview_20160923.tif, Worldview_20160930.tif] Completed 1/1 | Tie points kept: 1157"


@pytest.mark.parametrize("parallel,backend", [(False, "thread"), (True, "thread"), (True, "process")])
def test_results_join_the_correct_completion_line(capfd, parallel, backend):
    result = _run_image_tasks(
        _reporting_echo, [(1157,), (42,)],
        input_paths=[("a.tif", "b.tif"), ("b.tif", "c.tif")], output_paths=[None, None],
        parallel=parallel, backend=backend, workers=2,
    )
    assert result == [1157, 42]
    lines = capfd.readouterr().out.splitlines()
    assert len(lines) == 4
    first = next(line for line in lines if line.startswith("[a.tif, b.tif] Completed"))
    second = next(line for line in lines if line.startswith("[b.tif, c.tif] Completed"))
    assert first.endswith(" | Tie points kept: 1157 | PIF pixels: 2314")
    assert second.endswith(" | Tie points kept: 42 | PIF pixels: 84")


def test_standalone_results_stay_separate_and_context_resets(capsys):
    _run_image_tasks(_reporting_echo, [(1,)], input_paths=["a.tif"], output_paths=["out.tif"])
    capsys.readouterr()
    _report_image_result("Tie points kept", 1157)
    assert capsys.readouterr().out == "Tie points kept: 1157\n"
    with pytest.raises(RuntimeError):
        _run_image_tasks(_fail, [()], input_paths=["bad.tif"], output_paths=["out.tif"])
    assert "Partial count" not in capsys.readouterr().out
    _report_image_result("Summary", "done")
    assert capsys.readouterr().out == "Summary: done\n"


def test_progress_format_includes_all_outputs(capsys):
    _print_step_start("cloud_mask")
    _print_image_start("/input/scene.tif", ["/output/masked.tif", "/output/mask.tif"])
    assert capsys.readouterr().out.splitlines() == [
        "START CLOUD_MASK:",
        "[scene.tif] Start | in=/input/scene.tif | out=/output/masked.tif, /output/mask.tif",
    ]


def test_glob_search_is_announced_before_searching(monkeypatch, capsys):
    def search(pattern, recursive):
        assert capsys.readouterr().out.splitlines() == [
            "START SEARCH_PATHS:",
            "Searching for glob matches: /slow/input/*.tif",
        ]
        return ["/slow/input/scene.tif"]

    monkeypatch.setattr(handlers.glob, "glob", search)
    assert handlers.search_paths("/slow/input/*.tif") == ["/slow/input/scene.tif"]


@pytest.mark.parametrize("parallel,backend", [(False, "thread"), (True, "thread"), (True, "process")])
def test_image_progress_serial_and_parallel(capfd, parallel, backend):
    assert _run_image_tasks(
        _echo, [(1,), (2,)],
        input_paths=["a.tif", "b.tif"], output_paths=["out_a.tif", "out_b.tif"],
        parallel=parallel, backend=backend, workers=2,
    ) == [1, 2]
    lines = capfd.readouterr().out.splitlines()
    assert len(lines) == 4
    for name in ("a", "b"):
        start = f"[{name}.tif] Start | in={name}.tif | out=out_{name}.tif"
        completion = next(line for line in lines if line.startswith(f"[{name}.tif] Completed"))
        assert lines.index(start) < lines.index(completion)
    assert [line.rsplit(" ", 1)[1] for line in lines if "Completed" in line] == ["1/2", "2/2"]


def test_completion_counts_follow_finished_images(monkeypatch, capsys):
    release_first = Event()
    original = utils_multiprocessing._print_image_completed

    def process(value):
        _report_image_result("Result", value)
        if value == 1:
            assert release_first.wait(5)
        return value

    def completed(source, count, total, **kwargs):
        original(source, count, total, **kwargs)
        if source == "second.tif":
            release_first.set()

    monkeypatch.setattr(utils_multiprocessing, "_print_image_completed", completed)
    assert _run_image_tasks(
        process, [(1,), (2,)],
        input_paths=["first.tif", "second.tif"], output_paths=["first_out.tif", "second_out.tif"],
        parallel=True, workers=2,
    ) == [1, 2]
    assert [line for line in capsys.readouterr().out.splitlines() if "Completed" in line] == [
        "[second.tif] Completed 1/2 | Result: 2", "[first.tif] Completed 2/2 | Result: 1",
    ]


@pytest.mark.parametrize("parallel", [False, True])
def test_failed_image_is_never_reported_completed(capsys, parallel):
    with pytest.raises(RuntimeError, match="image failed"):
        _run_image_tasks(_fail, [()], input_paths=["bad.tif"], output_paths=["out.tif"], parallel=parallel, workers=1)
    assert capsys.readouterr().out.splitlines() == ["[bad.tif] Start | in=bad.tif | out=out.tif"]


def test_image_progress_uses_shared_dask_executor(capsys):
    captured = []

    def executor_factory(backend, workers, **kwargs):
        captured.append(kwargs)
        return ThreadPoolExecutor(1)

    _run_image_tasks(
        _echo, [(1,)], input_paths=["scene.tif"], output_paths=["out.tif"],
        parallel=True, concurrent_processing_backend="dask", dask_scheduler=("address", "tcp://scheduler:8786"),
        executor_factory=executor_factory,
    )
    assert captured == [{"concurrent_processing_backend": "dask", "dask_scheduler": ("address", "tcp://scheduler:8786")}]
    assert "[scene.tif] Completed 1/1" in capsys.readouterr().out


@pytest.mark.parametrize("workers", [None, 2])
def test_align_reports_progress_without_debug_logs_and_respects_resume(tmp_path, capfd, workers):
    inputs = [tmp_path / "a.tif", tmp_path / "b.tif"]
    outputs = [tmp_path / "out_a.tif", tmp_path / "out_b.tif"]
    for path in inputs:
        create_dummy_raster(path, count=1)
    align_rasters([str(p) for p in inputs], [str(p) for p in outputs], image_threads=workers)
    lines = capfd.readouterr().out.splitlines()
    assert lines[0] == "START ALIGN_RASTERS:"
    assert sum("Completed" in line for line in lines) == 2
    outputs[1].unlink()
    align_rasters([str(p) for p in inputs], [str(p) for p in outputs], image_threads=workers, resume_from_outputs="yes")
    lines = capfd.readouterr().out.splitlines()
    assert not any(line.startswith("[a.tif]") for line in lines)
    assert "[b.tif] Completed 1/1" in lines


def test_overview_copy_is_announced_before_copying(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.tif"
    output = tmp_path / "copy.tif"
    create_dummy_raster(source, count=1)
    original = handlers.gdal.Translate

    def translate(destination, input_path):
        assert f"[source.tif] Start | in={source} | out={output}" in capsys.readouterr().out
        return original(destination, input_path)

    monkeypatch.setattr(handlers.gdal, "Translate", translate)
    compute_overviews([str(source)], output_image_paths=[str(output)], window_scales=None)
    assert "[source.tif] Completed 1/1" in capsys.readouterr().out
    assert output.is_file()
