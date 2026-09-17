from threading import Event

import pytest

from spectralmatch import utils_multiprocessing
from spectralmatch.utils import align_rasters
from spectralmatch.utils_logging import _report_image_result
from spectralmatch.utils_multiprocessing import _run_image_tasks

from .utils_test import create_dummy_raster


def _fail():
    _report_image_result("Partial count", 7)
    raise RuntimeError("image failed")


def _reporting_echo(value):
    _report_image_result("Tie points kept", value)
    _report_image_result("PIF pixels", value * 2)
    return value


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
    assert sum(' Start | ' in line for line in lines) == 2
    assert [line.split(' | ')[0].rsplit(' ', 1)[1] for line in lines if ' Completed ' in line] == ['1/2', '2/2']
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


def test_align_reports_progress_without_debug_logs_and_respects_resume(tmp_path, capfd):
    inputs = [tmp_path / "a.tif", tmp_path / "b.tif"]
    outputs = [tmp_path / "out_a.tif", tmp_path / "out_b.tif"]
    for path in inputs:
        create_dummy_raster(path, count=1)
    align_rasters([str(p) for p in inputs], [str(p) for p in outputs], image_threads=2)
    lines = capfd.readouterr().out.splitlines()
    assert lines[0] == "START ALIGN_RASTERS:"
    assert sum("Completed" in line for line in lines) == 2
    outputs[1].unlink()
    align_rasters([str(p) for p in inputs], [str(p) for p in outputs], image_threads=2, resume_from_outputs="yes")
    lines = capfd.readouterr().out.splitlines()
    assert not any(line.startswith("[a.tif]") for line in lines)
    assert "[b.tif] Completed 1/1" in lines
