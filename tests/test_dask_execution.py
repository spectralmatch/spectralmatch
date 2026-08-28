import sys
import os
import importlib
from types import ModuleType

import pytest

from spectralmatch.types_and_validation import Universal
from spectralmatch.utils_multiprocessing import (
    _get_executor,
    _parse_dask_scheduler,
    _resolve_parallel_config,
)


class _ImmediateDaskFuture:
    def __init__(self, function, args, kwargs):
        try:
            self._result = function(*args, **kwargs)
            self._exception = None
        except BaseException as exc:
            self._result = None
            self._exception = exc

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result


def _install_fake_dask(monkeypatch, calls):
    class Client:
        def __init__(self, *args, **kwargs):
            calls["client_args"] = args
            calls["client_kwargs"] = kwargs

        def submit(self, function, *args, **kwargs):
            calls["pure"] = kwargs.pop("pure")
            return _ImmediateDaskFuture(function, args, kwargs)

        def cancel(self, futures):
            calls["cancelled"] = len(futures)

        def close(self):
            calls["closed"] = True

    dask = ModuleType("dask")
    distributed = ModuleType("dask.distributed")
    distributed.Client = Client
    dask.distributed = distributed
    monkeypatch.setitem(sys.modules, "dask", dask)
    monkeypatch.setitem(sys.modules, "dask.distributed", distributed)


def test_dask_scheduler_validation():
    Universal._validate(
        image_processing_backend="dask",
        dask_scheduler=("address", "tcp://scheduler:8786"),
    )
    with pytest.raises(ValueError, match="mode"):
        Universal._validate(
            cache=None,
            image_processing_backend="dask",
            dask_scheduler=("url", "tcp://scheduler:8786"),
        )
    with pytest.raises(ValueError, match="non-empty"):
        Universal._validate(
            image_processing_backend="dask",
            dask_scheduler=("address", ""),
        )
    assert _resolve_parallel_config(
        None, "dask", ("address", "tcp://scheduler:8786")
    ) == (
        True,
        None,
    )


@pytest.mark.parametrize(
    "backend,scheduler,image_threads,message",
    [
        ("dask", None, None, "requires dask_scheduler"),
        ("local", ("address", "tcp://scheduler:8786"), None, "requires image_processing_backend"),
        ("dask", ("address", "tcp://scheduler:8786"), 2, "image_threads must be None"),
    ],
)
def test_misaligned_parallel_parameters_are_rejected(
    backend, scheduler, image_threads, message
):
    with pytest.raises(ValueError, match=message):
        Universal._validate(
            image_processing_backend=backend,
            dask_scheduler=scheduler,
            image_threads=image_threads,
        )


def test_dask_executor_address_adapter(monkeypatch):
    calls = {}
    _install_fake_dask(monkeypatch, calls)
    with _get_executor(
        "thread",
        None,
        image_processing_backend="dask",
        dask_scheduler=("address", " tcp://scheduler:8786 "),
    ) as executor:
        assert executor.submit(pow, 3, 2).result() == 9
    assert calls["client_args"] == ("tcp://scheduler:8786",)
    assert calls["pure"] is False
    assert calls["closed"] is True


def test_dask_scheduler_file_is_normalized(tmp_path):
    scheduler_file = tmp_path / "scheduler.json"
    scheduler_file.write_text("{}", encoding="utf-8")
    assert _parse_dask_scheduler(("file", str(scheduler_file))) == (
        "file",
        str(scheduler_file.resolve()),
    )


def test_missing_dask_scheduler_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        _parse_dask_scheduler(("file", str(tmp_path / "missing.json")))


def test_whole_statistics_builds_masked_vrt_inside_worker(monkeypatch):
    module = importlib.import_module("spectralmatch.match.global_regression")
    observed = {}

    def create_masked(name, path, **kwargs):
        observed.update(name=name, path=path, **kwargs)
        assert os.path.isdir(kwargs["out_dir"])
        return os.path.join(kwargs["out_dir"], "masked.vrt")

    def calculate(*args):
        observed["calculation_path"] = args[2]
        assert os.path.isdir(os.path.dirname(args[2]))
        return {"image": {}}

    monkeypatch.setattr(module, "_create_masked_vrt", create_masked)
    monkeypatch.setattr(module, "_whole_stats_from_masked_image", calculate)
    cutline = ("include", "/shared/cutline.gpkg", "image")
    assert module._whole_stats_process_image(
        False, 1, "/shared/image.tif", 3, "image", True, cutline, 0, False
    ) == {"image": {}}
    assert observed["vector_mask"] == cutline
    assert observed["path"] == "/shared/image.tif"
    assert not os.path.exists(os.path.dirname(observed["calculation_path"]))


def test_overlap_statistics_builds_one_masked_pair_inside_worker(monkeypatch):
    module = importlib.import_module("spectralmatch.match.global_regression")
    created = []

    def create_masked(name, path, **kwargs):
        created.append((name, path, kwargs))
        return os.path.join(kwargs["out_dir"], f"{name}.vrt")

    def calculate(*args):
        assert all(os.path.isdir(os.path.dirname(path)) for path in args[3:5])
        return {"A": {"B": {}}, "B": {"A": {}}}

    monkeypatch.setattr(module, "_create_masked_vrt", create_masked)
    monkeypatch.setattr(module, "_overlap_stats_from_masked_images", calculate)
    cutline = ("include", "/shared/cutline.gpkg", "image")
    module._overlap_stats_process_image(
        False, 1, 3, "/shared/A.tif", "/shared/B.tif", "A", "B",
        (0, 0, 2, 2), (1, 1, 3, 3), True, cutline, 0, False,
    )
    assert [(name, path) for name, path, _ in created] == [
        ("A", "/shared/A.tif"),
        ("B", "/shared/B.tif"),
    ]
    assert created[0][2]["out_dir"] == created[1][2]["out_dir"]
    assert all(entry[2]["vector_mask"] == cutline for entry in created)
    assert not os.path.exists(created[0][2]["out_dir"])


def test_local_blocks_build_masked_vrt_inside_worker(monkeypatch):
    module = importlib.import_module("spectralmatch.match.local_block_adjustment")
    observed = {}

    def create_masked(name, path, **kwargs):
        observed.update(path=path, **kwargs)
        return os.path.join(kwargs["out_dir"], "masked.vrt")

    def calculate(*args):
        observed["calculation_path"] = args[1]
        return "image", None

    monkeypatch.setattr(module, "_create_masked_vrt", create_masked)
    monkeypatch.setattr(module, "_calculate_blocks_from_masked_image", calculate)
    cutline = ("exclude", "/shared/cutline.gpkg", "image")
    module._calculate_block_process_image(
        "image", "/shared/image.tif", (0, 0, 1, 1), 1, 1, 3,
        False, 0, "float32", False, 1, cutline,
    )
    assert observed["vector_mask"] == cutline
    assert observed["path"] == "/shared/image.tif"
    assert not os.path.exists(os.path.dirname(observed["calculation_path"]))
