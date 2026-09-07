import csv
import inspect
import math
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import pytest
from osgeo import gdal
from osgeo_utils import gdal_retile

from spectralmatch import Match, joint_coregistration, merge_rasters, pipeline, utils
from spectralmatch.handlers import _gdal_raster_is_valid
from spectralmatch.utils import compute_overviews
from .utils_test import create_dummy_raster


@pytest.fixture
def merge_sources(tmp_path):
    paths = []
    for i, origin in enumerate((0, 48)):
        path = tmp_path / f"source_{i}.tif"
        pixels = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) + 100 * i
        pixels[16:24, 16:24] = 65535
        create_dummy_raster(
            path, band_data=pixels, nodata=65535,
            transform=(origin, 1, 0, 64, 0, -1),
        )
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("resolution", [2, 0.5])
@pytest.mark.parametrize("output_tiles", [False, True])
def test_merge_numeric_resolution(tmp_path, resolution, output_tiles):
    source = tmp_path / "source.tif"
    create_dummy_raster(source, width=32, height=32, count=1, crs="EPSG:3857", fill_value=75)
    output = tmp_path / ("tiles" if output_tiles else "merged.tif")
    merge_rasters([str(source)], str(output), output_tiles=output_tiles, resolution=resolution, window_size=16)
    mosaic = output / "MergedImage.vrt" if output_tiles else output
    with gdal.Open(str(mosaic)) as dataset:
        assert dataset.RasterXSize == dataset.RasterYSize == int(32 / resolution)
        assert dataset.GetGeoTransform()[1] == resolution
        assert dataset.GetGeoTransform()[5] == -resolution
        np.testing.assert_array_equal(dataset.ReadAsArray(), 75)


@pytest.mark.parametrize("resolution", [None, True, False, 0, -2, float("nan"), float("inf"), "user", [2, 2]])
def test_merge_rejects_invalid_resolution(tmp_path, resolution):
    with pytest.raises(ValueError, match="resolution"):
        merge_rasters(["unused.tif"], str(tmp_path / "merged.tif"), resolution=resolution)


@pytest.mark.parametrize("workers", [None, 2])
def test_merge_tiles_matches_single_mosaic(merge_sources, tmp_path, workers):
    single = tmp_path / "merged.tif"
    folder = tmp_path / "tiles"
    merge_rasters(merge_sources, str(single), custom_nodata_value=65535)
    result = merge_rasters(
        merge_sources, str(folder), output_tiles=True, image_threads=workers,
        window_size=24, overlap=4, custom_tiles_csv="index.csv",
        custom_nodata_value=65535, output_dtype="Float32",
    )
    assert result == str(folder)
    tiles = sorted(folder.glob("*.tif"))
    assert len(tiles) == 18
    tiled_mosaic = gdal.BuildVRT("", [str(tile) for tile in tiles])
    expected = gdal.Open(str(single))
    np.testing.assert_array_equal(tiled_mosaic.ReadAsArray(), expected.ReadAsArray())
    assert tiled_mosaic.GetGeoTransform() == expected.GetGeoTransform()
    assert tiled_mosaic.GetProjection() == expected.GetProjection()
    with (folder / "index.csv").open() as stream:
        rows = list(csv.reader(stream, delimiter=";"))
    assert len(rows) == len(tiles)
    for name, minx, maxx, miny, maxy in rows:
        tile = gdal.Open(str(folder / name))
        x, dx, _, y, _, dy = tile.GetGeoTransform()
        assert tuple(map(float, (minx, maxx, miny, maxy))) == (
            x, x + dx * tile.RasterXSize, y + dy * tile.RasterYSize, y,
        )
        assert tile.GetRasterBand(1).GetNoDataValue() == 65535
        assert tile.GetRasterBand(1).DataType == gdal.GDT_Float32
    assert not list(folder.glob(".merge_rasters-*"))


def test_parallel_pyramids_match_native_retile(merge_sources, tmp_path):
    for workers, name in [(None, "serial"), (2, "parallel")]:
        merge_rasters(
            merge_sources, str(tmp_path / name), output_tiles=True,
            image_threads=workers, window_size=24, overlap=3,
            build_overviews=True, resampling_method="bilinear",
            custom_tiles_csv="index.csv",
        )
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    assert [p.name for p in sorted(serial.iterdir()) if p.is_dir()] == ["1", "2", "3", "4", "5"]
    assert {p.relative_to(serial) for p in serial.rglob("*")} == {
        p.relative_to(parallel) for p in parallel.rglob("*")
    }
    for path in serial.rglob("*.tif"):
        other = parallel / path.relative_to(serial)
        a, b = gdal.Open(str(path)), gdal.Open(str(other))
        assert a.GetGeoTransform() == b.GetGeoTransform()
        np.testing.assert_array_equal(a.ReadAsArray(), b.ReadAsArray())
    for path in serial.rglob("*.csv"):
        assert path.read_text() == (parallel / path.relative_to(serial)).read_text()


@pytest.mark.parametrize("workers", [None, 2])
def test_tile_resume_validates_and_repairs(merge_sources, tmp_path, workers):
    folder = tmp_path / "tiles"
    kwargs = dict(output_tiles=True, image_threads=workers, window_size=32,
                  build_overviews=True, custom_tiles_csv="index.csv")
    merge_rasters(merge_sources, str(folder), **kwargs)
    tiles = sorted(folder.rglob("*.tif"))
    original = {path: path.read_bytes() for path in tiles}
    untouched = tiles[-1]
    stamp = untouched.stat().st_mtime_ns
    broken, missing = tiles[0], tiles[1]
    missing.unlink()
    merge_rasters(merge_sources, str(folder), resume_from_outputs="yes", **kwargs)
    assert missing.exists()
    # Validate both base and pyramid tiles, and repair invalid files before resume.
    broken.write_bytes(b"invalid raster")
    next(folder.glob("*.tif")).write_bytes(b"invalid base tile")
    merge_rasters(merge_sources, str(folder), resume_from_outputs="validate", **kwargs)
    assert all(_gdal_raster_is_valid(str(path))[0] for path in tiles)
    assert all(path.read_bytes() == original[path] for path in tiles)
    assert untouched.stat().st_mtime_ns == stamp
    merge_rasters(merge_sources, str(folder), resume_from_outputs="no", **kwargs)
    assert untouched.stat().st_mtime_ns != stamp


@pytest.mark.parametrize("kwargs, message", [
    ({"output_tiles": "yes"}, "output_tiles"),
    ({"image_threads": 2}, "require output_tiles"),
    ({"concurrent_processing_backend": "process_pool"}, "require output_tiles"),
    ({"overlap": 1}, "require output_tiles"),
    ({"custom_tiles_csv": "index.csv"}, "require output_tiles"),
    ({"output_tiles": True, "overlap": -1}, "overlap"),
    ({"output_tiles": True, "overlap": 1.5}, "overlap"),
    ({"output_tiles": True, "overlap": True}, "overlap"),
    ({"output_tiles": True, "overlap": 256}, "overlap"),
    ({"output_tiles": True, "window_size": 16, "overlap": 16}, "overlap"),
    ({"output_tiles": True, "custom_tiles_csv": "../index.csv"}, "custom_tiles_csv"),
    ({"output_tiles": True, "custom_tiles_csv": True}, "custom_tiles_csv"),
    ({"resampling_method": "unknown"}, "resampling_method"),
    ({"resume_from_outputs": "maybe"}, "resume_from_outputs"),
    ({"window_scales": (2, 2)}, "window_scales"),
    ({"output_tiles": True, "window_scales": (2, 8)}, "window_scales"),
    ({"window_size": True}, "window_size"),
    ({"window_size": 17}, "multiple of 16"),
    ({"create_vrts": "Custom.vrt"}, "requires output_tiles"),
    ({"output_tiles": True, "create_vrts": "../Custom.vrt"}, "create_vrts"),
    ({"output_tiles": True, "create_vrts": "/Custom.vrt"}, "create_vrts"),
    ({"output_tiles": True, "create_vrts": "Custom.tif"}, "create_vrts"),
    ({"output_tiles": True, "create_vrts": ".vrt"}, "create_vrts"),
    ({"output_tiles": True, "create_vrts": None}, "create_vrts"),
    ({"output_tiles": True, "create_vrts": True}, "create_vrts"),
])
def test_merge_tile_validation(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        merge_rasters(["unused.tif"], str(tmp_path / "output"), **kwargs)


def test_merge_output_path_validation(tmp_path):
    for path in [tmp_path / "output.tif", tmp_path / "existing"]:
        if path.name == "existing":
            path.write_text("file")
        with pytest.raises(ValueError, match="must be a folder"):
            merge_rasters(["unused.tif"], str(path), output_tiles=True)
    with pytest.raises(ValueError, match="must be a file"):
        merge_rasters(["unused.tif"], str(tmp_path))


def test_retile_flags_and_overview_defaults(merge_sources, tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "_create_tile_vrts", lambda *args: None)
    calls = []
    monkeypatch.setattr(gdal_retile, "main", lambda argv: calls.append(argv) or 0)
    merge_rasters(
        merge_sources, str(tmp_path / "tiles"), output_tiles=True,
        window_size=32, overlap=5, build_overviews=True, debug_logs=True,
        resampling_method="cubic", custom_tiles_csv="index.csv",
        resume_from_outputs="validate",
    )
    argv = calls[0]
    assert argv[argv.index("-levels") + 1] == "5"
    assert argv[argv.index("-r") + 1] == "cubic"
    assert argv[argv.index("-overlap") + 1] == "5"
    assert argv[argv.index("-csv") + 1] == "index.csv"
    assert "-v" in argv and "-resume" in argv
    for function in (merge_rasters, compute_overviews, Match.global_regression, Match.local_block_adjustment, joint_coregistration):
        assert inspect.signature(function).parameters["window_scales"].default == (2, 4, 8, 16, 32)
    assert inspect.signature(pipeline).parameters["shared_window_scales"].default == (2, 4, 8, 16, 32)


def test_retile_failure_propagates(merge_sources, tmp_path, monkeypatch):
    monkeypatch.setattr(gdal_retile, "main", lambda argv: 1)
    with pytest.raises(RuntimeError, match="gdal_retile failed"):
        merge_rasters(merge_sources, str(tmp_path / "tiles"), output_tiles=True)
    assert not list((tmp_path / "tiles").glob(".merge_rasters-*"))


def test_resampling_applies_to_mixed_resolution_inputs(tmp_path):
    fine, coarse = tmp_path / "fine.tif", tmp_path / "coarse.tif"
    create_dummy_raster(fine, width=32, height=32, count=1, dtype="uint16",
                        transform=(0, 1, 0, 32, 0, -1))
    data = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16) + 1
    create_dummy_raster(coarse, band_data=data,
                        transform=(32, 2, 0, 32, 0, -2))
    outputs = {}
    for method in ("nearest", "bilinear"):
        single = tmp_path / f"{method}.tif"
        folder = tmp_path / method
        merge_rasters([str(fine), str(coarse)], str(single), resampling_method=method)
        merge_rasters([str(fine), str(coarse)], str(folder), output_tiles=True,
                      window_size=32, image_threads=2, resampling_method=method)
        reference = gdal.Open(str(single))
        tiled = gdal.BuildVRT("", [str(p) for p in sorted(folder.glob("*.tif"))])
        np.testing.assert_array_equal(reference.ReadAsArray(), tiled.ReadAsArray())
        outputs[method] = reference.ReadAsArray()
    assert np.any(outputs["nearest"] != outputs["bilinear"])


def test_single_file_overviews_and_resume(merge_sources, tmp_path):
    output = str(tmp_path / "merged.tif")
    merge_rasters(merge_sources, output, build_overviews=True)
    dataset = gdal.Open(output)
    assert dataset.GetRasterBand(1).GetOverviewCount() == 5
    dataset = None
    # Resume should short-circuit even if input files are no longer available.
    assert merge_rasters(["missing.tif"], output, resume_from_outputs="validate") == output


def test_small_mosaic_limits_pyramid_levels(tmp_path):
    source = tmp_path / "tiny.tif"
    create_dummy_raster(source, width=3, height=3, count=1)
    folder = tmp_path / "tiles"
    merge_rasters([str(source)], str(folder), output_tiles=True, build_overviews=True)
    assert (folder / "1" / "mosaic_1_1.tif").exists()
    assert not (folder / "2").exists()


def test_parallel_worker_error_propagates(merge_sources, tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "_get_executor", lambda *args, **kwargs: ThreadPoolExecutor(2))
    def fail(*args):
        raise RuntimeError("tile worker failed")
    monkeypatch.setattr(utils, "_retile_process_tile", fail)
    with pytest.raises(RuntimeError, match="tile worker failed"):
        merge_rasters(merge_sources, str(tmp_path / "tiles"),
                      output_tiles=True, image_threads=2)
    assert not list((tmp_path / "tiles").glob(".merge_rasters-*"))


def test_dask_uses_shared_executor(merge_sources, tmp_path, monkeypatch):
    submitted = []
    class Executor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, *args):
            submitted.append(args)
            future = Future()
            future.set_result(function(*args))
            return future

    def get_executor(backend, workers, **kwargs):
        assert backend == "process" and workers is None
        assert kwargs == {"concurrent_processing_backend": "dask",
                          "dask_scheduler": ("address", "tcp://scheduler:8786")}
        return Executor()

    monkeypatch.setattr(utils, "_get_executor", get_executor)
    merge_rasters(
        merge_sources, str(tmp_path / "tiles"), output_tiles=True, window_size=64,
        concurrent_processing_backend="dask",
        dask_scheduler=("address", "tcp://scheduler:8786"),
    )
    assert len(submitted) == 2


def test_vrt_links_all_bands_and_uses_existing_pyramids(tmp_path):
    source = tmp_path / "source.tif"
    create_dummy_raster(source, width=128, height=64, count=3, dtype="uint16", fill_value=100)
    folder = tmp_path / "tiles"
    merge_rasters([str(source)], str(folder), output_tiles=True, window_size=32, build_overviews=True, create_vrts="My Mosaic.vrt")
    for tile in (folder / "1").glob("*.tif"):
        dataset = gdal.Open(str(tile), gdal.GA_Update)
        for band in range(1, 4):
            dataset.GetRasterBand(band).Fill(1000 + band)
        dataset = None
    dataset = gdal.Open(str(folder / "My Mosaic.vrt"))
    assert dataset.RasterCount == 3
    for band in range(1, 4):
        raster_band = dataset.GetRasterBand(band)
        assert raster_band.GetOverviewCount() == 5
        np.testing.assert_array_equal(raster_band.ReadAsArray(buf_xsize=64, buf_ysize=32), 1000 + band)
        np.testing.assert_array_equal(raster_band.ReadAsArray(), 100)
    dataset = None
    for path in folder.rglob("*.vrt"):
        tree = ET.parse(path)
        assert all(node.attrib.get("relativeToVRT") == "1" for node in tree.findall(".//SourceFilename"))
        assert ".merge_rasters-" not in path.read_text()
    # Moving the complete folder must preserve both tile and overview links.
    moved = tmp_path / "moved"
    folder.rename(moved)
    dataset = gdal.Open(str(moved / "My Mosaic.vrt"))
    np.testing.assert_array_equal(dataset.ReadAsArray(), 100)
    np.testing.assert_array_equal(dataset.GetRasterBand(1).GetOverview(0).ReadAsArray(), 1001)


def test_vrt_overviews_preserve_extent_and_cover_truncated_edges(tmp_path):
    source = tmp_path / "odd.tif"
    create_dummy_raster(source, width=95, height=79, count=1, fill_value=100, transform=(0, 1, 0, 79, 0, -1))
    folder = tmp_path / "tiles"
    merge_rasters([str(source)], str(folder), output_tiles=True, window_size=24, build_overviews=True)
    dataset = gdal.Open(str(folder / "MergedImage.vrt"))
    for index, factor in enumerate((2, 4, 8, 16, 32)):
        overview = dataset.GetRasterBand(1).GetOverview(index)
        assert (overview.XSize, overview.YSize) == (math.ceil(95 / factor), math.ceil(79 / factor))
        np.testing.assert_array_equal(overview.ReadAsArray(), 100)
        level = gdal.Open(str(folder / str(index + 1) / "MergedImage.vrt"))
        x, dx, _, y, _, dy = level.GetGeoTransform()
        assert (x, y, x + dx * level.RasterXSize, y + dy * level.RasterYSize) == pytest.approx((0, 79, 95, 0))


def test_vrt_resume_rebuilds_missing_links_and_ignores_other_rasters(merge_sources, tmp_path):
    folder = tmp_path / "tiles"
    options = dict(output_tiles=True, window_size=32, build_overviews=True, window_scales=(2, 4))
    merge_rasters(merge_sources, str(folder), **options)
    vrt = folder / "MergedImage.vrt"
    expected = gdal.Open(str(vrt)).ReadAsArray()
    stamps = {path: path.stat().st_mtime_ns for path in folder.rglob("*.tif")}
    vrt.unlink()
    (folder / "1" / "MergedImage.vrt").unlink()
    create_dummy_raster(folder / "unrelated.tif", transform=(200, 1, 0, 10, 0, -1))
    merge_rasters(merge_sources, str(folder), resume_from_outputs="yes", **options)
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in stamps.items())
    dataset = gdal.Open(str(vrt))
    np.testing.assert_array_equal(dataset.ReadAsArray(), expected)
    assert dataset.GetRasterBand(1).GetOverviewCount() == 2
    assert dataset.GetRasterBand(1).GetOverview(0).ReadAsArray() is not None


def test_vrt_without_overviews_and_single_file_mode(merge_sources, tmp_path):
    folder = tmp_path / "tiles"
    merge_rasters(merge_sources, str(folder), output_tiles=True)
    dataset = gdal.Open(str(folder / "MergedImage.vrt"))
    assert dataset.GetRasterBand(1).GetOverviewCount() == 0
    single = tmp_path / "single.tif"
    merge_rasters(merge_sources, str(single))
    assert not (tmp_path / "MergedImage.vrt").exists()


@pytest.mark.parametrize("scales", [(2, 8), None, ()])
def test_compute_overviews_custom_scales(tmp_path, scales):
    source = tmp_path / "source.tif"
    create_dummy_raster(source, width=64, height=64, count=1)
    assert compute_overviews([str(source)], window_scales=scales) == [str(source)]
    dataset = gdal.Open(str(source))
    band = dataset.GetRasterBand(1)
    assert band.GetOverviewCount() == len(scales or ())
    for index, factor in enumerate(scales or ()):
        assert band.GetOverview(index).XSize == 64 // factor


@pytest.mark.parametrize("function", [Match.global_regression, Match.local_block_adjustment, joint_coregistration])
def test_overview_scales_validate_before_processing(function):
    with pytest.raises(ValueError, match="window_scales"):
        function(["missing.tif"], "unused", window_scales=(4, 2), build_overviews=True)
