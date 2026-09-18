import json
import os
import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal

from spectralmatch import pipeline
from shapely.geometry import box

from .utils_test import create_dummy_raster


def test_pipeline_tiled_merge_survives_intermediate_cleanup(tmp_path):
    input_paths = []
    for name, origin, value in [("A", 0, 50), ("B", 16, 75)]:
        path = tmp_path / f"{name}.tif"
        create_dummy_raster(path, width=32, height=32, count=1, transform=(origin, 1, 0, 32, 0, -1), fill_value=value)
        input_paths.append(str(path))
    output_dir = tmp_path / "tiles"
    temp_dir = tmp_path / "work"

    result = pipeline(
        shared_input_images=input_paths,
        shared_output_image_path=str(output_dir),
        shared_temp_dir=str(temp_dir),
        delete_previous_step=True,
        steps=("align", "merge"),
        shared_cache=None,
        shared_image_threads=2,
        shared_io_threads=1,
        shared_tile_threads=1,
        shared_window_size=16,
        shared_window_scales=(2, 4),
        merge_rasters_output_tiles=True,
        merge_rasters_build_overviews=True,
        merge_rasters_overlap=4,
        merge_rasters_resampling_method="bilinear",
        merge_rasters_custom_tiles_csv="index.csv",
        merge_rasters_create_vrts="Mosaic.vrt",
    )

    assert result["output"] == result["merge_rasters"] == str(output_dir)
    assert not temp_dir.exists()
    assert len(list(output_dir.glob("*.tif"))) > 1
    for level in (output_dir, output_dir / "1", output_dir / "2"):
        assert (level / "index.csv").is_file()
        assert (level / "Mosaic.vrt").is_file()
    with gdal.Open(str(output_dir / "Mosaic.vrt")) as dataset:
        assert (dataset.RasterXSize, dataset.RasterYSize) == (48, 32)
        np.testing.assert_array_equal(dataset.ReadAsArray()[:, :16], 50)
        np.testing.assert_array_equal(dataset.ReadAsArray()[:, 16:], 75)
        band = dataset.GetRasterBand(1)
        assert band.GetOverviewCount() == 2
        for i in range(2):
            assert np.all(band.GetOverview(i).ReadAsArray() > 0)


@pytest.mark.parametrize("output_tiles,backend", [(False, "dask"), (True, "process_pool"), (True, "dask")])
def test_pipeline_forwards_merge_concurrency_and_resume(tmp_path, monkeypatch, output_tiles, backend):
    input_path = tmp_path / "A.tif"
    create_dummy_raster(input_path, count=1)
    captured = {}

    def fake_merge(**kwargs):
        captured.update(kwargs)
        return kwargs["output_image_path"]

    monkeypatch.setattr("spectralmatch.chain.merge_rasters", fake_merge)
    workers = 2 if backend == "process_pool" else None
    scheduler = ("address", "tcp://localhost:8786") if backend == "dask" else None
    pipeline(
        shared_input_images=[str(input_path)],
        shared_output_image_path=str(tmp_path / ("tiles" if output_tiles else "merged.tif")),
        steps=("merge",),
        shared_resume_from_steps="validate",
        shared_image_threads=workers,
        shared_concurrent_processing_backend=backend,
        shared_dask_scheduler=scheduler,
        merge_rasters_resolution=2,
        merge_rasters_output_tiles=output_tiles,
        merge_rasters_resampling_method="cubic",
    )
    assert captured["output_tiles"] is output_tiles
    assert captured["image_threads"] == (workers if output_tiles else None)
    assert captured["concurrent_processing_backend"] == (backend if output_tiles else None)
    assert captured["dask_scheduler"] == (scheduler if output_tiles else None)
    assert captured["resume_from_outputs"] == "validate"
    assert captured["resampling_method"] == "cubic"
    assert captured["resolution"] == 2
    assert captured["create_vrts"] == "MergedImage.vrt"


@pytest.mark.parametrize("output_name,options,error", [
    ("tiles.tif", {"merge_rasters_output_tiles": True}, "must be a folder"),
    ("merged.tif", {"merge_rasters_overlap": 4}, "require output_tiles=True"),
    ("tiles", {"merge_rasters_output_tiles": True, "shared_window_scales": (2, 8)}, "consecutive powers"),
])
def test_pipeline_rejects_invalid_merge_options_before_cleanup(tmp_path, output_name, options, error):
    temp_dir = tmp_path / "work"
    temp_dir.mkdir()
    sentinel = temp_dir / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match=error):
        pipeline(
            shared_input_images=[str(tmp_path / "unused.tif")],
            shared_output_image_path=str(tmp_path / output_name),
            shared_temp_dir=str(temp_dir),
            steps=("merge",),
            **options,
        )
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("overview_options", [
    {},
    {
        "shared_window_scales": (2, 8),
        "joint_coregistration_build_overviews": True,
        "global_regression_build_overviews": True,
        "local_block_adjustment_build_overviews": True,
        "merge_rasters_build_overviews": True,
    },
])
def test_pipeline_full_default_flow(tmp_path, overview_options):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_path = tmp_path / "merged.tif"

    input_paths = []
    for name, x_origin, fill_value in [
        ("A", 0, 100),
        ("B", 8, 120),
    ]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path,
            width=32,
            height=32,
            count=1,
            transform=(x_origin, 1, 0, 32, 0, -1),
            fill_value=fill_value,
        )
        input_paths.append(str(path))

    ties_path = tmp_path / "ties.json"
    ties_path.write_text(
        json.dumps(
            {
                "tie_points": [
                    {
                        "image_1": "A",
                        "image_2": "B",
                        "points": [[[10.0, 10.0], [2.0, 10.0]]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results = pipeline(
        shared_input_images=input_paths,
        shared_output_image_path=str(output_path),
        delete_temp_dir=False,
        shared_debug_logs=True,
        shared_window_size=16,
        joint_coregistration_local_model="none",
        joint_coregistration_tie_load_path=str(ties_path),
        joint_coregistration_tie_robust_loss="none",
        global_regression_pif_method="entire",
        **{"merge_rasters_build_overviews": False, **overview_options},
    )

    assert os.path.exists(results["output"])
    assert os.path.isdir(results["temp_dir"])
    assert results["output"] == str(output_path)
    assert results["num_input_images"] == 2
    assert "start_time" in results
    assert "end_time" in results
    assert "duration_seconds" in results
    assert "resolved_shared_cache" in results
    assert "resolved_shared_image_threads" in results
    assert "resolved_shared_io_threads" in results
    assert "resolved_shared_tile_threads" in results
    assert results["steps"][0] == "joint_coregistration"
    assert "joint_coregistration" in results
    assert "align" not in results["steps"]
    for stage in ("joint_coregistration", "global_regression", "local_block_adjustment", "merge_rasters"):
        scales = overview_options.get("shared_window_scales")
        if scales is not None:
            outputs = results[stage]
            for path in outputs if isinstance(outputs, list) else [outputs]:
                dataset = gdal.Open(path)
                band = dataset.GetRasterBand(1)
                assert band.GetOverviewCount() == len(scales)
                assert [band.GetOverview(i).XSize for i in range(len(scales))] == [(dataset.RasterXSize + scale - 1) // scale for scale in scales]


def test_pipeline_merge_only_with_custom_temp_dir(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    input_paths = []
    for name, x_origin, fill_value in [
        ("A", 0, 50),
        ("B", 4, 75),
    ]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path,
            width=16,
            height=16,
            count=1,
            transform=(x_origin, 1, 0, 16, 0, -1),
            fill_value=fill_value,
        )
        input_paths.append(str(path))

    custom_temp_dir = tmp_path / "pipeline_temp"
    output_path = tmp_path / "merged.tif"

    results = pipeline(
        shared_input_images=input_paths,
        shared_output_image_path=str(output_path),
        shared_temp_dir=str(custom_temp_dir),
        steps=("merge",),
        merge_rasters_build_overviews=False,
    )

    assert results["temp_dir"] == str(custom_temp_dir)
    assert os.path.exists(results["output"])
    assert results["output"] == str(output_path)
    assert results["num_input_images"] == 2
    assert "global_regression" not in results
    assert "local_block_adjustment" not in results
    assert "align_rasters" not in results
    assert "voronoi_center_seamline" not in results
    assert "mask_rasters" not in results


def test_pipeline_weighted_seamline_step(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_paths = []
    for name, x_origin, fill_value in [
        ("A", 0, 50),
        ("B", 4, 75),
    ]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path,
            width=16,
            height=16,
            count=1,
            transform=(x_origin, 1, 0, 16, 0, -1),
            fill_value=fill_value,
        )
        input_paths.append(str(path))

    polygons_path = tmp_path / "footprints.gpkg"
    gdf = gpd.GeoDataFrame(
        [
            {
                "image": input_paths[0],
                "quality": 1.0,
                "geometry": box(0, 0, 10, 10),
            },
            {
                "image": input_paths[1],
                "quality": 2.0,
                "geometry": box(5, 0, 15, 10),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf.to_file(polygons_path, layer="footprints", driver="GPKG")

    results = pipeline(
        shared_input_images=input_paths,
        shared_output_image_path=str(tmp_path / "seamlines.gpkg"),
        shared_temp_dir=str(tmp_path / "pipeline_temp"),
        delete_temp_dir=False,
        steps=("weighted_seamline",),
        weighted_seamline_input_polygons=str(polygons_path),
        weighted_seamline_rank_function="{quality}",
        weighted_seamline_input_layer="footprints",
    )

    assert results["output"] == str(tmp_path / "seamlines.gpkg")
    assert os.path.exists(results["weighted_seamline"])


def test_pipeline_resume_from_existing_merge_output_yes(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_path = tmp_path / "merged.tif"

    input_paths = []
    for name, x_origin, fill_value in [
        ("A", 0, 50),
        ("B", 4, 75),
    ]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path,
            width=16,
            height=16,
            count=1,
            transform=(x_origin, 1, 0, 16, 0, -1),
            fill_value=fill_value,
        )
        input_paths.append(str(path))

    create_dummy_raster(
        output_path,
        width=32,
        height=16,
        count=1,
        transform=(0, 1, 0, 16, 0, -1),
        fill_value=123,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("merge_rasters should have resumed before building the VRT")

    monkeypatch.setattr("spectralmatch.utils.gdal.BuildVRT", fail_if_called)

    results = pipeline(
        shared_input_images=input_paths,
        shared_output_image_path=str(output_path),
        shared_resume_from_steps="yes",
        steps=("merge",),
        merge_rasters_build_overviews=False,
    )

    assert results["output"] == str(output_path)


def test_pipeline_resume_from_existing_merge_output_validate_reruns_invalid(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_path = tmp_path / "merged.tif"

    input_paths = []
    for name, x_origin, fill_value in [
        ("A", 0, 50),
        ("B", 4, 75),
    ]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path,
            width=16,
            height=16,
            count=1,
            transform=(x_origin, 1, 0, 16, 0, -1),
            fill_value=fill_value,
        )
        input_paths.append(str(path))

    output_path.write_text("not a raster", encoding="utf-8")

    results = pipeline(
        shared_input_images=input_paths,
        shared_output_image_path=str(output_path),
        shared_resume_from_steps="validate",
        steps=("merge",),
        merge_rasters_build_overviews=False,
    )

    assert results["output"] == str(output_path)
    assert os.path.getsize(output_path) > 0


def test_pipeline_delete_previous_step_removes_replaced_intermediate(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "global_outputs"

    input_paths = []
    for name, x_origin, fill_value in [
        ("A", 0, 50),
        ("B", 4, 75),
    ]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path,
            width=16,
            height=16,
            count=1,
            transform=(x_origin, 1, 0, 16, 0, -1),
            fill_value=fill_value,
        )
        input_paths.append(str(path))

    temp_dir = tmp_path / "pipeline_temp"

    results = pipeline(
        shared_input_images=input_paths,
        shared_output_image_path=str(output_dir),
        shared_temp_dir=str(temp_dir),
        delete_temp_dir=False,
        delete_previous_step=True,
        steps=("global_regression", "local_block_adjustment"),
        global_regression_pif_method="entire",
    )

    assert results["output"] == [
        str(output_dir / "A_Global_Local.tif"),
        str(output_dir / "B_Global_Local.tif"),
    ]
    assert not (temp_dir / "global").exists()


@pytest.mark.parametrize("seamline_step", ["voronoi_center_seamline", "weighted_seamline"])
@pytest.mark.parametrize("delete_previous_step", [False, True])
def test_pipeline_footprints_through_seamlines_mask_and_merge(tmp_path, seamline_step, delete_previous_step):
    input_paths = []
    for name, origin, value in [("A", 0, 50), ("B", 16, 75)]:
        path = tmp_path / f"{name}.tif"
        data = np.full((2, 32, 32), value, dtype=np.uint8)
        data[1, :, :4] = 0
        create_dummy_raster(
            path, band_data=data, crs="EPSG:32604",
            transform=(origin, 1, 0, 32, 0, -1),
        )
        input_paths.append(str(path))
    temp_dir = tmp_path / "work"
    output = tmp_path / "mosaic.tif"

    result = pipeline(
        shared_input_images=input_paths,
        shared_output_image_path=str(output),
        shared_temp_dir=str(temp_dir),
        delete_temp_dir=False,
        delete_previous_step=delete_previous_step,
        steps=("align", "create_footprints", "postprocess_footprints", seamline_step, "mask", "merge"),
        shared_cache=None,
        shared_image_threads=None,
        shared_io_threads=1,
        shared_tile_threads=1,
        shared_window_size=16,
        align_rasters_resampling_method="nearest",
        create_footprints_band=2,
        create_footprints_eight_connected=False,
        create_footprints_image_field_name="scene",
        create_footprints_output_layer="raw",
        postprocess_footprints_output_layer="processed",
        postprocess_footprints_edge_distance=0,
        postprocess_footprints_smoothing_radius=0.5,
        postprocess_footprints_simplify_tolerance=0,
        postprocess_footprints_area_rank=None,
        **{
            f"{seamline_step}_image_field_name": "scene",
            f"{seamline_step}_output_layer": "cuts",
                **({"weighted_seamline_rank_function": "1 + ({scene} == 'A_Align')"} if seamline_step == "weighted_seamline" else {}),
        },
    )

    assert result["output"] == str(output)
    with gdal.Open(str(output)) as dataset:
        assert dataset.RasterCount == 2
        assert np.count_nonzero(dataset.ReadAsArray()) > 0
        assert dataset.GetGeoTransform()[0] == 4
    seamlines = gpd.read_file(result[seamline_step], layer="cuts")
    assert set(seamlines["scene"]) == {os.path.splitext(os.path.basename(path))[0] for path in result["align"]}
    if delete_previous_step:
        assert not (temp_dir / "aligned").exists()
        assert not (temp_dir / "clip").exists()
        assert not os.path.exists(result["create_footprints"])
        assert not os.path.exists(result["postprocess_footprints"])
    else:
        raw = gpd.read_file(result["create_footprints"], layer="raw")
        processed = gpd.read_file(result["postprocess_footprints"], layer="processed")
        assert list(raw.area) == [28 * 32, 28 * 32]
        assert all(processed.area < raw.area)
        assert set(processed["scene"]) == set(raw["scene"])
    assert all(os.path.exists(path) for path in input_paths)


@pytest.mark.parametrize("step", ["create_footprints", "postprocess_footprints"])
def test_pipeline_final_footprints_output_and_resume(tmp_path, monkeypatch, step):
    source = tmp_path / "A.tif"
    create_dummy_raster(source, crs="EPSG:32604", count=1)
    output = tmp_path / "output.gpkg"
    options = dict(
        shared_input_images=[str(source)],
        shared_output_image_path=str(output),
        shared_temp_dir=str(tmp_path / "work"),
        delete_previous_step=True,
        shared_image_threads=None,
        steps=("create_footprints",) if step == "create_footprints" else ("create_footprints", "postprocess_footprints"),
        postprocess_footprints_edge_distance=0,
        postprocess_footprints_smoothing_radius=0,
        postprocess_footprints_simplify_tolerance=0,
    )
    result = pipeline(**options)
    assert result["output"] == result[step] == str(output)
    assert output.is_file()
    assert not (tmp_path / "work").exists()
    assert list(gpd.read_file(output)["image"]) == ["A"]
    modified = output.stat().st_mtime_ns

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Existing final footprints should be reused")

    worker = "_footprint_from_image" if step == "create_footprints" else "_postprocess_polygon"
    monkeypatch.setattr(f"spectralmatch.seamline.footprints.{worker}", fail_if_called)
    resumed = pipeline(**options, shared_resume_from_steps="validate")
    assert resumed["output"] == str(output)
    assert output.stat().st_mtime_ns == modified


@pytest.mark.parametrize("input_layer", [None, "external"])
@pytest.mark.parametrize("step", ["postprocess_footprints", "voronoi_center_seamline", "weighted_seamline"])
def test_pipeline_explicit_polygons_override_generated_footprints(tmp_path, step, input_layer):
    source = tmp_path / "A.tif"
    create_dummy_raster(source, crs="EPSG:32604", count=1)
    polygons = tmp_path / "external.gpkg"
    external = gpd.GeoDataFrame(
        {"image": ["external"], "quality": [7], "geometry": [box(2, 2, 8, 8)]},
        crs="EPSG:32604",
    )
    external.to_file(polygons, layer="external", driver="GPKG")

    result = pipeline(
        shared_input_images=[str(source)],
        shared_output_image_path=str(tmp_path / "result.gpkg"),
        steps=("create_footprints", step),
        shared_image_threads=None,
        create_footprints_output_layer="generated",
        postprocess_footprints_edge_distance=0,
        postprocess_footprints_smoothing_radius=0,
        postprocess_footprints_simplify_tolerance=0,
        weighted_seamline_rank_function="{quality}",
        **{f"{step}_input_polygons": str(polygons), f"{step}_input_layer": input_layer},
    )

    actual = gpd.read_file(result["output"])
    assert list(actual["image"]) == ["external"]
    assert actual.geometry.iloc[0].equals(external.geometry.iloc[0])
    if step in {"postprocess_footprints", "weighted_seamline"}:
        assert list(actual["quality"]) == [7]
    assert polygons.is_file()


def test_pipeline_postprocessed_attributes_feed_weighted_seamline(tmp_path):
    source = tmp_path / "A.tif"
    create_dummy_raster(source, crs="EPSG:32604", count=1)
    polygons = tmp_path / "external.gpkg"
    gpd.GeoDataFrame(
        {"image": ["A", "B"], "quality": [1, 2], "geometry": [box(0, 0, 10, 10), box(5, 0, 15, 10)]},
        crs="EPSG:32604",
    ).to_file(polygons, layer="external", driver="GPKG")
    result = pipeline(
        shared_input_images=[str(source)],
        shared_output_image_path=str(tmp_path / "result.gpkg"),
        steps=("postprocess_footprints", "weighted_seamline"),
        shared_image_threads=None,
        postprocess_footprints_input_polygons=str(polygons),
        postprocess_footprints_input_layer="external",
        postprocess_footprints_output_layer="processed",
        postprocess_footprints_edge_distance=0,
        postprocess_footprints_smoothing_radius=0,
        postprocess_footprints_simplify_tolerance=0,
        weighted_seamline_rank_function="{quality}",
    )
    actual = gpd.read_file(result["output"]).set_index("image")
    assert actual.loc["A"].geometry.area == 50
    assert actual.loc["B"].geometry.area == 100


def test_pipeline_implicit_footprints_use_creation_options(tmp_path):
    source = tmp_path / "A.tif"
    data = np.ones((2, 10, 10), dtype=np.uint8)
    data[1, :, :4] = 0
    create_dummy_raster(source, band_data=data, crs="EPSG:32604")
    result = pipeline(
        shared_input_images=[str(source)],
        shared_output_image_path=str(tmp_path / "seamlines.gpkg"),
        shared_temp_dir=str(tmp_path / "work"),
        delete_temp_dir=False,
        steps=("voronoi_center_seamline",),
        shared_image_threads=None,
        create_footprints_band=2,
        create_footprints_eight_connected=False,
        create_footprints_output_layer="valid_pixels",
        voronoi_center_seamline_image_field_name="scene",
    )
    footprints = gpd.read_file(result["create_footprints"], layer="valid_pixels")
    seamlines = gpd.read_file(result["output"])
    assert list(seamlines["scene"]) == ["A"]
    assert footprints.geometry.iloc[0].equals(box(4, 0, 10, 10))
    assert seamlines.geometry.iloc[0].equals(footprints.geometry.iloc[0])


def test_pipeline_keeps_footprints_across_intervening_raster_step(tmp_path):
    source = tmp_path / "A.tif"
    create_dummy_raster(source, count=1, crs="EPSG:32604")
    result = pipeline(
        shared_input_images=[str(source)],
        shared_output_image_path=str(tmp_path / "processed.gpkg"),
        shared_temp_dir=str(tmp_path / "work"),
        delete_temp_dir=False,
        delete_previous_step=True,
        shared_image_threads=None,
        steps=("create_footprints", "align", "postprocess_footprints"),
        create_footprints_output_layer="raw",
        postprocess_footprints_edge_distance=0,
        postprocess_footprints_smoothing_radius=0,
        postprocess_footprints_simplify_tolerance=0,
    )
    actual = gpd.read_file(result["output"])
    assert list(actual["image"]) == ["A"]
    assert actual.geometry.iloc[0].equals(box(0, 0, 10, 10))
    assert not os.path.exists(result["create_footprints"])


@pytest.mark.parametrize("step", ["postprocess_footprints", "weighted_seamline"])
def test_pipeline_missing_polygon_source(tmp_path, step):
    source = tmp_path / "A.tif"
    create_dummy_raster(source, count=1)
    with pytest.raises(ValueError, match=f"{step} requires input polygons"):
        pipeline(
            shared_input_images=[str(source)],
            shared_output_image_path=str(tmp_path / "result.gpkg"),
            steps=(step,),
        )
