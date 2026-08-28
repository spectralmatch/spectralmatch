import json
import os
import geopandas as gpd

from spectralmatch import pipeline
from shapely.geometry import box

from .utils_test import create_dummy_raster


def test_pipeline_full_default_flow(tmp_path):
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

    tie_points_path = tmp_path / "tie_points.json"
    tie_points_path.write_text(
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
        joint_coregistration_load_adjustments=str(tie_points_path),
        joint_coregistration_robust_loss="none",
        merge_rasters_build_overviews=False,
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


def test_pipeline_forwards_global_pif_tie_point_path(tmp_path, monkeypatch):
    input_path = tmp_path / "A.tif"
    create_dummy_raster(input_path, count=1)
    output_dir = tmp_path / "output"
    tie_path = str(tmp_path / "tie_points.json")
    captured = {}

    def fake_global_regression(**kwargs):
        captured.update(kwargs)
        return [str(output_dir / "A_Global.tif")]

    monkeypatch.setattr(
        "spectralmatch.chain.Match.global_regression",
        fake_global_regression,
    )

    pipeline(
        shared_input_images=[str(input_path)],
        shared_output_image_path=str(output_dir),
        steps=("global_regression",),
        global_regression_pif_method="flood_from_match_points",
        global_regression_pif_load_tie_points=tie_path,
    )

    assert captured["pif_load_tie_points"] == tie_path
