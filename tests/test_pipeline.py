import os

from spectralmatch import pipeline

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

    results = pipeline(
        shared_input_images=input_paths,
        shared_output_image_path=str(output_path),
        delete_temp_dir=False,
        shared_debug_logs=True,
        shared_window_size=16,
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
        matching_order=(),
        align_method=None,
        seamline_method=None,
        clip_method=None,
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
