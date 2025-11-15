import os
import pytest

from spectralmatch import global_regression, local_block_adjustment
from .utils_test import create_dummy_raster


# global_regression
def test_global_regression_full_options_save_model(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    paths = []
    for name in ["A", "B"]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path, 16, 16, count=1, fill_value=100 if name == "A" else 120
        )
        paths.append(str(path))
    output_paths = [
        str(output_dir / f"{os.path.splitext(os.path.basename(p))[0]}_GlobalMatch.tif")
        for p in paths
    ]
    model_path = tmp_path / "adjustments.json"

    result = global_regression(
        input_images=paths,
        output_images=output_paths,
        calculation_dtype="float64",
        output_dtype="uint16",
        custom_nodata_value=0,
        io_threads=2,
        image_threads=2,
        tile_threads=2,
        window_size=16,
        save_as_cog=True,
        specify_model_images=("include", ["A"]),
        custom_mean_factor=1.0,
        custom_std_factor=1.0,
        debug_logs=True,
        save_adjustments=str(model_path),
    )

    assert all(os.path.exists(p) for p in result)
    assert model_path.exists()


def test_global_regression_full_options_load_model(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    paths = []
    for name in ["X", "Y"]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path, 16, 16, count=1, fill_value=130 if name == "X" else 110
        )
        paths.append(str(path))
    output_paths = [
        str(output_dir / f"{os.path.splitext(os.path.basename(p))[0]}_Match.tif")
        for p in paths
    ]
    model_path = tmp_path / "preload.json"

    # Pre-save model
    global_regression(
        input_images=paths, output_images=output_paths, save_adjustments=str(model_path)
    )

    new_output_paths = [p.replace("_Match", "_Reloaded") for p in output_paths]
    result = global_regression(
        input_images=paths,
        output_images=new_output_paths,
        calculation_dtype="float32",
        load_adjustments=str(model_path),
        io_threads=2,
        image_threads=2,
        tile_threads=2,
        window_size=None,
        debug_logs=True,
    )

    assert all(os.path.exists(p) for p in result)


# local_block_adjustment
def test_local_block_adjustment_all_params_save(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    block_dir = tmp_path / "blocks"
    input_dir.mkdir()
    output_dir.mkdir()
    block_dir.mkdir()
    paths = []
    for name in ["Img1", "Img2"]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path, 16, 16, count=1, fill_value=50 if name == "Img1" else 80
        )
        paths.append(str(path))
    output_paths = [
        str(output_dir / f"{os.path.splitext(os.path.basename(p))[0]}_Local.tif")
        for p in paths
    ]

    result = local_block_adjustment(
        input_images=paths,
        output_images=output_paths,
        calculation_dtype="float64",
        output_dtype="uint16",
        custom_nodata_value=99,
        io_threads=2,
        image_threads=2,
        tile_threads=2,
        window_size=16,
        save_as_cog=True,
        number_of_blocks=(2, 2),
        alpha=0.75,
        correction_method="linear",
        save_block_maps=(str(block_dir / "ref.tif"), str(block_dir / "$_block.tif")),
        override_bounds_canvas_coords=(0, 0, 16, 16),
        block_valid_pixel_threshold=0.01,
        debug_logs=True,
    )

    assert all(os.path.exists(p) for p in result)
    assert (block_dir / "ref.tif").exists()


def test_local_block_adjustment_all_params_load(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    block_dir = tmp_path / "blocks"
    input_dir.mkdir()
    output_dir.mkdir()
    block_dir.mkdir()
    paths = []
    for name in ["X", "Y"]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(path, 16, 16, count=1, fill_value=60 if name == "X" else 90)
        paths.append(str(path))
    output_paths = [
        str(output_dir / f"{os.path.splitext(os.path.basename(p))[0]}_Reloaded.tif")
        for p in paths
    ]
    ref_map = block_dir / "ref.tif"
    local_maps = [block_dir / f"{name}_block.tif" for name in ["X", "Y"]]

    # Pre-save block maps
    local_block_adjustment(
        input_images=paths,
        output_images=output_paths,
        save_block_maps=(str(ref_map), str(block_dir / "$_block.tif")),
    )

    # Rerun with load_block_maps
    new_output_paths = [p.replace("_Reloaded", "_FromLoad") for p in output_paths]
    result = local_block_adjustment(
        input_images=paths,
        output_images=new_output_paths,
        calculation_dtype="float32",
        output_dtype=None,
        load_block_maps=(str(ref_map), [str(p) for p in local_maps]),
        io_threads=2,
        image_threads=2,
        tile_threads=2,
        debug_logs=True,
    )

    assert all(os.path.exists(p) for p in result)
