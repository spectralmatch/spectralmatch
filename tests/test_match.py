import os
import pytest

from spectralmatch import Match
from spectralmatch.types_and_validation import Match as MatchValidation
from spectralmatch.pif import pif as pif_module
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

    result = Match(
        calculation_dtype="float64",
        output_dtype="uint16",
        custom_nodata_value=0,
        io_threads=2,
        image_threads=2,
        tile_threads=2,
        window_size=16,
        save_as_cog=True,
        debug_logs=True,
    ).global_regression(
        input_images=paths,
        output_images=output_paths,
        specify_model_images=("include", ["A"]),
        custom_mean_factor=1.0,
        custom_std_factor=1.0,
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
    Match().global_regression(
        input_images=paths, output_images=output_paths, save_adjustments=str(model_path)
    )

    new_output_paths = [p.replace("_Match", "_Reloaded") for p in output_paths]
    result = Match(
        calculation_dtype="float32",
        io_threads=2,
        image_threads=2,
        tile_threads=2,
        debug_logs=True,
    ).global_regression(
        input_images=paths,
        output_images=new_output_paths,
        load_adjustments=str(model_path),
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

    result = Match(
        calculation_dtype="float64",
        output_dtype="uint16",
        custom_nodata_value=99,
        io_threads=2,
        image_threads=2,
        tile_threads=2,
        window_size=16,
        save_as_cog=True,
        debug_logs=True,
    ).local_block_adjustment(
        input_images=paths,
        output_images=output_paths,
        number_of_blocks=(2, 2),
        alpha=0.75,
        correction_method="linear",
        save_block_maps=(str(block_dir / "ref.tif"), str(block_dir / "$_block.tif")),
        override_bounds_canvas_coords=(0, 0, 16, 16),
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
    Match().local_block_adjustment(
        input_images=paths,
        output_images=output_paths,
        save_block_maps=(str(ref_map), str(block_dir / "$_block.tif")),
    )

    # Rerun with load_block_maps
    new_output_paths = [p.replace("_Reloaded", "_FromLoad") for p in output_paths]
    result = Match(
        calculation_dtype="float32",
        io_threads=2,
        image_threads=2,
        tile_threads=2,
        debug_logs=True,
    ).local_block_adjustment(
        input_images=paths,
        output_images=new_output_paths,
        load_block_maps=(str(ref_map), [str(p) for p in local_maps]),
    )

    assert all(os.path.exists(p) for p in result)


def test_pif_none_sample_limits_skip_sampling_and_minimums(monkeypatch):
    sampled = {"called": False}

    monkeypatch.setattr(
        pif_module,
        "_build_overlap_vrts",
        lambda reference_path, sensed_path, tmpdir: ("ref.vrt", "sensed.vrt", 10, 10, (0, 1, 0, 0, 0, -1), ""),
    )
    monkeypatch.setattr(pif_module, "_build_valid_mask_raster", lambda *args, **kwargs: "valid.tif")
    monkeypatch.setattr(pif_module, "_build_inz_stable_mask_raster", lambda *args, **kwargs: "stable.tif")
    monkeypatch.setattr(pif_module, "geometric_correction", lambda *args, **kwargs: ("corrected.tif", [(1, 1, 1, 1)]))
    monkeypatch.setattr(pif_module, "_build_seed_mask_raster", lambda *args, **kwargs: "seed.tif")
    monkeypatch.setattr(pif_module, "_combine_masks_raster", lambda *args, **kwargs: "pif_mask.tif")
    monkeypatch.setattr(pif_module, "_count_mask_pixels", lambda *args, **kwargs: 5)

    def _unexpected_sample(*args, **kwargs):
        sampled["called"] = True
        raise AssertionError("_sample_mask_raster should not run when max_samples=None")

    monkeypatch.setattr(pif_module, "_sample_mask_raster", _unexpected_sample)
    monkeypatch.setattr(
        pif_module,
        "_masked_band_stats",
        lambda raster_path, band_index, mask_path: {"mean": 1.0, "std": 0.5, "size": 5},
    )

    pair_stats, whole_updates = pif_module._calculate_pair_pif_stats(
        reference_path="a.tif",
        sensed_path="b.tif",
        reference_name="A",
        sensed_name="B",
        num_bands=1,
        nodata_value=0,
        calculation_dtype="float32",
        red_band_index=None,
        nir_band_index=None,
        vegetation_threshold=0.2,
        inz_threshold=0.25,
        region_radius=5,
        max_samples=None,
        min_samples=None,
        feature_method="orb",
        cache=None,
        io_threads=None,
        tile_threads=None,
        save_inz_path=None,
        debug_logs=False,
    )

    assert sampled["called"] is False
    assert pair_stats["A"]["B"][0]["size"] == 5
    assert pair_stats["B"]["A"][0]["size"] == 5
    assert whole_updates["A"][0]["size"] == 5
    assert whole_updates["B"][0]["size"] == 5


def test_pif_save_inz_two_placeholder_path_resolution():
    resolved = pif_module._resolve_pair_output_path(
        "/tmp/$_to_$_INZ.tif",
        "SensedImage",
        "ReferenceImage",
        0,
        2,
    )
    assert resolved == "/tmp/SensedImage_to_ReferenceImage_INZ.tif"


def test_pif_save_inz_validation_rejects_single_placeholder():
    with pytest.raises(ValueError, match="exactly two '\\$' placeholders"):
        MatchValidation.validate_global_regression(pif_save_inz="/tmp/$_INZ.tif")
