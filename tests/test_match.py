import json
import os

import numpy as np
import pytest
from osgeo import gdal

from spectralmatch import Match
from spectralmatch.types_and_validation import Match as MatchValidation
from spectralmatch.pif import pif as pif_module
from .utils_test import create_dummy_raster


def _make_two_test_rasters(tmp_path, names_and_values, input_dir_name="input", output_dir_name="output", suffix="_Out.tif"):
    input_dir = tmp_path / input_dir_name
    output_dir = tmp_path / output_dir_name
    input_dir.mkdir()
    output_dir.mkdir()
    paths = []
    for name, value in names_and_values:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(path, 16, 16, count=1, fill_value=value)
        paths.append(str(path))
    output_paths = [
        str(output_dir / f"{os.path.splitext(os.path.basename(p))[0]}{suffix}")
        for p in paths
    ]
    return paths, output_paths, input_dir, output_dir


def _global_shared_kwargs(**overrides):
    kwargs = {
        "calculation_dtype": "float64",
        "output_dtype": "uint16",
        "custom_nodata_value": 0,
        "io_threads": 2,
        "image_threads": 2,
        "tile_threads": 2,
        "window_size": 16,
        "save_as_cog": True,
        "debug_logs": True,
        "pif_method": "entire",
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("method", ["global_regression", "local_block_adjustment"])
def test_match_custom_overview_scales(tmp_path, method):
    paths, outputs, _, _ = _make_two_test_rasters(tmp_path, [("A", 100), ("B", 120)])
    options = {"pif_method": "entire"} if method == "global_regression" else {"number_of_blocks": 4}
    getattr(Match, method)(paths, outputs, build_overviews=True, window_scales=(2, 4), **options)
    for output in outputs:
        dataset = gdal.Open(output)
        band = dataset.GetRasterBand(1)
        assert band.GetOverviewCount() == 2
        assert [band.GetOverview(i).XSize for i in range(2)] == [8, 4]


# global_regression
def test_global_regression_full_options_save_model(tmp_path):
    paths, output_paths, _, _ = _make_two_test_rasters(
        tmp_path,
        [("A", 100), ("B", 120)],
        input_dir_name="in",
        output_dir_name="out",
        suffix="_GlobalMatch.tif",
    )
    model_path = tmp_path / "adjustments.json"

    result = Match.global_regression(
        input_images=paths,
        output_images=output_paths,
        **_global_shared_kwargs(),
        specify_model_images=("include", ["A"]),
        custom_mean_factor=1.0,
        custom_std_factor=1.0,
        save_adjustments=str(model_path),
    )

    assert all(os.path.exists(p) for p in result)
    assert model_path.exists()


def test_global_regression_full_options_load_model(tmp_path):
    paths, output_paths, _, _ = _make_two_test_rasters(
        tmp_path,
        [("X", 130), ("Y", 110)],
        suffix="_Match.tif",
    )
    model_path = tmp_path / "preload.json"

    # Pre-save model
    Match.global_regression(
        input_images=paths,
        output_images=output_paths,
        save_adjustments=str(model_path),
        pif_method="entire",
    )

    new_output_paths = [p.replace("_Match", "_Reloaded") for p in output_paths]
    result = Match.global_regression(
        input_images=paths,
        output_images=new_output_paths,
        **_global_shared_kwargs(
            calculation_dtype="float32",
            output_dtype=None,
            save_as_cog=False,
        ),
        load_adjustments=str(model_path),
    )

    assert all(os.path.exists(p) for p in result)


def test_global_regression_method_level_shared_params(tmp_path):
    paths, output_paths, _, _ = _make_two_test_rasters(
        tmp_path,
        [("A", 100), ("B", 120)],
        suffix="_Global.tif",
    )

    result = Match.global_regression(
        input_images=paths,
        output_images=output_paths,
        **_global_shared_kwargs(),
    )

    assert all(os.path.exists(p) for p in result)


def test_global_regression_forwards_pif_load_tie_points(tmp_path, monkeypatch):
    paths, output_paths, _, _ = _make_two_test_rasters(
        tmp_path,
        [("A", 100), ("B", 120)],
        suffix="_TiePointGlobal.tif",
    )
    tie_path = str(tmp_path / "tie_points.json")
    captured = {}

    def fake_flood_from_match_points(**kwargs):
        captured.update(kwargs)
        return np.asarray([[[1.0], [0.0], [1.0], [0.0]]])

    monkeypatch.setattr(
        pif_module.Pif,
        "flood_from_match_points",
        staticmethod(fake_flood_from_match_points),
    )

    result = Match.global_regression(
        input_images=paths,
        output_images=output_paths,
        pif_method="flood_from_match_points",
        pif_load_tie_points=tie_path,
    )

    assert result == output_paths
    assert captured["load_tie_points"] == tie_path


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

    result = Match.local_block_adjustment(
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
        debug_logs=True,
        number_of_blocks=(2, 2),
        alpha=0.75,
        correction_method="linear",
        save_block_maps=(str(block_dir / "ref.tif"), str(block_dir / "$_block.tif")),
        override_bounds_canvas_coords=(0, 0, 16, 16),
    )

    assert all(os.path.exists(p) for p in result)
    assert (block_dir / "ref.tif").exists()


def test_local_block_adjustment_method_level_shared_params(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
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

    result = Match.local_block_adjustment(
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
        debug_logs=True,
        number_of_blocks=(2, 2),
        alpha=0.75,
        correction_method="linear",
        override_bounds_canvas_coords=(0, 0, 16, 16),
    )

    assert all(os.path.exists(p) for p in result)


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
    Match.local_block_adjustment(
        input_images=paths,
        output_images=output_paths,
        save_block_maps=(str(ref_map), str(block_dir / "$_block.tif")),
    )

    # Rerun with load_block_maps
    new_output_paths = [p.replace("_Reloaded", "_FromLoad") for p in output_paths]
    result = Match.local_block_adjustment(
        input_images=paths,
        output_images=new_output_paths,
        calculation_dtype="float32",
        io_threads=2,
        image_threads=2,
        tile_threads=2,
        debug_logs=True,
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
    monkeypatch.setattr(pif_module, "_coregister_overlap", lambda *args, **kwargs: ("corrected.tif", [(1, 1, 1, 1)]))
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
        lambda raster_path, band_index, mask_path, mask_pixel_count=None: {
            "mean": 1.0,
            "std": 0.5,
            "size": mask_pixel_count,
        },
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
        source_tie_points=None,
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


def test_pif_mask_count_uses_exact_binary_histogram(tmp_path):
    mask_path = tmp_path / "pif_count_mask.tif"
    create_dummy_raster(mask_path, width=8, height=6, count=1, fill_value=0)
    dataset = gdal.Open(str(mask_path), gdal.GA_Update)
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1, 2] = 1
    mask[3, 4] = 1
    mask[5, 7] = 1
    dataset.GetRasterBand(1).WriteArray(mask)
    dataset = None

    assert pif_module._count_mask_pixels(str(mask_path)) == 3


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
        MatchValidation._validate_global_regression(pif_save_inz="/tmp/$_INZ.tif")


def test_pif_load_tie_points_validation_requires_flood_method():
    with pytest.raises(
        ValueError,
        match="pif_load_tie_points requires pif_method='flood_from_match_points'",
    ):
        MatchValidation._validate_global_regression(
            pif_method="entire",
            pif_load_tie_points="tie_points.json",
        )

    with pytest.raises(ValueError, match="must be a string or None"):
        MatchValidation._validate_global_regression(
            pif_method="flood_from_match_points",
            pif_load_tie_points=[],
        )


def test_pif_loads_compact_tie_point_json_for_current_pair(tmp_path, monkeypatch):
    paths, _, _, _ = _make_two_test_rasters(
        tmp_path,
        [("A", 100), ("B", 120)],
    )
    tie_path = tmp_path / "tie_points.json"
    tie_path.write_text(
        json.dumps(
            {
                "tie_points": [
                    {
                        "image_1": "A",
                        "image_2": "B",
                        "points": [
                            [[3, 4], [5, 6]],
                            [[7, 8], [9, 10]],
                            [[11, 12], [13, 14]],
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_pair_stats(*args):
        captured["points"] = args[15]
        stats = {"mean": 1.0, "std": 0.5, "size": 2}
        return {"A": {"B": {0: stats}}, "B": {"A": {0: stats}}}, {
            "A": {0: stats},
            "B": {0: stats},
        }

    monkeypatch.setattr(pif_module, "_calculate_pair_pif_stats", fake_pair_stats)
    monkeypatch.setattr(
        pif_module,
        "_solve_pif_global_model",
        lambda **kwargs: np.zeros((1, 4, 1)),
    )

    pif_module.Pif.flood_from_match_points(
        input_images=paths,
        overlapping_pairs=(("A", "B"),),
        load_tie_points=str(tie_path),
    )

    assert captured["points"] == pytest.approx(
        np.asarray([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]], dtype=float)
    )


def test_pif_loaded_json_requires_every_processed_overlap_pair(tmp_path):
    paths, _, _, _ = _make_two_test_rasters(
        tmp_path,
        [("A", 100), ("B", 120)],
    )
    tie_path = tmp_path / "missing_pair.json"
    tie_path.write_text(json.dumps({"tie_points": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing overlap pair: A <-> B"):
        pif_module.Pif.flood_from_match_points(
            input_images=paths,
            overlapping_pairs=(("A", "B"),),
            load_tie_points=str(tie_path),
        )


def test_pif_loaded_json_rejects_insufficient_pair_points(tmp_path):
    paths, _, _, _ = _make_two_test_rasters(
        tmp_path,
        [("A", 100), ("B", 120)],
    )
    tie_path = tmp_path / "insufficient_pair.json"
    tie_path.write_text(
        json.dumps(
            {
                "tie_points": [
                    {
                        "image_1": "A",
                        "image_2": "B",
                        "points": [[[3, 4], [5, 6]], [[7, 8], [9, 10]]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must contain at least 3 points"):
        pif_module.Pif.flood_from_match_points(
            input_images=paths,
            overlapping_pairs=(("A", "B"),),
            load_tie_points=str(tie_path),
        )


def test_pif_unusable_loaded_points_raise_before_feature_matching(monkeypatch):
    monkeypatch.setattr(
        pif_module,
        "_build_overlap_vrts",
        lambda *args: ("ref.vrt", "sensed.vrt", 10, 10, (0, 1, 0, 0, 0, -1), ""),
    )
    monkeypatch.setattr(pif_module, "_build_valid_mask_raster", lambda *args: "valid.tif")
    monkeypatch.setattr(
        pif_module,
        "_source_tie_points_to_overlap_pairs",
        lambda *args: [],
    )

    def fail_correction(*args, **kwargs):
        raise AssertionError("Overlap coregistration and ORB must not run")

    monkeypatch.setattr(pif_module, "_coregister_overlap", fail_correction)

    with pytest.raises(ValueError, match="fewer than 3 usable points"):
        pif_module._calculate_pair_pif_stats(
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
            source_tie_points=np.ones((3, 4)),
            cache=None,
            io_threads=None,
            tile_threads=None,
            save_inz_path=None,
            debug_logs=False,
        )


def test_loaded_source_pixels_convert_to_overlap_row_column_pairs(tmp_path):
    reference = tmp_path / "A.tif"
    sensed = tmp_path / "B.tif"
    create_dummy_raster(
        reference,
        width=20,
        height=20,
        count=1,
        transform=(100, 2, 0, 200, 0, -2),
        crs="EPSG:3857",
    )
    create_dummy_raster(
        sensed,
        width=20,
        height=20,
        count=1,
        transform=(104, 2, 0, 200, 0, -2),
        crs="EPSG:3857",
    )
    overlap_dir = tmp_path / "overlap"
    overlap_dir.mkdir()
    ref_vrt, sensed_vrt, width, height, gt, _ = pif_module._build_overlap_vrts(
        str(reference), str(sensed), str(overlap_dir)
    )
    valid_mask = overlap_dir / "valid.tif"
    create_dummy_raster(
        valid_mask,
        width=width,
        height=height,
        count=1,
        transform=gt,
        crs="EPSG:3857",
        fill_value=1,
    )
    source_points = np.asarray(
        [
            [3, 4, 1, 4],
            [5, 6, 3, 6],
            [7, 8, 5, 8],
            [50, 50, 50, 50],
        ],
        dtype=float,
    )

    converted = pif_module._source_tie_points_to_overlap_pairs(
        source_points,
        str(reference),
        str(sensed),
        ref_vrt,
        sensed_vrt,
        str(valid_mask),
    )

    assert converted == [(4, 1, 4, 1), (6, 3, 6, 3), (8, 5, 8, 5)]


def test_loaded_tie_points_follow_reversed_pair_orientation():
    loaded = {("A", "B"): np.asarray([[1.0, 2.0, 3.0, 4.0]])}

    reversed_points = pif_module._loaded_points_for_pair(loaded, "B", "A")

    assert reversed_points == pytest.approx(np.asarray([[3.0, 4.0, 1.0, 2.0]]))
