import importlib
import json

import numpy as np
import pytest
from osgeo import gdal

from spectralmatch import joint_coregistration
from .utils_test import create_dummy_raster


coregistration_module = importlib.import_module(
    "spectralmatch.joint_coregistration.joint_coregistration"
)


def _write_tie_points(path, records):
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"tie_points": records}, file)


def _pair_record(name_i, name_j, pixel_i=(10.0, 10.0), pixel_j=(10.0, 10.0)):
    return {
        "image_1": name_i,
        "image_2": name_j,
        "points": [[list(pixel_i), list(pixel_j)]],
    }


def _translation_fixture(tmp_path):
    image_a = tmp_path / "a.tif"
    image_b = tmp_path / "b.tif"
    create_dummy_raster(image_a, width=24, height=24, count=1, crs="EPSG:3857", transform=(0, 1, 0, 24, 0, -1))
    create_dummy_raster(image_b, width=24, height=24, count=1, crs="EPSG:3857", transform=(2, 1, 0, 24, 0, -1))
    ties = tmp_path / "ties.json"
    _write_tie_points(ties, [_pair_record("a", "b")])
    return [str(image_a), str(image_b)], str(ties)


def test_joint_coregistration_global_weights_move_less_trusted_image_less(tmp_path):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / "out_a.tif"), str(tmp_path / "out_b.tif")]

    result = joint_coregistration(
        inputs,
        outputs,
        global_model="translation",
        global_image_position_preservation_weights={"a": 100.0, "b": 1.0},
        local_model="none",
        robust_loss="none",
        load_adjustments=ties,
    )

    assert result == outputs
    transforms = [gdal.Open(path).GetGeoTransform() for path in outputs]
    movement_a = transforms[0][0] - 0
    movement_b = transforms[1][0] - 2
    assert abs(movement_a) < abs(movement_b)
    assert (10.5 + movement_a) == pytest.approx(12.5 + movement_b, abs=1e-5)


def test_zero_global_strength_preserves_original_geotransforms(tmp_path):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / "zero_a.tif"), str(tmp_path / "zero_b.tif")]
    joint_coregistration(
        inputs,
        outputs,
        global_tie_point_alignment_strength=0.0,
        local_model="none",
        load_adjustments=ties,
    )
    assert gdal.Open(outputs[0]).GetGeoTransform() == pytest.approx((0, 1, 0, 24, 0, -1))
    assert gdal.Open(outputs[1]).GetGeoTransform() == pytest.approx((2, 1, 0, 24, 0, -1))


def test_joint_coregistration_applies_shared_resolution_and_tap(tmp_path):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / "grid_a.tif"), str(tmp_path / "grid_b.tif")]
    joint_coregistration(
        inputs,
        outputs,
        local_model="none",
        load_adjustments=ties,
        resolution=2.0,
        tap=True,
    )
    for path in outputs:
        transform = gdal.Open(path).GetGeoTransform()
        assert abs(transform[1]) == pytest.approx(2.0)
        assert abs(transform[5]) == pytest.approx(2.0)
        assert transform[0] / 2 == pytest.approx(round(transform[0] / 2))
        assert transform[3] / 2 == pytest.approx(round(transform[3] / 2))


def test_joint_coregistration_rejects_integer_resolution(tmp_path):
    inputs, _ = _translation_fixture(tmp_path)
    with pytest.raises(ValueError, match="positive float"):
        joint_coregistration(
            inputs,
            [str(tmp_path / "bad_a.tif"), str(tmp_path / "bad_b.tif")],
            resolution=2,
        )


def test_tie_point_save_file_contains_only_raw_pixel_points(tmp_path):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / "save_a.tif"), str(tmp_path / "save_b.tif")]
    saved = tmp_path / "saved.json"
    joint_coregistration(
        inputs,
        outputs,
        global_model="translation",
        local_model="none",
        load_adjustments=ties,
        save_adjustments=str(saved),
    )
    model = json.loads(saved.read_text(encoding="utf-8"))
    assert set(model) == {"tie_points"}
    assert model["tie_points"] == [_pair_record("a", "b")]
    assert model["tie_points"][0]["points"][0] == [
        [10.0, 10.0],
        [10.0, 10.0],
    ]


def test_tie_point_compact_json_round_trip(tmp_path):
    path = tmp_path / "compact_tie_points.json"
    expected = np.asarray(
        [
            [1.25, 2.5, 3.75, 4.0],
            [5.0, 6.25, 7.5, 8.75],
        ]
    )
    coregistration_module._save_tie_points(path, {("a", "b"): expected})

    model = json.loads(path.read_text(encoding="utf-8"))
    assert model == {
        "tie_points": [
            {
                "image_1": "a",
                "image_2": "b",
                "points": [
                    [[1.25, 2.5], [3.75, 4.0]],
                    [[5.0, 6.25], [7.5, 8.75]],
                ],
            }
        ]
    }
    loaded = coregistration_module._load_tie_points(str(path))
    assert set(loaded) == {("a", "b")}
    assert loaded[("a", "b")] == pytest.approx(expected)


def test_partial_tie_point_load_calculates_only_missing_pairs(tmp_path, monkeypatch):
    paths = []
    for index, x_origin in enumerate((0, 2, 4)):
        path = tmp_path / f"image_{index}.tif"
        create_dummy_raster(path, width=16, height=16, count=1, crs="EPSG:3857", transform=(x_origin, 1, 0, 16, 0, -1))
        paths.append(str(path))
    infos = {
        f"image_{index}": coregistration_module._read_image_info(f"image_{index}", path)
        for index, path in enumerate(paths)
    }
    loaded = {("image_0", "image_1"): np.asarray([[5.0, 5.0, 5.0, 5.0]])}
    calculated = []

    def fake_extract(info_i, info_j, *args):
        pair = coregistration_module._canonical_pair(info_i.name, info_j.name)
        calculated.append(pair)
        points = np.asarray([[5.0, 5.0, 5.0, 5.0]])
        return pair, points, points, 3.0

    monkeypatch.setattr(coregistration_module, "_extract_pair_tie_points", fake_extract)
    filtered, raw, _ = coregistration_module._collect_tie_points(
        (("image_0", "image_1"), ("image_0", "image_2"), ("image_1", "image_2")),
        infos,
        loaded,
        "orb",
        None,
        3.0,
        None,
        False,
    )
    assert ("image_0", "image_1") not in calculated
    assert set(calculated) == {("image_0", "image_2"), ("image_1", "image_2")}
    assert set(filtered) == set(raw) == {
        ("image_0", "image_1"),
        ("image_0", "image_2"),
        ("image_1", "image_2"),
    }


def test_three_image_global_network_is_solved_jointly(tmp_path):
    infos = {}
    for name, origin in (("a", 0), ("b", 2), ("c", 4)):
        path = tmp_path / f"{name}.tif"
        create_dummy_raster(path, width=16, height=16, count=1, crs="EPSG:3857", transform=(origin, 1, 0, 16, 0, -1))
        infos[name] = coregistration_module._read_image_info(name, str(path))
    ties = {
        ("a", "b"): np.asarray([[5.0, 5.0, 5.0, 5.0]]),
        ("b", "c"): np.asarray([[5.0, 5.0, 5.0, 5.0]]),
    }
    parameters = coregistration_module._solve_global_alignment(
        infos,
        ties,
        "translation",
        {"a": 1.0, "b": 1.0, "c": 1.0},
        1.0,
        "none",
        3.0,
        False,
    )
    corrected_x = []
    for name in ("a", "b", "c"):
        point = coregistration_module._pixels_to_map(
            infos[name].transform, np.asarray([[5.0, 5.0]])
        )
        corrected_x.append(
            coregistration_module._evaluate_global(
                infos[name], parameters[name], "translation", point
            )[0, 0]
        )
    assert corrected_x == pytest.approx([corrected_x[0]] * 3, abs=1e-6)


def test_local_weights_move_less_trusted_image_less(tmp_path):
    inputs, _ = _translation_fixture(tmp_path)
    infos = {
        name: coregistration_module._read_image_info(name, path)
        for name, path in zip(("a", "b"), inputs)
    }
    ties = {("a", "b"): np.asarray([[10.0, 10.0, 10.0, 10.0]])}
    meshes = coregistration_module._solve_local_alignment(
        infos,
        ties,
        "none",
        {"a": np.zeros(0), "b": np.zeros(0)},
        "bilinear",
        {"a": 100.0, "b": 1.0},
        1.0,
        8.0,
        1.0,
        1.0,
        6.0,
        "none",
        3.0,
        False,
    )
    displacements = {}
    for name in ("a", "b"):
        point = coregistration_module._pixels_to_map(
            infos[name].transform, np.asarray([[10.0, 10.0]])
        )
        displacements[name] = coregistration_module._evaluate_mesh(
            meshes[name], point, "bilinear"
        )[0, 0]
    assert abs(displacements["a"]) < abs(displacements["b"])


def test_local_alignment_writes_readable_geolocation_warps(tmp_path):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / "local_a.tif"), str(tmp_path / "local_b.tif")]
    joint_coregistration(
        inputs,
        outputs,
        global_model="none",
        local_model="bilinear",
        local_grid_spacing=8.0,
        local_anchor_falloff_distance=6.0,
        robust_loss="none",
        load_adjustments=ties,
    )
    for path in outputs:
        dataset = gdal.Open(path)
        assert dataset is not None
        assert dataset.RasterXSize > 0 and dataset.RasterYSize > 0
        dataset = None


def test_local_geolocation_grid_does_not_shift_by_half_sample_step(tmp_path):
    source_path = tmp_path / "local_grid_source.tif"
    output_path = tmp_path / "local_grid_output.tif"
    create_dummy_raster(
        source_path,
        width=24,
        height=24,
        count=1,
        crs="EPSG:3857",
        transform=(0, 1, 0, 24, 0, -1),
        fill_value=0,
    )
    source = gdal.Open(str(source_path), gdal.GA_Update)
    values = np.zeros((24, 24), dtype=np.float32)
    values[12, 12] = 100
    source.GetRasterBand(1).WriteArray(values)
    source = None

    info = coregistration_module._read_image_info("source", str(source_path))
    mesh = coregistration_module._build_mesh(info, 8.0)
    mesh.displacement[:, :, 0] = 1e-6
    coregistration_module._write_local_warp_output(
        info,
        str(output_path),
        "none",
        np.zeros(0),
        "bilinear",
        mesh,
        "nearest",
        "Float32",
        0,
        "GTiff",
        [],
        False,
        None,
        str(tmp_path),
    )

    output = gdal.Open(str(output_path))
    corrected = output.GetRasterBand(1).ReadAsArray()
    output = None
    assert np.unravel_index(np.argmax(corrected), corrected.shape) == (12, 12)


def test_coregister_overlap_helper_keeps_contract(tmp_path, monkeypatch):
    reference = tmp_path / "reference.tif"
    sensed = tmp_path / "sensed.tif"
    output = tmp_path / "corrected.tif"
    create_dummy_raster(reference, width=24, height=24, count=1, crs="EPSG:3857")
    create_dummy_raster(sensed, width=24, height=24, count=1, crs="EPSG:3857")
    pairs = [(4, 4, 4, 4), (4, 18, 4, 18), (18, 4, 18, 4), (18, 18, 18, 18)]
    monkeypatch.setattr(coregistration_module, "_extract_conjugate_point_pairs", lambda *args: pairs)
    result_path, result_pairs = coregistration_module._coregister_overlap(
        str(reference), str(sensed), str(output)
    )
    assert result_path == str(output)
    assert result_pairs == pairs
    assert gdal.Open(result_path) is not None


def test_coregister_overlap_reuses_supplied_tie_points(tmp_path, monkeypatch):
    reference = tmp_path / "loaded_reference.tif"
    sensed = tmp_path / "loaded_sensed.tif"
    output = tmp_path / "loaded_corrected.tif"
    create_dummy_raster(reference, width=24, height=24, count=1, crs="EPSG:3857")
    create_dummy_raster(sensed, width=24, height=24, count=1, crs="EPSG:3857")
    pairs = [(4, 4, 4, 4), (4, 18, 4, 18), (18, 4, 18, 4), (18, 18, 18, 18)]

    def fail_extraction(*args):
        raise AssertionError("ORB extraction must not run for supplied tie points")

    monkeypatch.setattr(coregistration_module, "_extract_conjugate_point_pairs", fail_extraction)
    result_path, result_pairs = coregistration_module._coregister_overlap(
        str(reference),
        str(sensed),
        str(output),
        tie_point_pairs=pairs,
    )

    assert result_path == str(output)
    assert result_pairs == pytest.approx(pairs)
    assert gdal.Open(result_path) is not None


def test_unusable_supplied_tie_points_raise_without_feature_matching(tmp_path, monkeypatch):
    reference = tmp_path / "invalid_reference.tif"
    sensed = tmp_path / "invalid_sensed.tif"
    output = tmp_path / "invalid_corrected.tif"
    create_dummy_raster(reference, width=24, height=24, count=1, crs="EPSG:3857")
    create_dummy_raster(sensed, width=24, height=24, count=1, crs="EPSG:3857")
    def fail_extraction(*args):
        raise AssertionError("ORB extraction must not run for supplied tie points")

    monkeypatch.setattr(coregistration_module, "_extract_conjugate_point_pairs", fail_extraction)

    with pytest.raises(ValueError, match="At least 3 conjugate point pairs"):
        coregistration_module._coregister_overlap(
            str(reference),
            str(sensed),
            str(output),
            tie_point_pairs=[(8, 8, 8, 8)] * 3,
        )


def test_joint_coregistration_validates_weights_and_duplicate_basenames(tmp_path):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / "invalid_a.tif"), str(tmp_path / "invalid_b.tif")]
    with pytest.raises(ValueError, match="not found"):
        joint_coregistration(
            inputs,
            outputs,
            global_image_position_preservation_weights={"missing": 1.0},
            load_adjustments=ties,
        )

    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    duplicate = duplicate_dir / "a.tif"
    create_dummy_raster(duplicate, crs="EPSG:3857")
    with pytest.raises(ValueError, match="basenames must be unique"):
        joint_coregistration(
            [inputs[0], str(duplicate)],
            outputs,
            global_model="none",
            local_model="none",
        )
