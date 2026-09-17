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


def _write_ties(path, records):
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
    _write_ties(ties, [_pair_record("a", "b")])
    return [str(image_a), str(image_b)], str(ties)


def test_joint_coregistration_custom_overview_scales(tmp_path):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / "out_a.tif"), str(tmp_path / "out_b.tif")]
    joint_coregistration(inputs, outputs, local_model="none", tie_load_path=ties, build_overviews=True, window_scales=(2, 4))
    for path in outputs:
        dataset = gdal.Open(path)
        band = dataset.GetRasterBand(1)
        assert band.GetOverviewCount() == 2
        assert [band.GetOverview(i).XSize for i in range(2)] == [12, 6]


def test_joint_coregistration_global_weights_move_less_trusted_image_less(tmp_path):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / "out_a.tif"), str(tmp_path / "out_b.tif")]

    result = joint_coregistration(
        inputs,
        outputs,
        global_model="translation",
        global_image_movement_penalty_weights={"a*": 100.0, "*": 1.0},
        local_model="none",
        tie_robust_loss="none",
        tie_load_path=ties,
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
        global_tie_alignment_strength=0.0,
        local_model="none",
        tie_load_path=ties,
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
        tie_load_path=ties,
        resolution=2.0,
        tap=True,
    )
    for path in outputs:
        transform = gdal.Open(path).GetGeoTransform()
        assert abs(transform[1]) == pytest.approx(2.0)
        assert abs(transform[5]) == pytest.approx(2.0)
        assert transform[0] / 2 == pytest.approx(round(transform[0] / 2))
        assert transform[3] / 2 == pytest.approx(round(transform[3] / 2))


@pytest.mark.parametrize("resolution", [True, 0, float("nan")])
def test_joint_coregistration_rejects_invalid_numeric_resolution(tmp_path, resolution):
    inputs, _ = _translation_fixture(tmp_path)
    with pytest.raises(ValueError, match="positive int or float"):
        joint_coregistration(
            inputs,
            [str(tmp_path / "bad_a.tif"), str(tmp_path / "bad_b.tif")],
            resolution=resolution,
        )


def test_tie_compact_json_round_trip(tmp_path):
    path = tmp_path / "compact_ties.json"
    expected = np.asarray(
        [
            [1.25, 2.5, 3.75, 4.0],
            [5.0, 6.25, 7.5, 8.75],
        ]
    )
    coregistration_module._save_ties(path, {("a", "b"): expected})

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
    loaded = coregistration_module._load_ties(str(path))
    assert set(loaded) == {("a", "b")}
    assert loaded[("a", "b")] == pytest.approx(expected)


def test_partial_tie_load_calculates_only_missing_pairs(tmp_path, monkeypatch):
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
        return pair, points, 3.0

    monkeypatch.setattr(coregistration_module, "_extract_pair_ties", fake_extract)
    filtered, _ = coregistration_module._collect_ties(
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
    assert set(filtered) == {
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
        tie_robust_loss="none",
        tie_load_path=ties,
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


def test_coregister_overlap_reuses_supplied_ties(tmp_path, monkeypatch):
    reference = tmp_path / "loaded_reference.tif"
    sensed = tmp_path / "loaded_sensed.tif"
    output = tmp_path / "loaded_corrected.tif"
    create_dummy_raster(reference, width=24, height=24, count=1, crs="EPSG:3857")
    create_dummy_raster(sensed, width=24, height=24, count=1, crs="EPSG:3857")
    pairs = [(4, 4, 4, 4), (4, 18, 4, 18), (18, 4, 18, 4), (18, 18, 18, 18)]

    def fail_extraction(*args):
        raise AssertionError("ORB extraction must not run for supplied tie points")

    monkeypatch.setattr(coregistration_module, "_extract_conjugate_point_pairs", fail_extraction)
    monkeypatch.setattr(coregistration_module, "_filter_point_pairs", fail_extraction)
    result_path, result_pairs = coregistration_module._coregister_overlap(
        str(reference),
        str(sensed),
        str(output),
        tie_pairs=pairs,
    )

    assert result_path == str(output)
    assert result_pairs == pytest.approx(pairs)
    assert gdal.Open(result_path) is not None


def test_unusable_supplied_ties_raise_without_feature_matching(tmp_path, monkeypatch):
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
            tie_pairs=[(8, 8, 8, 8)] * 3,
        )


def test_joint_coregistration_defaults_unmatched_weights_and_rejects_duplicate_basenames(tmp_path):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / "invalid_a.tif"), str(tmp_path / "invalid_b.tif")]
    joint_coregistration(
        inputs,
        outputs,
        local_model="none",
        global_image_movement_penalty_weights={"missing*": 100.0},
        tie_load_path=ties,
    )
    assert [gdal.Open(path).GetGeoTransform()[0] for path in outputs] == pytest.approx([1, 1])

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


@pytest.mark.parametrize("radius", [
    128, "100", "10m", "0px", "1e999crs",
])
def test_joint_coregistration_rejects_invalid_search_radius(tmp_path, radius):
    inputs, _ = _translation_fixture(tmp_path)
    with pytest.raises(ValueError, match="tie_search_radius must be a positive finite number followed by"):
        joint_coregistration(inputs, str(tmp_path / "output"), tie_search_radius=radius)


@pytest.mark.parametrize("radius,expected", [
    ("12.5crs", 12.5), ("12.5px", 25),
])
def test_search_radius_unit_conversion(radius, expected):
    from types import SimpleNamespace

    fine = SimpleNamespace(pixel_size=0.5)
    coarse = SimpleNamespace(pixel_size=2.0)
    assert coregistration_module._resolved_search_radius(fine, coarse, radius) == expected
    assert coregistration_module._resolved_search_radius(coarse, fine, radius) == expected


def _textured_pair(tmp_path, *, sensed_resolution=2):
    texture = np.random.default_rng(7).integers(1, 256, size=(768, 768), dtype=np.uint8)
    infos = []
    for name, origin, resolution in (("a", 100, 2), ("b", 108, sensed_resolution)):
        data = texture if resolution == 2 else np.repeat(np.repeat(texture, 2, axis=0), 2, axis=1)
        path = tmp_path / f"{name}.tif"
        create_dummy_raster(
            path, crs="EPSG:3857", transform=(origin, resolution, 0, 2000, 0, -resolution),
            band_data=data,
        )
        infos.append(coregistration_module._read_image_info(name, str(path)))
    return infos


@pytest.mark.parametrize("sensed_resolution,search_radius,radius", [(1, "128px", 256), (2, "160crs", 160)])
def test_grid_orb_reads_only_windows_and_recovers_original_pixels(tmp_path, monkeypatch, sensed_resolution, search_radius, radius):
    info_a, info_b = _textured_pair(tmp_path, sensed_resolution=sensed_resolution)
    # The coarser raster has 2-unit pixels, so automatic 256-pixel windows have
    # a 256-unit radius even when the sensed raster has 1-unit pixels.
    reads = []
    original_read = gdal.Band.ReadAsArray

    def record_read(band, *args, **kwargs):
        # Every feature AND mask read must specify a bounded window.
        assert len(args) == 4
        xoff, yoff, width, height = args
        assert 0 <= xoff < band.XSize and 0 <= yoff < band.YSize
        assert 0 < width <= radius + 1 and 0 < height <= radius + 1
        assert xoff + width <= band.XSize and yoff + height <= band.YSize
        reads.append(args)
        return original_read(band, *args, **kwargs)

    monkeypatch.setattr(gdal.Band, "ReadAsArray", record_read)
    # Reverse order also exercises canonical pair coordinates.
    pair, points, _ = coregistration_module._extract_pair_ties(
        info_b, info_a, "orb", 20, 4, True, grid_spacing=512, search_radius=search_radius,
    )
    assert pair == ("a", "b")
    assert 6 <= len(points) <= 9
    assert len(reads) == 9 * 4  # Two masks and two data bands per window.
    xs, ys = coregistration_module._tie_grid(info_a, info_b, 512)
    inverse = gdal.InvGeoTransform((108, 2, 0, 2000, 0, -2))
    expected_windows = set()
    for y in ys:
        for x in xs:
            left, top = gdal.ApplyGeoTransform(inverse, x - radius, y + radius)
            right, bottom = gdal.ApplyGeoTransform(inverse, x + radius, y - radius)
            xoff, yoff = max(0, int(np.floor(left))), max(0, int(np.floor(top)))
            xend, yend = min(764, int(np.ceil(right))), min(768, int(np.ceil(bottom)))
            expected_windows.add((
                xoff, yoff, xend - xoff, yend - yoff,
            ))
    assert set(reads) == expected_windows
    # Corresponding original pixels agree after accounting for native resolution.
    expected_sensed_pixels = (points[:, :2] + 0.5) * (2 / sensed_resolution) - 0.5
    # Bilinear resampling and ORB pyramid levels can shift a detected corner by
    # a few overlap pixels; compare at the shared (coarser) matching resolution.
    np.testing.assert_allclose(points[:, 2:], expected_sensed_pixels, atol=3 * (2 / sensed_resolution))
    coordinates = coregistration_module._pixels_to_map(info_a.transform, points[:, :2])
    cells = np.floor((coordinates - (xs[0], ys[0])) / 512 + 0.5).astype(int)
    assert len(np.unique(cells, axis=0)) == len(points)


@pytest.mark.parametrize("keep", [1, 2, None])
def test_window_selection_prefers_own_centers_and_keeps_nearby_matches(tmp_path, keep):
    path = tmp_path / "grid.tif"
    create_dummy_raster(path, width=600, height=600, count=1, crs="EPSG:3857", transform=(0, 1, 0, 600, 0, -1))
    info = coregistration_module._read_image_info("grid", str(path))
    reference = np.asarray([[100, 100], [105, 105], [195, 300], [205, 300], [500, 500], [300, 500]])
    sensed = reference.copy()
    sensed[-1] = (495, 495)  # Keep matches that are close in the sensed image too.
    # Invert the north-up geotransform, including pixel-center convention.
    def pixels(coordinates):
        return np.column_stack((coordinates[:, 0] - 0.5, 599.5 - coordinates[:, 1]))
    points = np.column_stack((pixels(reference), pixels(sensed)))
    # The middle matches originate in windows on the opposite side of the cell boundary.
    centers = np.asarray([[100, 100], [100, 100], [300, 300], [100, 300], [500, 500], [300, 500]])
    selected = coregistration_module._select_window_ties(points, centers, info, keep)
    assert len(selected) == (5 if keep == 1 else 6)
    np.testing.assert_array_equal(selected[0], points[0])
    # Only the farther candidate from the same window is removed. Close matches
    # from separate windows or in the sensed image are retained.
    expected = points[[0, 2, 3, 4, 5]] if keep == 1 else points
    assert set(map(tuple, selected)) == set(map(tuple, expected))


def test_overlapping_windows_keep_matches_closer_to_another_center(tmp_path, monkeypatch):
    infos = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.tif"
        create_dummy_raster(path, width=400, height=200, count=1, crs="EPSG:3857", transform=(0, 1, 0, 200, 0, -1), fill_value=100)
        infos.append(coregistration_module._read_image_info(name, str(path)))
    # Centers are x=100 and x=300. Both matches are closer to x=300, but each
    # originating window must retain its own match. The second read starts at x=80.
    window_matches = iter((np.asarray([[240, 100, 240, 100.]], dtype=float), np.asarray([[190, 100, 190, 100.]], dtype=float)))
    monkeypatch.setattr(coregistration_module, "_match_orb_windows", lambda *args: next(window_matches))
    _, points, _ = coregistration_module._extract_pair_ties(
        *infos, "orb", None, None, False, grid_spacing=200, search_radius="220px",
    )
    np.testing.assert_array_equal(np.sort(points[:, 0]), [240, 270])


def test_window_selection_merges_only_identical_selected_pairs(tmp_path):
    path = tmp_path / "grid.tif"
    create_dummy_raster(path, width=400, height=200, count=1, crs="EPSG:3857", transform=(0, 1, 0, 200, 0, -1))
    info = coregistration_module._read_image_info("grid", str(path))
    points = np.asarray([[200, 100, 200, 100], [200, 100, 200, 100], [201, 100, 201, 100]], dtype=float)
    centers = np.asarray([[100, 100], [300, 100], [300, 100]])
    for keep in (2, None):
        selected = coregistration_module._select_window_ties(points, centers, info, keep)
        assert set(map(tuple, selected)) == {tuple(points[0]), tuple(points[2])}


def test_pair_inlier_mask_preserves_duplicate_window_membership(tmp_path, monkeypatch):
    info_a, info_b = _textured_pair(tmp_path)
    points = np.asarray([[10, 10, 10, 10], [30, 10, 30, 10], [10, 30, 10, 30], [10, 10, 10, 10], [50, 50, 50, 50], [-1, 1, 1, 1], [100, 100, 400, 400]], dtype=float)

    def affine_inliers(source, target, **kwargs):
        # The duplicate gets one RANSAC vote; invalid/out-of-range shifts are excluded.
        assert len(source) == len(target) == 4
        inliers = np.ones((4, 1), dtype=np.uint8)
        inliers[np.argmax(source[:, 0])] = 0
        return np.zeros((2, 3)), inliers

    import cv2
    monkeypatch.setattr(cv2, "estimateAffine2D", affine_inliers)
    keep = coregistration_module._filter_map_ties(points, info_a, info_b, 20, 4)
    np.testing.assert_array_equal(keep, [True, True, True, True, False, False, False])


def test_small_overlap_clips_grid_windows_and_skips_nodata(tmp_path, monkeypatch):
    paths = [tmp_path / f"{name}.tif" for name in ("a", "b")]
    for path in paths:
        create_dummy_raster(
            path, width=100, height=80, count=1, crs="EPSG:3857",
            transform=(0, 1, 0, 80, 0, -1), fill_value=0,
        )
    infos = [coregistration_module._read_image_info(name, str(path)) for name, path in zip(("a", "b"), paths)]
    xs, ys = coregistration_module._tie_grid(*infos, 500)
    np.testing.assert_array_equal(xs, [50])
    np.testing.assert_array_equal(ys, [40])
    reads = []
    original_read = gdal.Band.ReadAsArray

    def record_read(band, *args, **kwargs):
        reads.append(args)
        return original_read(band, *args, **kwargs)

    def fail_gray(*args):
        raise AssertionError("Do not read image bands for an empty validity mask")

    monkeypatch.setattr(gdal.Band, "ReadAsArray", record_read)
    monkeypatch.setattr(coregistration_module, "_read_uint8_gray", fail_gray)
    _, points, _ = coregistration_module._extract_pair_ties(
        *infos, "orb", None, None, False, grid_spacing=500, search_radius="100crs",
    )
    assert reads == [(0, 0, 100, 80)] * 2
    assert points.shape == (0, 4)


def test_grid_windows_skip_untextured_cells_without_losing_other_matches(tmp_path):
    info_a, info_b = _textured_pair(tmp_path)
    for info in (info_a, info_b):
        dataset = gdal.Open(info.path, gdal.GA_Update)
        dataset.GetRasterBand(1).WriteArray(np.full((256, 256), 100, dtype=np.uint8), 256, 256)
        dataset = None
    _, points, _ = coregistration_module._extract_pair_ties(
        info_a, info_b, "orb", 20, 4, False, grid_spacing=512, search_radius="160crs",
    )
    assert 6 <= len(points) <= 8


def test_window_points_convert_exactly_to_rotated_native_pixels(tmp_path):
    from dataclasses import replace

    info_a, info_b = _textured_pair(tmp_path, sensed_resolution=1)
    info_b = replace(info_b, transform=(108, 1, 0.2, 2000, 0.1, -1))
    overlap_transform = (108, 2, 0, 2000, 0, -2)
    window_points = np.asarray([[120.25, 240.75, 124.25, 240.75]])
    converted = coregistration_module._overlap_to_original_pixels(window_points, overlap_transform, info_a, info_b)
    for info, columns in ((info_a, slice(0, 2)), (info_b, slice(2, 4))):
        expected = coregistration_module._pixels_to_map(overlap_transform, window_points[:, columns])
        actual = coregistration_module._pixels_to_map(info.transform, converted[:, columns])
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_joint_coregistration_passes_grid_options_to_parallel_matching(tmp_path, monkeypatch):
    search_radius, features, keep = "80px", 250, 3
    info_a, info_b = _textured_pair(tmp_path)
    original_extract = coregistration_module._extract_pair_ties
    grids = []
    selected = {}

    def record_extract(*args):
        grids.append(args[6:])
        result = original_extract(*args)
        selected[result[0]] = result[1].copy()
        return result

    monkeypatch.setattr(coregistration_module, "_extract_pair_ties", record_extract)
    outputs = [str(tmp_path / f"corrected_{name}.tif") for name in ("a", "b")]
    saved = tmp_path / "selected.json"
    joint_coregistration(
        [info_a.path, info_b.path], outputs, local_model="none", tie_grid_spacing=512, local_grid_spacing=64,
        tie_maximum_displacement=20,
        tie_ransac_reprojection_threshold=4, tie_save_path=str(saved), image_threads=2,
        tie_orb_max_features=features, tie_max_matches_per_window=keep,
        tie_search_radius=search_radius,
    )
    assert grids == [(512, search_radius, features, keep)]
    loaded = coregistration_module._load_ties(str(saved))[("a", "b")]
    assert len(loaded) >= 6
    assert len(loaded) <= 9 * keep
    np.testing.assert_array_equal(loaded, selected[("a", "b")])
    origins = [gdal.Open(path).GetGeoTransform()[0] for path in outputs]
    assert abs(origins[0] - origins[1]) < 1


@pytest.mark.parametrize("option,value", [
    ("tie_orb_max_features", 0),
    ("tie_max_matches_per_window", 1.5),
    ("tie_orb_max_features", True),
])
def test_orb_feature_counts_require_positive_integers(tmp_path, option, value):
    inputs, _ = _translation_fixture(tmp_path)
    with pytest.raises(ValueError, match=f"{option} must be a positive integer"):
        joint_coregistration(inputs, str(tmp_path / "out"), **{option: value})


@pytest.mark.parametrize("features,keep", [(1, 10), (500, 50)])
def test_orb_supports_small_and_large_feature_counts(tmp_path, monkeypatch, features, keep):
    import cv2

    info_a, info_b = _textured_pair(tmp_path)
    original_create = cv2.ORB_create
    requested = []

    def record_create(**kwargs):
        requested.append(kwargs["nfeatures"])
        return original_create(**kwargs)

    monkeypatch.setattr(cv2, "ORB_create", record_create)
    _, points, _ = coregistration_module._extract_pair_ties(
        info_a, info_b, "orb", 20, 4, False, grid_spacing=512,
        tie_orb_max_features=features, tie_max_matches_per_window=keep,
    )
    assert requested == [features]
    assert points.ndim == 2 and points.shape[1] == 4
    if len(points) and keep is not None:
        xs, ys = coregistration_module._tie_grid(info_a, info_b, 512)
        coordinates = coregistration_module._pixels_to_map(info_a.transform, points[:, :2])
        cells = np.floor((coordinates - (xs[0], ys[0])) / 512 + 0.5).astype(int)
        _, counts = np.unique(cells, axis=0, return_counts=True)
        assert (counts <= keep).all()


def test_missing_tie_cache_is_saved_then_reused_without_processing(tmp_path, monkeypatch):
    info_a, info_b = _textured_pair(tmp_path)
    cache = tmp_path / "new_cache" / "selected.json"
    outputs = [str(tmp_path / f"output_{name}.tif") for name in ("a", "b")]
    options = dict(
        input_images=[info_a.path, info_b.path], output_images=outputs, local_model="none",
        tie_grid_spacing=512, tie_load_path=str(cache), tie_save_path=str(cache),
    )
    with pytest.warns(RuntimeWarning, match="file not found"):
        joint_coregistration(**options)
    selected = coregistration_module._load_ties(str(cache))[("a", "b")]
    assert 6 <= len(selected) <= 9
    first_transforms = [gdal.Open(path).GetGeoTransform() for path in outputs]

    def fail_processing(*args, **kwargs):
        raise AssertionError("Selected cache must bypass feature processing")

    for helper in ("_extract_pair_ties", "_filter_map_ties", "_select_window_ties"):
        monkeypatch.setattr(coregistration_module, helper, fail_processing)
    # Detection settings do not change an already selected cache.
    joint_coregistration(**options, tie_search_radius="1crs", tie_orb_max_features=1, tie_max_matches_per_window=2)
    np.testing.assert_array_equal(coregistration_module._load_ties(str(cache))[("a", "b")], selected)
    assert [gdal.Open(path).GetGeoTransform() for path in outputs] == first_transforms


def test_empty_selected_pair_is_reused(tmp_path, monkeypatch):
    inputs, _ = _translation_fixture(tmp_path)
    infos = {name: coregistration_module._read_image_info(name, path) for name, path in zip(("a", "b"), inputs)}
    empty = np.empty((0, 4))

    def fail_extract(*args):
        raise AssertionError("A cached empty pair should not be recalculated")

    monkeypatch.setattr(coregistration_module, "_extract_pair_ties", fail_extract)
    points, _ = coregistration_module._collect_ties(
        [("a", "b")], infos, {("a", "b"): empty}, "orb", None, None, None, False,
    )
    assert points[("a", "b")].shape == (0, 4)


def test_malformed_tie_cache_still_raises(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        coregistration_module._load_ties(str(path))


def test_ordered_image_rules_use_first_match_and_default_to_one():
    names = ['base', 'base_02', 'scene_a', 'scene_b', 'scene_3', 'Scene_a', '.hidden', 'other']
    rules = {'base': 0, 'base*': 0.25, 'scene_{a,b}': 0.5, '@(scene_3|absent)': 0.75, 'missing*': 0}
    assert coregistration_module._resolve_image_values(rules, names) == dict(zip(names, [0, 0.25, 0.5, 0.5, 0.75, 1, 1, 1]))
    assert coregistration_module._resolve_image_values({'*': 0, 'base': 1}, names) == dict.fromkeys(names, 0)
    assert coregistration_module._resolve_image_values({}, names) == dict.fromkeys(names, 1)
    assert coregistration_module._resolve_image_values(None, names) == dict.fromkeys(names, 1)


@pytest.mark.parametrize('pattern,matched', [
    ('base|scene_1', ['base', 'scene_1']),
    ('!base', ['scene_1', 'scene_2', 'scene_10', '.hidden', 'BASE']),
])
def test_image_rules_use_extended_glob_syntax(pattern, matched):
    names = ['base', 'scene_1', 'scene_2', 'scene_10', '.hidden', 'BASE']
    rules = {pattern: 0.25, '*': 0.75}
    assert coregistration_module._resolve_image_values(rules, names) == {name: 0.25 if name in matched else 0.75 for name in names}


@pytest.mark.parametrize('option,value', [
    ('global_tie_alignment_strength', -0.1),
    ('local_tie_alignment_strength', 1.1),
    ('global_tie_alignment_strength', True),
    ('global_tie_alignment_strength', float('nan')),
    ('local_tie_alignment_strength', '[]'),
    ('global_tie_alignment_strength', '{bad'),
    ('local_tie_alignment_strength', '{"*": -1}'),
    ('local_tie_alignment_strength', '{"*": NaN}'),
])
def test_alignment_strength_requires_scalar_or_json_rules(tmp_path, option, value):
    inputs, _ = _translation_fixture(tmp_path)
    with pytest.raises(ValueError, match=option + ' must be a number from 0 to 1 or a JSON object string'):
        joint_coregistration(inputs, str(tmp_path / 'invalid'), **{option: value})


@pytest.mark.parametrize('strength,expected', [(0.5, [0.5, 1.5]), ('{"a": 0, "*": 0.5}', [0, 1.5])])
def test_global_strength_rules_scale_the_joint_solution(tmp_path, strength, expected):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / f'scaled_{name}.tif') for name in ('a', 'b')]
    joint_coregistration(inputs, outputs, local_model='none', global_tie_alignment_strength=strength, tie_load_path=ties, tie_robust_loss='none')
    assert [gdal.Open(path).GetGeoTransform()[0] for path in outputs] == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize('strength', [0.25, {'a': 0, 'b': 0.5}])
def test_local_strengths_scale_each_mesh_after_joint_solving(tmp_path, strength):
    inputs, _ = _translation_fixture(tmp_path)
    infos = {name: coregistration_module._read_image_info(name, path) for name, path in zip(('a', 'b'), inputs)}
    ties = {('a', 'b'): np.asarray([[10, 10, 10, 10]], dtype=float)}
    args = (infos, ties, 'none', {'a': np.zeros(0), 'b': np.zeros(0)}, 'bilinear', {'a': 1, 'b': 1})
    options = (8.0, 1.0, 1.0, 6.0, 'none', 3.0, False)
    full = coregistration_module._solve_local_alignment(*args, 1.0, *options)
    scaled = coregistration_module._solve_local_alignment(*args, strength, *options)
    for name in infos:
        factor = strength[name] if isinstance(strength, dict) else strength
        np.testing.assert_allclose(scaled[name].displacement, full[name].displacement * factor, atol=1e-10)


def test_resolved_image_controls_are_logged_and_forwarded(tmp_path, monkeypatch, capsys):
    inputs, ties = _translation_fixture(tmp_path)
    original_global = coregistration_module._solve_global_alignment
    original_local = coregistration_module._solve_local_alignment

    def record_global(*args):
        assert args[3] == pytest.approx({'a': 20 / 11, 'b': 2 / 11})
        assert args[4] == {'a': 0, 'b': 1}
        return original_global(*args)

    def record_local(*args):
        assert args[5] == pytest.approx({'a': 0.5, 'b': 1.5})
        assert args[6] == {'a': 0.25, 'b': 1}
        # The local solve receives the already scaled global correction for each image.
        np.testing.assert_array_equal(args[3]['a'], [0, 0])
        assert args[3]['b'][0] == pytest.approx(-20 / 11)
        return original_local(*args)

    monkeypatch.setattr(coregistration_module, '_solve_global_alignment', record_global)
    monkeypatch.setattr(coregistration_module, '_solve_local_alignment', record_local)
    joint_coregistration(
        inputs, str(tmp_path / 'logged'), tie_load_path=ties,
        global_tie_alignment_strength='{"a": 0, "*": 1}', local_tie_alignment_strength='{"a": 0.25}',
        global_image_movement_penalty_weights={'a': 10, '*': 1}, local_image_movement_penalty_weights={'b*': 3},
        local_grid_spacing=8, debug_logs=True,
    )
    output = capsys.readouterr().out
    assert 'a: global_strength=0, local_strength=0.25, global_movement_penalty=10' in output
    assert 'b: global_strength=1, local_strength=1, global_movement_penalty=1' in output
    assert 'local_movement_penalty=3 (normalized=1.5)' in output


@pytest.mark.parametrize('extension', ['gpkg', 'geojson'])
def test_tie_vector_export_uses_each_original_rotated_geotransform(tmp_path, extension):
    import fiona

    infos = {}
    for name, transform in [('a', (100, 2, 0.25, 200, 0.1, -2)), ('b', (108, 1, -0.2, 202, 0.1, -1))]:
        path = tmp_path / f'{name}.tif'
        create_dummy_raster(path, width=24, height=24, count=1, crs='EPSG:3857', transform=transform)
        infos[name] = coregistration_module._read_image_info(name, str(path))
    points = np.asarray([[0, 0, 1, 2], [2.25, 3.5, 4.75, 1.25]])
    output = tmp_path / 'vectors' / f'ties.{extension}'
    coregistration_module._save_ties_crs(str(output), {('a', 'b'): points}, infos)
    with fiona.open(output) as source:
        assert source.crs.to_epsg() == 3857
        assert source.schema['geometry'] == 'Point'
        records = {(item['properties']['tie_id'], item['properties']['image']): item for item in source}
    assert len(records) == 4
    expected = {(1, 'a'): (101.125, 199.05), (1, 'b'): (109, 199.65), (2, 'a'): (106.5, 192.275), (2, 'b'): (112.9, 200.775)}
    for key, xy in expected.items():
        feature = records[key]
        assert tuple(feature['geometry']['coordinates']) == pytest.approx(xy)
        properties = feature['properties']
        name = key[1]
        assert properties['match_img'] == ('b' if name == 'a' else 'a')
        expected_pixel = points[key[0] - 1, :2] if name == 'a' else points[key[0] - 1, 2:]
        assert (properties['pixel_col'], properties['pixel_row']) == pytest.approx(expected_pixel)
    if extension == 'gpkg':
        assert fiona.listlayers(output) == ['ties']


def test_tie_vector_export_reuses_selected_points_and_keeps_original_positions(tmp_path, monkeypatch):
    import fiona

    inputs, cache = _translation_fixture(tmp_path)
    output = tmp_path / 'selected.gpkg'
    saved = tmp_path / 'selected.json'

    def fail_detection(*args, **kwargs):
        raise AssertionError('Exporting cached selected ties must not run ORB')

    monkeypatch.setattr(coregistration_module, '_extract_pair_ties', fail_detection)
    joint_coregistration(inputs, str(tmp_path / 'outputs'), global_model='translation', local_model='none', tie_load_path=cache, tie_save_path=str(saved), tie_save_crs_path=str(output))
    with fiona.open(output) as source:
        coordinates = {feature['properties']['image']: tuple(feature['geometry']['coordinates']) for feature in source}
    assert coordinates == {'a': (10.5, 13.5), 'b': (12.5, 13.5)}
    assert json.loads(saved.read_text()) == json.loads(open(cache).read())


def test_tie_vector_export_alone_triggers_matching(tmp_path, monkeypatch):
    import fiona

    inputs, _ = _translation_fixture(tmp_path)
    calls = []

    def select_pair(info_a, info_b, *args):
        calls.append((info_a.name, info_b.name))
        return ('a', 'b'), np.asarray([[1, 2, 3, 4]], dtype=float), 3.0

    monkeypatch.setattr(coregistration_module, '_extract_pair_ties', select_pair)
    output = tmp_path / 'selected.gpkg'
    joint_coregistration(inputs, str(tmp_path / 'outputs'), global_model='none', local_model='none', tie_save_crs_path=str(output))
    assert calls == [('a', 'b')]
    with fiona.open(output) as source:
        assert len(source) == 2


def test_tie_vector_export_writes_empty_layer_with_crs(tmp_path):
    import fiona

    inputs, _ = _translation_fixture(tmp_path)
    infos = {name: coregistration_module._read_image_info(name, path) for name, path in zip(('a', 'b'), inputs)}
    output = tmp_path / 'empty.gpkg'
    coregistration_module._save_ties_crs(str(output), {('a', 'b'): np.empty((0, 4))}, infos)
    with fiona.open(output) as source:
        assert len(list(source)) == 0
        assert source.crs.to_epsg() == 3857


def test_tie_vector_export_replaces_ties_and_preserves_other_geopackage_layers(tmp_path):
    import fiona

    inputs, _ = _translation_fixture(tmp_path)
    infos = {name: coregistration_module._read_image_info(name, path) for name, path in zip(('a', 'b'), inputs)}
    output = tmp_path / 'ties.gpkg'
    with fiona.open(output, 'w', driver='GPKG', layer='notes', crs='EPSG:3857', schema={'geometry': 'Point', 'properties': {'note': 'str'}}) as destination:
        destination.write({'geometry': {'type': 'Point', 'coordinates': (0, 0)}, 'properties': {'note': 'keep'}})
    coregistration_module._save_ties_crs(str(output), {('a', 'b'): np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)}, infos)
    coregistration_module._save_ties_crs(str(output), {('a', 'b'): np.asarray([[1, 2, 3, 4]], dtype=float)}, infos)
    with fiona.open(output, layer='ties') as source:
        assert len(source) == 2
    with fiona.open(output, layer='notes') as source:
        assert len(source) == 1
        assert next(iter(source))['properties']['note'] == 'keep'


def test_tie_vector_export_runs_when_raster_outputs_are_reused(tmp_path, monkeypatch):
    import fiona

    inputs, cache = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / f'output_{name}.tif') for name in ('a', 'b')]
    joint_coregistration(inputs, outputs, global_model='none', local_model='none')

    def fail_solve(*args, **kwargs):
        raise AssertionError('Reused raster outputs do not need solving or warping')

    for helper in ('_solve_global_alignment', '_solve_local_alignment', '_apply_alignment_process_image'):
        monkeypatch.setattr(coregistration_module, helper, fail_solve)
    vector = tmp_path / 'ties.gpkg'
    result = joint_coregistration(inputs, outputs, tie_load_path=cache, tie_save_crs_path=str(vector), resume_from_outputs='yes')
    assert result == outputs
    with fiona.open(vector) as source:
        assert len(source) == 2


@pytest.mark.parametrize('value', [123, ''])
def test_tie_vector_export_validates_path_type(tmp_path, value):
    inputs, _ = _translation_fixture(tmp_path)
    with pytest.raises(ValueError, match='tie_save_crs_path must be'):
        joint_coregistration(inputs, str(tmp_path / 'outputs'), tie_save_crs_path=value)


@pytest.mark.parametrize('destination', ['input', 'output', 'cache'])
def test_tie_vector_export_rejects_colliding_paths(tmp_path, destination):
    inputs, cache = _translation_fixture(tmp_path)
    outputs = [str(tmp_path / f'output_{name}.tif') for name in ('a', 'b')]
    path = {'input': inputs[0], 'output': outputs[0], 'cache': cache}[destination]
    with pytest.raises(ValueError, match='must differ from raster and tie-cache paths'):
        joint_coregistration(inputs, outputs, tie_load_path=cache, tie_save_crs_path=path)


def test_tie_vector_export_rejects_unknown_format(tmp_path):
    inputs, _ = _translation_fixture(tmp_path)
    infos = {name: coregistration_module._read_image_info(name, path) for name, path in zip(('a', 'b'), inputs)}
    with pytest.raises(ValueError, match='tie_save_crs_path needs a supported vector extension'):
        coregistration_module._save_ties_crs(str(tmp_path / 'ties.unknown'), {}, infos)
