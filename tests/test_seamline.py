import os
import pytest
import geopandas as gpd
from shapely.geometry import box

from spectralmatch import Seamline, create_footprints, postprocess_footprints
from .test_utils import create_dummy_raster


# voronoi_center_seamline
def test_voronoi_center_seamline_all_params(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create dummy input rasters
    input_paths = []
    for name in ["A", "B"]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(
            path,
            width=256,
            height=256,
            count=1,
            transform=(10 if name == "A" else 20, 1, 0, -10, 0, -1),
            fill_value=100 if name == "A" else 150,
        )
        input_paths.append(str(path))

    output_mask = str(output_dir / "seamlines.gpkg")
    debug_vectors = str(output_dir / "debug_vectors.gpkg")

    footprints = create_footprints(
        input_paths, str(output_dir / "footprints.gpkg"), image_field_name="source"
    )
    Seamline.voronoi(
        input_polygons=footprints,
        output_mask=output_mask,
        image_field_name="source",
        debug_logs=True,
        debug_vectors_path=debug_vectors,
    )

    assert os.path.exists(output_mask)
    assert os.path.exists(debug_vectors)


def test_weighted_seamline_ranked_overlay(tmp_path):
    polygons_path = tmp_path / "footprints.gpkg"
    output_path = tmp_path / "weighted_seamlines.gpkg"

    gdf = gpd.GeoDataFrame(
        [
            {
                "image": "A",
                "quality": 5.0,
                "cloud": 20.0,
                "geometry": box(0, 0, 10, 10),
            },
            {
                "image": "B",
                "quality": 10.0,
                "cloud": 5.0,
                "geometry": box(5, 0, 15, 10),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf.to_file(polygons_path, layer="footprints", driver="GPKG")

    result = Seamline.weighted(
        input_polygons=str(polygons_path),
        output_mask=str(output_path),
        input_layer="footprints",
        rank_function="{quality} - {cloud}",
        image_field_name="image",
        debug_logs=True,
    )

    assert result == str(output_path)
    assert os.path.exists(output_path)

    output_gdf = gpd.read_file(output_path, layer="seamlines")
    assert set(output_gdf["image"]) == {"A", "B"}
    ranks = dict(zip(output_gdf["image"], output_gdf["weighted_rank"]))
    assert ranks["B"] < ranks["A"]
    areas = dict(zip(output_gdf["image"], output_gdf.geometry.area))
    assert areas["B"] == pytest.approx(100.0)
    assert areas["A"] == pytest.approx(50.0)


@pytest.mark.parametrize("image_threads", [None, 2])
def test_footprints_mask_coordinates_and_islands(tmp_path, image_threads):
    import numpy as np
    from osgeo import gdal, osr
    from shapely.geometry import Point

    path = str(tmp_path / "masked.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 12, 10, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((100, 2, 0, 200, 0, -2))
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(32604)
    ds.SetProjection(crs.ExportToWkt())
    pixels = np.zeros((10, 12), dtype=np.uint8)
    pixels[1:8, 1:8] = 1
    pixels[3:5, 3:5] = 0
    pixels[1:3, 10:12] = 1
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.GetRasterBand(1).WriteArray(pixels)
    ds = None
    output = create_footprints(
        [path], str(tmp_path / "footprints.gpkg"), image_threads=image_threads
    )
    frame = gpd.read_file(output)
    geometry = frame.geometry.iloc[0]
    assert geometry.area == pytest.approx(np.count_nonzero(pixels) * 4)
    assert geometry.bounds == (102, 184, 124, 198)
    assert len(geometry.geoms) == 2
    assert not geometry.covers(Point(107, 193))
    assert frame.image.iloc[0] == "masked"
    frame["quality"] = 1
    frame.to_file(output, layer="footprints", driver="GPKG")
    weighted = Seamline.weighted(
        output, str(tmp_path / "weighted.gpkg"), rank_function="{quality}"
    )
    voronoi = Seamline.voronoi(output, str(tmp_path / "voronoi.gpkg"))
    for result in (weighted, voronoi):
        assert gpd.read_file(result).geometry.iloc[0].equals(geometry)


@pytest.mark.parametrize("method", ["corridor", "buffer"])
def test_postprocess_opens_edge_holes_preserves_center_and_attributes(tmp_path, method):
    from shapely.geometry import Polygon, Point

    original = Polygon(
        box(0, 0, 100, 100).exterior.coords,
        [box(2, 40, 12, 50).exterior.coords, box(45, 45, 55, 55).exterior.coords],
    )
    source = str(tmp_path / "input.gpkg")
    gpd.GeoDataFrame(
        {"image": ["A"], "quality": [7]}, geometry=[original], crs=32604
    ).to_file(source)
    result = postprocess_footprints(
        source,
        str(tmp_path / "processed.gpkg"),
        edge_distance=3,
        hole_to_hole_distance=0,
        cut_width=4,
        cut_method=method,
        smoothing_radius=0.5,
        simplify_tolerance=0.5,
    )
    frame = gpd.read_file(result)
    geometry = frame.geometry.iloc[0]
    assert geometry.is_valid
    assert original.covers(geometry)
    assert len(geometry.interiors) == 1
    assert not geometry.covers(Point(50, 50))
    assert frame.quality.iloc[0] == 7
    assert frame.image.iloc[0] == "A"


def test_inner_shortcuts_preserve_vertices_holes_and_containment():
    from shapely.geometry import Polygon
    from spectralmatch.seamline.footprints import _simplify_inner

    original = Polygon(
        [(0, 0), (5, 0), (10, 0), (10, 10), (6, 10), (5, 11), (4, 10), (0, 10)],
        [box(2, 2, 3, 3).exterior.coords],
    )
    result = _simplify_inner(original, 1.1, 0.1)
    assert original.covers(result)
    assert result.is_valid
    assert len(result.exterior.coords) < len(original.exterior.coords)
    assert set(result.exterior.coords).issubset(set(original.exterior.coords))
    assert len(result.interiors) == 1


def test_postprocess_uses_original_exterior_for_hole_selection():
    from shapely.geometry import Polygon
    from spectralmatch.seamline.footprints import _postprocess_polygon

    original = Polygon(
        box(0, 0, 100, 100).exterior.coords,
        [box(2, 40, 12, 50).exterior.coords, box(14, 40, 24, 50).exterior.coords],
    )
    result = _postprocess_polygon(
        original, 3, None, 2, "corridor", 0, 0, 0.5, hole_to_hole_distance=0
    )
    assert len(result.interiors) == 1
    assert original.covers(result)


def test_postprocess_rejects_geographic_crs(tmp_path):
    path = str(tmp_path / "input.gpkg")
    gpd.GeoDataFrame({"image": ["A"]}, geometry=[box(0, 0, 1, 1)], crs=4326).to_file(
        path
    )
    with pytest.raises(ValueError, match="projected CRS"):
        postprocess_footprints(path, str(tmp_path / "out.gpkg"))


def test_footprints_rotated_transform_and_empty_mask(tmp_path):
    from osgeo import gdal
    from shapely.geometry import Polygon

    path = str(tmp_path / "rotated.tif")
    create_dummy_raster(
        path, 8, 8, count=1, transform=(100, 2, 1, 200, 1, -2), fill_value=1
    )
    output = create_footprints([path], str(tmp_path / "out.gpkg"))
    expected = Polygon([(100, 200), (116, 208), (124, 192), (108, 184)])
    assert gpd.read_file(output).geometry.iloc[0].equals(expected)
    ds = gdal.Open(path, gdal.GA_Update)
    ds.GetRasterBand(1).Fill(0)
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds = None
    with pytest.raises(ValueError, match="No valid pixels"):
        create_footprints([path], str(tmp_path / "empty.gpkg"))


def test_footprints_dask_and_postprocessing_process_pool(tmp_path, monkeypatch):
    from .test_dask_execution import _install_fake_dask

    calls = {}
    _install_fake_dask(monkeypatch, calls)
    path = str(tmp_path / "A.tif")
    create_dummy_raster(path, 8, 8, count=1, fill_value=1)
    output = create_footprints(
        [path],
        str(tmp_path / "footprints.gpkg"),
        concurrent_processing_backend="dask",
        dask_scheduler=("address", "tcp://scheduler:8786"),
    )
    assert calls["closed"]
    frame = gpd.read_file(output).set_crs(32604, allow_override=True)
    frame.to_file(output, layer="footprints", driver="GPKG")
    result = postprocess_footprints(
        output,
        str(tmp_path / "processed.gpkg"),
        image_threads=2,
        edge_distance=0,
        smoothing_radius=0,
        simplify_tolerance=0,
    )
    assert gpd.read_file(result).geometry.iloc[0].equals(frame.geometry.iloc[0])


def test_postprocess_streams_completed_features_and_marks_failure(
    tmp_path, monkeypatch
):
    import importlib
    import fiona

    module = importlib.import_module("spectralmatch.seamline.footprints")
    source = str(tmp_path / "source.gpkg")
    output = str(tmp_path / "output.gpkg")
    frame = gpd.GeoDataFrame(
        {"image": ["A", "B"], "quality": [None, 4.5]},
        geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
        crs=32604,
    )
    frame.to_file(source, layer="footprints", driver="GPKG")
    actual = module._postprocess_polygon
    calls = []

    def fail_after_first(*args):
        if calls:
            with fiona.open(output, layer="footprints") as saved:
                assert len(saved) == 1
                assert next(iter(saved))["properties"]["image"] == "A"
            raise RuntimeError("worker failed")
        calls.append(True)
        return actual(*args)

    monkeypatch.setattr(module, "_postprocess_polygon", fail_after_first)
    with pytest.raises(RuntimeError, match="worker failed"):
        postprocess_footprints(
            source, output, edge_distance=0, smoothing_radius=0, simplify_tolerance=0
        )
    assert os.path.exists(output + ".incomplete")
    assert len(gpd.read_file(output)) == 1
    monkeypatch.setattr(module, "_postprocess_polygon", actual)
    postprocess_footprints(
        source,
        output,
        resume_from_outputs="yes",
        edge_distance=0,
        smoothing_radius=0,
        simplify_tolerance=0,
    )
    result = gpd.read_file(output)
    assert len(result) == 2
    assert result.quality.iloc[1] == 4.5
    assert not os.path.exists(output + ".incomplete")


def test_task_callback_receives_completion_order_before_logging(monkeypatch):
    from concurrent.futures import Future
    import spectralmatch.utils_multiprocessing as tasks

    events = []

    class Executor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, func, *args):
            future = Future()
            future.set_result(func(*args))
            return future

    monkeypatch.setattr(
        tasks, "as_completed", lambda futures: iter(reversed(list(futures)))
    )
    monkeypatch.setattr(
        tasks,
        "_print_image_completed",
        lambda source, *args, **kwargs: events.append(("log", source)),
    )
    result = tasks._run_image_tasks(
        lambda value: value * 2,
        [(2,), (3,)],
        input_paths=["A", "B"],
        output_paths=["out", "out"],
        parallel=True,
        executor_factory=lambda *args, **kwargs: Executor(),
        collect_results=False,
        result_callback=lambda index, geometry: events.append(
            ("write", index, geometry)
        ),
    )
    assert result == []
    assert events == [("write", 1, 6), ("log", "B"), ("write", 0, 4), ("log", "A")]


def test_inner_simplification_preserves_holes_for_both_ring_orientations():
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    from spectralmatch.seamline.footprints import _simplify_inner

    original = Polygon(
        [
            (0, 0),
            (5, 0),
            (10, 0),
            (10, 10),
            (7, 10),
            (6, 8),
            (5, 11),
            (4, 8),
            (3, 10),
            (0, 10),
        ],
        [box(2, 2, 4, 4).exterior.coords],
    )
    for sign in [-1, 1]:
        result = _simplify_inner(orient(original, sign), 2, 0.1)
        assert result.is_valid and original.covers(result)
        assert len(result.interiors) == 1
        assert len(result.exterior.coords) < len(original.exterior.coords)


@pytest.mark.parametrize(
    "threshold,rank,areas",
    [
        (None, 1, [9]),
        (None, 2, [4, 9]),
        (None, -1, [1]),
        (None, -2, [1, 4]),
        (None, 0, [1, 4, 9]),
        (None, None, [1, 4, 9]),
        (4.0, None, [4, 9]),
        (4.0, -1, [4]),
        (20.0, 1, []),
        (None, 10, [1, 4, 9]),
    ],
)
def test_area_component_filter(threshold, rank, areas):
    from shapely.geometry import MultiPolygon
    from spectralmatch.seamline.footprints import _filter_polygon_area, _polygon_parts

    source = MultiPolygon([box(0, 0, 1, 1), box(10, 0, 12, 2), box(20, 0, 23, 3)])
    result = _filter_polygon_area(source, threshold, rank)
    assert sorted(part.area for part in _polygon_parts(result)) == areas
    assert source.covers(result) or result.is_empty


@pytest.mark.parametrize(
    "params",
    [
        {"area_rank": 1.5},
        {"area_rank": True},
        {"area_filter": -1.0},
        {"area_filter": float("nan")},
        {"area_filter": float("inf")},
        {"area_filter": True},
    ],
)
def test_area_filter_parameter_validation(tmp_path, params):
    with pytest.raises(ValueError, match="area_"):
        postprocess_footprints("unused.gpkg", str(tmp_path / "out.gpkg"), **params)


def test_area_rank_applies_per_feature_after_smoothing(tmp_path):
    from shapely.geometry import Polygon, MultiPolygon
    from spectralmatch.seamline.footprints import _postprocess_polygon

    # The long, narrow component is initially larger but disappears under erosion.
    narrow = box(0, 0, 100, 1)
    compact = Polygon(
        box(200, 0, 208, 8).exterior.coords, [box(203, 3, 205, 5).exterior.coords]
    )
    source = MultiPolygon([narrow, compact])
    result = _postprocess_polygon(source, 0, None, 1, "corridor", 0.6, 0, 0.5, None, 1)
    assert not result.is_empty and compact.covers(result)
    assert len(result.interiors) == 1
    path = str(tmp_path / "source.gpkg")
    out = str(tmp_path / "output.gpkg")
    gpd.GeoDataFrame(
        {"image": ["A", "B"], "quality": [5, 8]},
        geometry=[
            MultiPolygon([box(0, 0, 1, 1), box(10, 0, 12, 2)]),
            MultiPolygon([box(20, 0, 23, 3), box(30, 0, 34, 4)]),
        ],
        crs=32604,
    ).to_file(path)
    postprocess_footprints(
        path, out, edge_distance=0, smoothing_radius=0, simplify_tolerance=0
    )
    frame = gpd.read_file(out)
    assert list(frame.geometry.area) == [4, 16]
    assert list(frame.quality) == [5, 8]


def test_preferred_postprocess_defaults():
    import inspect

    expected = dict(
        edge_distance=800,
        hole_to_hole_distance=800,
        cut_width="maximum_inscribed_circle",
        smoothing_radius=240,
        simplify_tolerance=120,
        area_filter=None,
        area_rank=1,
    )
    signature = inspect.signature(postprocess_footprints)
    assert {name: signature.parameters[name].default for name in expected} == expected


def test_hole_diameter_matches_farthest_vertices():
    import math
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.affinity import rotate
    from spectralmatch.seamline.footprints import _hole_diameter

    rectangle = rotate(box(0, 0, 6, 8), 37)
    assert _hole_diameter(rectangle) == pytest.approx(10)
    rng = np.random.default_rng(83)
    for _ in range(30):
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        radii = rng.uniform(2, 10, 40)
        coords = list(zip(radii * np.cos(angles), radii * np.sin(angles)))
        polygon = Polygon(coords)
        expected = max(math.dist(a, b) for a in coords for b in coords)
        assert _hole_diameter(polygon) == pytest.approx(expected)


@pytest.mark.parametrize("method", ["corridor", "buffer"])
def test_hole_size_width_is_computed_per_selected_hole(tmp_path, method):
    from shapely.geometry import Polygon, LineString
    from shapely.ops import nearest_points, unary_union

    original = Polygon(
        box(0, 0, 200, 200).exterior.coords,
        [
            box(2, 20, 8, 28).exterior.coords,
            box(2, 100, 5, 104).exterior.coords,
            box(95, 95, 105, 105).exterior.coords,
        ],
    )
    cuts = []
    for ring, width in zip(list(original.interiors)[:2], [10, 5]):
        hole = Polygon(ring)
        if method == "corridor":
            cuts.append(
                LineString(nearest_points(hole, original.exterior)).buffer(width / 2)
            )
        else:
            cuts.append(hole.buffer(hole.distance(original.exterior) + width / 2))
    expected = original.difference(unary_union(cuts))
    source = str(tmp_path / "holes.gpkg")
    output = str(tmp_path / "result.gpkg")
    gpd.GeoDataFrame({"image": ["A"]}, geometry=[original], crs=32604).to_file(source)
    postprocess_footprints(
        source,
        output,
        edge_distance=3,
        hole_to_hole_distance=0,
        cut_width="hole_size",
        cut_method=method,
        smoothing_radius=0,
        simplify_tolerance=0,
        area_rank=0,
    )
    result = gpd.read_file(output).geometry.iloc[0]
    assert result.equals(expected)
    assert original.covers(result)
    assert len(result.interiors) == 1


@pytest.mark.parametrize(
    "width", [0, -1, 1.0, 1.5, True, None, "diameter", float("nan"), float("inf")]
)
def test_cut_width_rejects_invalid_types_and_values(tmp_path, width):
    with pytest.raises(ValueError, match="cut_width"):
        postprocess_footprints(
            "unused.gpkg", str(tmp_path / "out.gpkg"), cut_width=width
        )


@pytest.mark.parametrize("angle", [0, 17, 45, 89, 135])
def test_inscribed_width_measures_thickness_at_any_angle(angle):
    from shapely.affinity import rotate
    from spectralmatch.seamline.footprints import _hole_cut_width

    hole = rotate(box(0, 0, 1000, 200), angle)
    assert _hole_cut_width(hole, "maximum_inscribed_circle") == pytest.approx(
        200, abs=1
    )


def test_inscribed_width_uses_largest_lobe_of_concave_hole():
    from shapely.ops import unary_union
    from spectralmatch.seamline.footprints import _hole_cut_width

    hole = unary_union([box(0, 0, 6, 6), box(6, 2, 30, 4), box(30, 1, 34, 5)])
    assert _hole_cut_width(hole, "maximum_inscribed_circle") == pytest.approx(
        6, abs=0.03
    )


@pytest.mark.parametrize("method", ["corridor", "buffer"])
@pytest.mark.parametrize("distance,expected_holes", [(0, 3), (9.99, 3), (10, 2)])
def test_hole_pair_distance_is_independent_of_edge_distance(
    method, distance, expected_holes
):
    from shapely.geometry import Polygon, Point
    from spectralmatch.seamline.footprints import _postprocess_polygon

    original = Polygon(
        box(0, 0, 200, 200).exterior.coords,
        [
            box(50, 50, 60, 60).exterior.coords,
            box(70, 50, 80, 60).exterior.coords,
            box(130, 130, 140, 140).exterior.coords,
        ],
    )
    result = _postprocess_polygon(
        original,
        0,
        0,
        "maximum_inscribed_circle",
        method,
        0,
        0,
        0.5,
        hole_to_hole_distance=distance,
    )
    assert result.is_valid and original.covers(result)
    assert result.exterior.equals(original.exterior)
    assert len(result.interiors) == expected_holes
    for point in [Point(55, 55), Point(75, 55), Point(135, 135)]:
        assert not result.covers(point)


@pytest.mark.parametrize("angle", [0, 31, 90])
@pytest.mark.parametrize(
    "width,expected", [(4, 4), ("hole_size", 10), ("maximum_inscribed_circle", 6)]
)
def test_hole_pair_corridor_uses_smaller_width_and_ignores_ring_order(
    angle, width, expected
):
    from shapely.affinity import rotate
    from shapely.geometry import Polygon, LineString
    from spectralmatch.seamline.footprints import _postprocess_polygon

    outer = box(0, 0, 200, 200)
    holes = [box(50, 50, 70, 70), box(90, 50, 98, 56)]
    section = rotate(LineString([(80, 0), (80, 200)]), angle, origin=(0, 0))
    results = []
    for rings in (holes, holes[::-1]):
        original = rotate(
            Polygon(outer.exterior.coords, [h.exterior.coords for h in rings]),
            angle,
            origin=(0, 0),
        )
        result = _postprocess_polygon(
            original,
            0,
            None,
            width,
            "corridor",
            0,
            0,
            0.5,
            hole_to_hole_distance=21,
        )
        assert result.is_valid
        # Rotated overlay intersections can differ at floating-point precision,
        # making an exact covers predicate unreliable even with no added area.
        assert result.difference(original).area == pytest.approx(0, abs=1e-8)
        assert len(result.interiors) == 1
        removed = original.difference(result)
        assert removed.intersection(section).length == pytest.approx(expected, abs=0.03)
        results.append(result)
    assert results[0].symmetric_difference(results[1]).area < 1e-8


@pytest.mark.parametrize("image_threads", [None, 2])
def test_default_inscribed_width_applies_to_edge_and_hole_pair_cuts(
    tmp_path, image_threads
):
    from shapely.geometry import Polygon, LineString

    original = Polygon(
        box(0, 0, 200, 200).exterior.coords,
        [
            box(4, 20, 64, 30).exterior.coords,
            box(50, 100, 70, 120).exterior.coords,
            box(90, 100, 98, 106).exterior.coords,
        ],
    )
    source = str(tmp_path / "source.gpkg")
    output = str(tmp_path / "result.gpkg")
    gpd.GeoDataFrame(
        {"image": ["A"], "quality": [7]}, geometry=[original], crs=32604
    ).to_file(source, layer="footprints")
    postprocess_footprints(
        source,
        output,
        input_layer="footprints",
        edge_distance=5,
        hole_to_hole_distance=20,
        smoothing_radius=0,
        simplify_tolerance=0,
        image_threads=image_threads,
    )
    frame = gpd.read_file(output, layer="footprints")
    result = frame.geometry.iloc[0]
    assert result.is_valid and original.covers(result)
    assert len(result.interiors) == 1
    removed = original.difference(result)
    assert removed.intersection(LineString([(2, 0), (2, 200)])).length == pytest.approx(
        10
    )
    assert removed.intersection(
        LineString([(80, 0), (80, 200)])
    ).length == pytest.approx(6)
    assert frame.crs.to_epsg() == 32604
    assert frame.image.iloc[0] == "A" and frame.quality.iloc[0] == 7


def test_hole_pairs_use_original_distances_without_changing_edge_eligibility():
    from shapely.geometry import Polygon, Point
    from spectralmatch.seamline.footprints import _postprocess_polygon

    original = Polygon(
        box(0, 0, 200, 200).exterior.coords,
        [box(x, 40, x + 10, 50).exterior.coords for x in (2, 14, 27)],
    )
    result = _postprocess_polygon(
        original,
        3,
        None,
        2,
        "corridor",
        0,
        0,
        0.5,
        hole_to_hole_distance=2,
    )
    assert result.is_valid and original.covers(result)
    # First pair opens to the exterior; the third hole is beyond the pair limit
    # and cannot become edge-eligible through the first two cuts.
    assert len(result.interiors) == 1
    assert Polygon(result.interiors[0]).covers(Point(32, 45))


def test_default_hole_pair_distance_connects_only_pairs_within_800(tmp_path):
    from shapely.geometry import Polygon

    geometries = [
        Polygon(
            box(0, 0, 20000, 20000).exterior.coords,
            [
                box(5000, 5000, 5100, 5100).exterior.coords,
                box(5100 + gap, 5000, 5200 + gap, 5100).exterior.coords,
            ],
        )
        for gap in (799, 801)
    ]
    source = str(tmp_path / "source.gpkg")
    output = str(tmp_path / "result.gpkg")
    gpd.GeoDataFrame(
        {"image": ["near", "far"]}, geometry=geometries, crs=32604
    ).to_file(source)
    postprocess_footprints(source, output, smoothing_radius=0, simplify_tolerance=0)
    actual = gpd.read_file(output).set_index("image")
    assert len(actual.loc["near"].geometry.interiors) == 1
    assert len(actual.loc["far"].geometry.interiors) == 2
    assert actual.geometry.is_valid.all()


def test_hole_pairs_are_limited_to_original_polygon_components():
    from shapely.geometry import Polygon, MultiPolygon
    from spectralmatch.seamline.footprints import _postprocess_polygon

    parts = [
        Polygon(
            box(x, 0, x + 50, 50).exterior.coords,
            [box(x + 10, 10, x + 20, 20).exterior.coords],
        )
        for x in (0, 60)
    ]
    original = MultiPolygon(parts)
    result = _postprocess_polygon(
        original,
        0,
        None,
        "maximum_inscribed_circle",
        "corridor",
        0,
        0,
        0.5,
        area_rank=0,
        hole_to_hole_distance=1400,
    )
    assert result.equals(original)


@pytest.mark.parametrize("distance", [-1, float("nan"), float("inf"), True, None, "10"])
def test_hole_pair_distance_validation(tmp_path, distance):
    with pytest.raises(ValueError, match="hole_to_hole_distance"):
        postprocess_footprints(
            "unused.gpkg", str(tmp_path / "out.gpkg"), hole_to_hole_distance=distance
        )
