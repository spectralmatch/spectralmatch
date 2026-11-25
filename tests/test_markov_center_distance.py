"""Spatial ranking uses original image centers and batched triangle positions."""

import importlib
import math

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, box

from .test_markov_triangles import _inputs, _run, _assert_partition
from .test_dask_execution import _install_fake_dask

module = importlib.import_module("spectralmatch.seamline.markov_triangles")
expressions = importlib.import_module("spectralmatch.seamline.weighted_seamline")


def _face(x, y=0, scale=1 / math.sqrt(3)):
    return Polygon([(x - scale, y - scale), (x + scale, y - scale), (x, y + 2 * scale)])


def _ranking_frame():
    return gpd.GeoDataFrame(
        {"image": ["a", "b"], "quality": [10.0, 10.0], "center_distance": [999, 999]},
        geometry=[box(-30, -10, 30, 10), box(-20, -20, 60, 20)], crs=3857,
    )


@pytest.mark.parametrize("descending", [False, True])
def test_raw_distance_changes_for_each_triangle_and_image(monkeypatch, descending):
    frame = _ranking_frame()
    ranking = module._triangle_rank_context(
        frame, "image", "{var:quality} - 0.01 * {stat:center_distance}", descending
    )
    faces = [_face(2), _face(18)]
    visible = np.ones((2, 2), dtype=bool)
    candidates = [{row.image: row.geometry} for row in frame.itertuples()]
    calls = []
    evaluate = module._evaluate_weighted_expression

    def batch(*args, **kwargs):
        calls.append(kwargs["array_size"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(module, "_evaluate_weighted_expression", batch)
    unary = module._triangle_rank_unaries(
        faces, visible, candidates, np.arange(2), visible, ranking, 1.0
    )
    # Despite different footprint sizes, a 16-unit distance difference costs .16.
    expected = np.array([[0, .16], [.16, 0]])
    np.testing.assert_allclose(unary, expected if descending else expected[:, ::-1], atol=1e-12)
    assert calls == [2, 2]  # One array evaluation per source, not per triangle.
    assert frame.center_distance.tolist() == [999, 999]


@pytest.mark.parametrize("geometry", [
    box(0, 0, 10, 10),
    Polygon(box(0, 0, 10, 10).exterior.coords, [box(3, 3, 7, 7).exterior.coords]),
    Polygon([(0, 0), (10, 0), (10, 2), (2, 2), (2, 10), (0, 10)]),
    MultiPolygon([box(0, 0, 2, 2), box(8, 0, 10, 2)]),
])
def test_footprint_centroid_snaps_to_nearest_valid_coverage(geometry):
    frame = gpd.GeoDataFrame({"image": ["a"]}, geometry=[geometry], crs=3857)
    ranking = module._triangle_rank_context(frame, "image", "-{stat:center_distance}", True)
    center = Point(ranking["images"]["a"]["center"])
    assert geometry.covers(center)
    assert center.distance(geometry.centroid) == pytest.approx(geometry.distance(geometry.centroid))


@pytest.mark.parametrize("expression", [
    "{var:quality} - .01 * {stat:center_distance} + {mean:quality}",
    "sqrt(abs({stat:center_distance} - 2)) + log(max({stat:center_distance}, 1))",
    "min({stat:center_distance}, 3, 5) + round({stat:center_distance} / 3, 2)",
    "{var:quality} if 1 <= {stat:center_distance} < 5 else -{stat:center_distance}",
    "(not ({stat:center_distance} > 3)) + ({stat:center_distance} > 0 and {stat:center_distance} < 5)",
    "({stat:center_distance} and 2) + ({stat:center_distance} or 3)",
    "0 if {stat:center_distance} == 0 else 1 / {stat:center_distance}",
    "{stat:center_distance} if {var:quality} > 0 else 1 / 0",
    "log(max({stat:center_distance}, 1), 2)",
    "{var:center_distance} - {stat:center_distance} - { stat : center_distance}",
])
def test_batched_expressions_match_scalar_semantics(expression):
    distances = np.array([0., 1., 3., 8.])
    code, variables, _, statistics = expressions._parse_rank_expression(expression, vectorized=True)
    scalar, _, _, _ = expressions._parse_rank_expression(expression)
    row = {"quality": 10., "center_distance": 999.}
    values = {symbol: distances for symbol in statistics}
    _, _, summaries, _ = expressions._rank_context(_ranking_frame(), expression)
    actual = expressions._evaluate_weighted_expression(
        row, code, variables, {**summaries, **values}, expression, array_size=len(distances)
    )
    expected = [expressions._evaluate_weighted_expression(
        row, scalar, variables,
        {**summaries, **{symbol: float(distance) for symbol in statistics}}, expression
    ) for distance in distances]
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("expression", [
    "{stat:center_distance}",
    "{stat:triangle_area}",
    "{stat:gap_coverage_fraction} + {stat:center_distance}",
])
def test_center_distance_requires_triangle_context(expression):
    for rank in (expressions._rank_polygons, expressions._rank_gap_polygons):
        with pytest.raises(ValueError, match="requires Markov triangle ranking"):
            result = rank(_ranking_frame(), "image", expression, True)
            if rank is expressions._rank_gap_polygons:
                list(result)


@pytest.mark.parametrize("raster_costs", [False, True])
def test_center_ranking_owns_triangles_nearest_each_image(tmp_path, monkeypatch, raster_costs):
    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    frame["quality"] = 10.
    frame.to_file(source, layer="footprints")
    faces, visible = [], []
    mesh = module._mesh

    def capture(*args):
        result = mesh(*args)
        faces.extend(result[0])
        visible.extend(result[1])
        return result

    monkeypatch.setattr(module, "_mesh", capture)
    output = _run(
        source, str(tmp_path / "center.gpkg"),
        image_rank_function="{var:quality} - 0.01 * {stat:center_distance}",
        mesh_max_triangle_area=2.,
        **({} if raster_costs else {"edge_variables_weights_gsds_bands": []}),
    )
    result = _assert_partition(source, output).set_index("image")
    centers = [geometry.centroid for geometry in frame.geometry]
    compared = 0
    for face, eligible in zip(faces, visible):
        if not all(eligible):
            continue
        distances = [face.centroid.distance(center) for center in centers]
        if abs(distances[0] - distances[1]) < 1e-10:
            continue
        owner = frame.image.iloc[np.argmin(distances)]
        assert result.loc[owner].geometry.covers(face.centroid)
        compared += 1
    assert compared > 20
    assert result.quality_score.isna().all()
    assert "center_distance" not in gpd.read_file(source)


def test_zero_weight_skips_triangle_distance_work(tmp_path, monkeypatch):
    source = _inputs(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("Triangle distances must not be evaluated at zero weight")

    monkeypatch.setattr(module, "_triangle_rank_unaries", unexpected)
    outputs = []
    for index, options in enumerate(({}, {"image_rank_function": "-{stat:center_distance}"})):
        output = _run(source, str(tmp_path / f"zero{index}.gpkg"),
                      image_quality_weight=0, edge_variables_weights_gsds_bands=[], **options)
        outputs.append(_assert_partition(source, output).set_index("image"))
    for name in outputs[0].index:
        assert outputs[0].loc[name].geometry.equals(outputs[1].loc[name].geometry)


def test_expression_coefficient_balances_quality_against_distance():
    frame = _ranking_frame()
    frame["quality"] = [10., 10.2]
    ranking = module._triangle_rank_context(
        frame, "image", "{var:quality} - .01 * {stat:center_distance}", True
    )
    visible = np.ones((2, 2), dtype=bool)
    unary = module._triangle_rank_unaries(
        [_face(2), _face(18)], visible, [{row.image: row.geometry} for row in frame.itertuples()],
        np.arange(2), visible, ranking, 1.,
    )
    # B's .2 extra quality outweighs its .16 distance disadvantage at the first face.
    np.testing.assert_allclose(unary, [[.04, 0], [.36, 0]], atol=1e-12)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("contracted", [False, True])
def test_triangle_area_weighting_is_explicit_and_applied_once(weighted, contracted):
    frame = _ranking_frame()
    expression = "{var:quality} - .01 * {stat:center_distance}"
    if weighted:
        expression = "{stat:triangle_area} * (" + expression + ")"
    ranking = module._triangle_rank_context(frame, "image", expression, True)
    faces = [_face(2), _face(4, scale=2 / math.sqrt(3))]  # Areas 1 and 4.
    visible = np.ones((2, 2), dtype=bool)
    components = np.zeros(2, dtype=int) if contracted else np.arange(2)
    allowed = np.ones((1 if contracted else 2, 2), dtype=bool)
    unary = module._triangle_rank_unaries(
        faces, visible, [{row.image: row.geometry} for row in frame.itertuples()],
        components, allowed, ranking, 2.,
    )
    expected = np.array([[0., .32], [0., .96 if weighted else .24]])
    if contracted:
        expected = expected.sum(axis=0, keepdims=True)
    np.testing.assert_allclose(unary, expected, atol=1e-12)


def test_triangle_area_works_without_center_distance(monkeypatch):
    frame = _ranking_frame()
    frame["quality"] = [1., 2.]
    frame["triangle_area"] = 999.

    def unexpected(*args, **kwargs):
        pytest.fail("Area-only expressions do not need centroid calculations")

    monkeypatch.setattr(module.shapely, "centroid", unexpected)
    ranking = module._triangle_rank_context(
        frame, "image", "{stat:triangle_area} * ({var:quality} + 0 * {var:triangle_area})", True
    )
    visible = np.ones((2, 2), dtype=bool)
    unary = module._triangle_rank_unaries(
        [_face(2), _face(4, scale=2 / math.sqrt(3))], visible,
        [{row.image: row.geometry} for row in frame.itertuples()],
        np.arange(2), visible, ranking, 1.,
    )
    np.testing.assert_allclose(unary, [[1, 0], [4, 0]])
    assert frame.triangle_area.tolist() == [999., 999.]


def test_child_mosaics_keep_original_centers_and_frozen_summaries():
    frame = gpd.GeoDataFrame(
        {"image": ["a", "b", "c"], "quality": [10., 10., 10.]},
        geometry=[box(-30, -10, 30, 10), box(-10, -10, 50, 10), box(-20, -10, 40, 10)], crs=3857,
    )
    ranking = module._triangle_rank_context(
        frame, "image", "{var:quality} - {mean:quality} - .01 * {stat:center_distance}", True
    )
    candidates = [{"a": box(0, -5, 10, 5), "b": box(10, -5, 20, 5)},
                  {"c": box(0, -5, 20, 5)}]
    visible = np.ones((2, 2), dtype=bool)
    unary = module._triangle_rank_unaries(
        [_face(2), _face(18)], visible, candidates, np.arange(2), visible, ranking, 1.
    )
    np.testing.assert_allclose(unary, [[0, .06], [0, .06]], atol=1e-12)


def test_child_source_lookup_handles_only_coordinate_roundoff():
    candidate = {"a": box(0, -5, 10, 5), "b": box(15, -5, 20, 5)}
    names, sources = module._triangle_sources(candidate, np.array([[10 + 1e-14, 0], [18, 0]]))
    assert [names[index] for index in sources] == ["a", "b"]
    with pytest.raises(ValueError, match="Cannot identify a source image"):
        module._triangle_sources(candidate, np.array([[12., 0.]]))


@pytest.mark.parametrize("area_weighted", [False, True])
def test_spatial_ranking_matches_across_serial_process_and_dask(tmp_path, monkeypatch, area_weighted):
    source = _inputs(tmp_path, offsets=(0, 10, 20, 30))
    expression = "-{stat:center_distance}"
    if area_weighted:
        expression = "{stat:triangle_area} * (" + expression + ")"
    calls = {}
    _install_fake_dask(monkeypatch, calls)
    outputs = []
    for mode in ({}, {"image_threads": 2}, {
        "concurrent_processing_backend": "dask",
        "dask_scheduler": ("address", "tcp://scheduler:8786"),
    }):
        output = _run(
            source, str(tmp_path / f"hierarchy{len(outputs)}.gpkg"),
            image_rank_function=expression, mesh_max_triangle_area=8.,
            quadtree_max_images_per_leaf=2, edge_variables_weights_gsds_bands=[], **mode,
        )
        outputs.append(_assert_partition(source, output).set_index("image"))
    for result in outputs[1:]:
        assert set(result.index) == set(outputs[0].index)
        for name in result.index:
            assert result.loc[name].geometry.symmetric_difference(outputs[0].loc[name].geometry).area < 1e-7
    assert calls["closed"]


@pytest.mark.parametrize("statistic", ["area_crs", "area"])
def test_ambiguous_area_statistic_names_are_rejected(statistic):
    with pytest.raises(ValueError, match="Unknown geometry statistic"):
        expressions._parse_rank_expression("{stat:" + statistic + "}")


@pytest.mark.parametrize("expression", [
    "{var:quality} / {stat:center_distance}",
    "sqrt(-{stat:center_distance})",
])
def test_nonfinite_triangle_expressions_raise_clear_errors(expression):
    frame = _ranking_frame()
    ranking = module._triangle_rank_context(frame, "image", expression, True)
    visible = np.ones((1, 2), dtype=bool)
    with pytest.raises(ValueError, match="Cannot rank image.*finite numeric"):
        module._triangle_rank_unaries(
            [_face(0)], visible, [{row.image: row.geometry} for row in frame.itertuples()],
            np.arange(1), visible, ranking, 1.,
        )
