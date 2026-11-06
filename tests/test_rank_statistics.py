"""Image-level ranking placeholders and standardized angle preferences."""

import inspect
import math

import geopandas as gpd
import pytest
from shapely.geometry import box, mapping

from spectralmatch import create_footprints, markov_triangles, weighted_seamline
from spectralmatch.seamline import weighted_seamline as module
from .utils_test import create_dummy_raster


ANGLE_EXPRESSION = (
    "-({var:view_zenith} - {mean:view_zenith}) / max({std:view_zenith}, 1e-12)"
    " - ({var:solar_zenith} - {mean:solar_zenith}) / max({std:solar_zenith}, 1e-12)"
)


@pytest.fixture
def frame():
    return gpd.GeoDataFrame(
        {"image": ["a", "b", "c"], "view_zenith": [10.0, 20.0, 30.0],
         "solar_zenith": [50.0, 10.0, 30.0], "cloud_cover": [0.0, 99.0, 0.0]},
        geometry=[box(0, 0, 10, 10), box(0, 0, 1, 1), box(0, 0, 1, 1)], crs=3857,
    )


def test_equal_standardized_angle_weights_favor_lower_angles(frame):
    ranked = module._rank_polygons(frame, "image", ANGLE_EXPRESSION, True)
    assert list(ranked.image) == ["b", "a", "c"]
    assert ranked.set_index("image").weighted_score.to_dict() == pytest.approx(
        {"a": 0, "b": math.sqrt(1.5), "c": -math.sqrt(1.5)}
    )
    # The preferred image can have more cloud: cloud cover is absent from this score.
    assert ranked.iloc[0].cloud_cover == 99


@pytest.mark.parametrize("statistic,expected", [
    ("mean", 20), ("std", math.sqrt(200 / 3)), ("variance", 200 / 3),
    ("min", 10), ("max", 30), ("median", 20), ("sum", 60), ("count", 3), ("range", 20),
])
def test_basic_summaries_are_image_weighted_population_values(frame, statistic, expected):
    ranked = module._rank_polygons(frame, "image", "{" + statistic + ":view_zenith}", True)
    assert ranked.weighted_score.tolist() == pytest.approx([expected] * 3)


@pytest.mark.parametrize("size", [1, 3])
def test_constant_angles_and_single_image_have_zero_standardized_score(frame, size):
    frame = frame.iloc[:size].copy()
    frame["view_zenith"] = 0.1
    frame["solar_zenith"] = 30.0
    ranked = module._rank_polygons(frame, "image", ANGLE_EXPRESSION, True)
    assert list(ranked.image) == sorted(frame.image)
    assert ranked.weighted_score.tolist() == [0.0] * size


def test_statistics_are_computed_once_per_field_and_statistic(frame, monkeypatch):
    calls = []
    actual = module._SUMMARY_FUNCTIONS["std"]

    def summarize(values):
        calls.append(list(values))
        return actual(values)

    monkeypatch.setitem(module._SUMMARY_FUNCTIONS, "std", summarize)
    module._rank_polygons(frame, "image", "{std:view_zenith} + { std : view_zenith} + {var:view_zenith}", True)
    assert calls == [[10.0, 20.0, 30.0]]


def test_new_and_legacy_fields_share_math_functions_and_support_nonidentifier_names(frame):
    frame["view angle-degrees"] = frame.view_zenith
    ranked = module._rank_polygons(
        frame, "image", "sqrt(abs({var:view angle-degrees} - {view_zenith})) + {mean:view angle-degrees}", True,
    )
    assert ranked.weighted_score.tolist() == [20.0] * 3


@pytest.mark.parametrize("expression,message", [
    ("{var:missing}", "Field 'missing'"),
    ("{mean:missing}", "Field 'missing'"),
    ("{unknown:view_zenith}", "Unknown summary"),
    ("{mean:}", "Invalid ranking placeholder"),
    ("{var:}", "Invalid ranking placeholder"),
    ("{wrong:view_zenith}", "Unknown summary"),
    ("{stat:mean:view_zenith}", "Unknown summary"),
    ("{var:view_zenith", "Invalid rank_function"),
    ("1 + 2", "placeholder"),
    ("{var:view_zenith} / 0", "Could not evaluate"),
    ("{var:view_zenith}.__class__", "arithmetic"),
    ("open({var:view_zenith})", "documented math functions"),
])
def test_invalid_expressions_raise_clear_errors(frame, expression, message):
    with pytest.raises(ValueError, match=message):
        module._rank_polygons(frame, "image", expression, True)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "unknown"])
def test_summary_fields_require_finite_numeric_values_for_every_image(frame, value):
    frame["view_zenith"] = frame.view_zenith.astype(object)
    frame.loc[0, "view_zenith"] = value
    with pytest.raises(ValueError, match="finite numeric"):
        module._rank_polygons(frame, "image", ANGLE_EXPRESSION, True)


def test_coverage_summaries_exclude_images_outside_the_outline(tmp_path):
    outline = str(tmp_path / "outline.gpkg")
    gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=3857).to_file(outline)
    records = [{"view_zenith": view, "solar_zenith": solar,
                "geometry": mapping(box(0, 0, 10, 10))}
               for view, solar in ((10, 50), (20, 10), (30, 30))]
    records.append({"view_zenith": 1000, "solar_zenith": 1000, "geometry": mapping(box(20, 20, 30, 30))})
    output = create_footprints(
        [f"{name}.tif" for name in ("a", "b", "c", "outside")], str(tmp_path / "coverage.gpkg"),
        footprint_source="metadata", metadata_records=records, metadata_crs="EPSG:3857",
        outline_path=outline, image_rank_function=ANGLE_EXPRESSION,
    )
    result = gpd.read_file(output)
    assert list(result.image) == ["b"]
    assert result.weighted_score.iloc[0] == pytest.approx(math.sqrt(1.5))
    assert result.geometry.iloc[0].equals(box(0, 0, 10, 10))


@pytest.mark.parametrize("method", ["weighted", "markov"])
def test_other_seamline_methods_share_the_placeholder_evaluator(tmp_path, method):
    if method == "markov":
        pytest.importorskip("maxflow")
    paths = [str(tmp_path / f"{name}.tif") for name in ("a", "b", "c")]
    for path in paths:
        create_dummy_raster(path, width=4, height=4, count=1, crs="EPSG:3857", transform=(0, 1, 0, 4, 0, -1), fill_value=1)
    raw = create_footprints(
        paths, str(tmp_path / "raw.gpkg"),
        metadata_records=[{"view_zenith": view, "solar_zenith": solar} for view, solar in ((10, 50), (20, 10), (30, 30))],
    )
    if method == "weighted":
        output = weighted_seamline(raw, str(tmp_path / "ranked.gpkg"), rank_function=ANGLE_EXPRESSION)
        score = "weighted_score"
    else:
        output = markov_triangles(paths, str(tmp_path / "ranked.gpkg"), input_polygons=raw,
                                  image_rank_function=ANGLE_EXPRESSION, mesh_spacing=2,
                                  edge_variables_weights_gsds_bands=[])
        score = "quality_score"
    result = gpd.read_file(output)
    assert list(result.image) == ["b"]
    assert result[score].iloc[0] == pytest.approx(math.sqrt(1.5))
    assert result.geometry.iloc[0].area == pytest.approx(16)


def test_unset_image_ranking_keeps_all_raw_outlines(tmp_path):
    assert inspect.signature(create_footprints).parameters["image_rank_function"].default is None
    geometry = box(0, 0, 10, 10)
    output = create_footprints(
        ["a.tif", "b.tif"], str(tmp_path / "raw.gpkg"), footprint_source="metadata",
        metadata_records=[{"view_zenith": angle, "solar_zenith": angle, "geometry": mapping(geometry)} for angle in (10, 30)],
    )
    result = gpd.read_file(output)
    assert set(result.image) == {"a", "b"}
    assert "weighted_score" not in result
    assert all(polygon.equals(geometry) for polygon in result.geometry)
