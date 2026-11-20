"""Gap-aware scores must change after each unbuffered ownership decision."""
import math

import geopandas as gpd
import pytest
import shapely
from shapely.geometry import MultiPolygon, box, mapping

from spectralmatch import create_footprints
from spectralmatch.seamline.footprints import _coverage_polygons
from spectralmatch.seamline.weighted_seamline import _rank_polygons, _gap_intersection_areas

AREA = "5 * log(max({stat:area_crs}, 1) / 1000000) / log(2)"
ANGLES = "({var:view_zenith} + {var:solar_zenith})"


def hole_candidates():
    hole = box(200, 200, 800, 800)
    return gpd.GeoDataFrame(
        {"image": ["a", "b", "c"], "view_zenith": [0., 10., 30.],
         "solar_zenith": [0., 0., 0.]},
        geometry=[box(0, 0, 1000, 1000).difference(hole),
                  box(200, 200, 680, 800), hole], crs=3857,
    )


def coverage(frame, expression, **options):
    settings = dict(outline_path=None, outline_layer=None, image_rank_function=expression,
                    image_rank_descending=True, extrude_distance=0, min_polygon_area=1.)
    settings.update(options)
    return _coverage_polygons(frame, "image", **settings)


@pytest.mark.parametrize("weight,owners", [(0, ["a", "b", "c"]), (20, ["a", "b", "c"]), (200, ["a", "c"])])
def test_gap_weight_can_prefer_one_worse_image_over_two_fillers(weight, owners):
    frame = hole_candidates()
    expression = f"{AREA} + {weight} * {{stat:gap_coverage_fraction}} - {ANGLES}"
    result = coverage(frame, expression)
    assert result.image.tolist() == owners
    assert result.geometry.union_all().equals(box(0, 0, 1000, 1000))
    assert result.geometry.iloc[0].equals(frame.geometry.iloc[0])
    # Bootstrap preserves the existing area-and-angle score, without a gap bonus.
    assert result.weighted_score.iloc[0] == pytest.approx(5 * math.log2(.64))
    if weight == 200:
        assert result.geometry.iloc[1].equals(frame.geometry.iloc[2])
        assert result.weighted_score.iloc[1] == pytest.approx(200 + 5 * math.log2(.36) - 30)
        assert result.weighted_rank.tolist() == [1, 2]
        assert result.geometry.iloc[0].intersection(result.geometry.iloc[1]).area == 0


def test_largest_remaining_gap_is_recomputed_and_summaries_stay_fixed():
    large, small = box(1, 1, 5, 5), box(6, 1, 9, 4)
    frame = gpd.GeoDataFrame(
        {"image": ["a", "b", "c"], "quality": [1000., 0., 20.]},
        geometry=[box(0, 0, 10, 10).difference(large.union(small)), large, small], crs=3857,
    )
    expression = "{var:quality} - {mean:quality} + 200 * {stat:gap_coverage_fraction}"
    result = coverage(frame, expression)
    assert result.image.tolist() == ["a", "b", "c"]
    assert result.weighted_score.tolist() == pytest.approx([660, -140, -120])


def test_area_remains_original_component_area_and_stat_does_not_alias_column():
    frame = hole_candidates()
    frame["gap_coverage_fraction"] = 999.
    # A's second component is covered by C but must not lend its area bonus to A.
    frame.loc[0, "geometry"] = MultiPolygon([frame.geometry.iloc[0], box(250, 250, 260, 260)])
    expression = f"{AREA} + 200 * {{stat: gap_coverage_fraction}} - {ANGLES} + 0 * {{var:gap_coverage_fraction}}"
    result = coverage(frame, expression)
    assert result.image.tolist() == ["a", "c"]
    assert result.gap_coverage_fraction.tolist() == [999., 999.]
    assert result.weighted_score.iloc[1] == pytest.approx(200 + 5 * math.log2(.36) - 30)


@pytest.mark.parametrize("descending", [True, False])
def test_direction_and_ties_are_deterministic(descending):
    frame = hole_candidates().rename(columns={"view_zenith": "quality"})
    frame["quality"] = [1000., 10., 0.]
    expression = "{var:quality} + 200 * {stat:gap_coverage_fraction}"
    if not descending:
        expression = "-(" + expression + ")"
    result = coverage(frame.iloc[::-1], expression, image_rank_descending=descending)
    assert result.image.tolist() == ["a", "c"]
    frame.loc[1, "geometry"] = frame.geometry.iloc[2]
    frame.loc[1, "quality"] = 0.
    tied = coverage(frame.iloc[::-1], expression, image_rank_descending=descending)
    assert tied.image.tolist() == ["a", "b"]


def test_outline_limits_gap_scores_and_buffering_does_not_change_selection(tmp_path):
    frame = gpd.GeoDataFrame(
        {"image": ["a", "b", "c"], "quality": [1000., 0., 30.]},
        geometry=[box(0, 0, 5, 10), box(5, 0, 15, 10), box(5, 0, 10, 8)], crs=3857,
    )
    path = str(tmp_path / "outline.gpkg")
    gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=3857).to_file(path)
    expression = "{var:quality} + 200 * {stat:gap_coverage_fraction}"
    raw = coverage(frame, expression, outline_path=path)
    buffered = coverage(frame, expression, outline_path=path, extrude_distance=10)
    assert raw.image.tolist() == buffered.image.tolist() == ["a", "b"]
    assert raw.weighted_score.tolist() == buffered.weighted_score.tolist() == [1000., 200.]
    assert raw.geometry.iloc[1].equals(box(5, 0, 10, 10))
    assert buffered.geometry.iloc[1].equals(frame.geometry.iloc[1])


def test_gap_fraction_requires_coverage_context():
    with pytest.raises(ValueError, match="requires create_footprints coverage ranking"):
        _rank_polygons(hole_candidates(), "image", "{stat:gap_coverage_fraction}", True)


def test_gap_area_shortcut_matches_exact_intersection_with_holes_and_boundaries():
    gap = box(0, 0, 10, 10).difference(box(3, 3, 5, 5))
    geometries = gpd.GeoSeries([
        box(1, 1, 2, 2), box(9, 0, 11, 2), box(3, 3, 4, 4),
        box(10, 0, 12, 2), box(20, 20, 21, 21), box(-1, -1, 11, 11),
        MultiPolygon([box(1, 1, 2, 2), box(6, 6, 7, 7)]),
    ]).array
    expected = shapely.area(shapely.intersection(geometries, gap))
    assert _gap_intersection_areas(geometries, gap) == pytest.approx(expected)
    assert len(_gap_intersection_areas(geometries[:0], gap)) == 0


def test_reranking_failure_preserves_existing_output(tmp_path):
    frame = hole_candidates()
    output = tmp_path / "coverage.gpkg"
    output.write_bytes(b"previous output")
    records = frame.drop(columns=["image", "geometry"]).to_dict("records")
    for record, geometry in zip(records, frame.geometry):
        record["geometry"] = mapping(geometry)
    with pytest.raises(ValueError, match="Could not evaluate"):
        create_footprints(
            [name + ".tif" for name in frame.image], str(output), footprint_source="metadata",
            metadata_crs="EPSG:3857", metadata_records=records,
            image_rank_function=f"-{ANGLES} + {{stat:gap_coverage_fraction}} / (1 - {{stat:gap_coverage_fraction}})",
        )
    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob(".coverage-*"))
