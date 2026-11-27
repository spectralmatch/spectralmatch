"""Buffered coverage grows from owned areas without acquiring new islands."""
import pytest
from shapely.geometry import box
from spectralmatch.seamline.footprints import _grow_owned_coverage, _polygon_parts
from .test_mosaic_coverage import write_coverage


def test_growth_rejects_large_detached_source_island():
    owned = box(0, 0, 10, 10)
    island = box(12, 0, 22, 10)
    source = owned.union(island)
    assert _grow_owned_coverage(owned, source, 20).equals(owned)


def test_growth_cannot_jump_a_gap_in_a_connected_source():
    left = box(0, 0, 10, 30)
    right = box(15, 0, 25, 30)
    source = left.union(right).union(box(0, 25, 25, 30))
    assert source.geom_type == "Polygon"
    owned = box(0, 0, 10, 10)
    result = _grow_owned_coverage(owned, source, 6)
    assert result.covers(owned)
    assert source.covers(result)
    assert result.area > owned.area
    assert result.intersection(right).is_empty
    # A larger margin can reach the bridge: the other arm is then connected.
    assert _grow_owned_coverage(owned, source, 40).equals(source)


def test_growth_keeps_every_existing_owner_including_small_components():
    owned = box(0, 0, 1, 1).union(box(12, 0, 12.1, 0.1))
    island = box(6, 0, 7, 1)
    assert _grow_owned_coverage(owned, owned.union(island), 6).equals(owned)
    assert _grow_owned_coverage(owned, owned.union(island), 0).equals(owned)


def test_growth_can_join_owned_components_and_preserves_source_holes():
    source = box(0, 0, 20, 10).difference(box(9, 4, 11, 6))
    owned = box(1, 1, 3, 3).union(box(15, 1, 17, 3))
    result = _grow_owned_coverage(owned, source, 8)
    assert len(_polygon_parts(result)) == 1
    assert result.covers(owned)
    assert source.covers(result)
    assert result.intersection(box(9, 4, 11, 6)).area == 0


@pytest.mark.parametrize("postprocess", [False, True])
def test_public_coverage_growth_does_not_resurrect_filtered_components(tmp_path, postprocess):
    main = box(0, 0, 10, 10)
    tiny = box(12, 0, 12.2, 0.2)
    result = write_coverage(tmp_path, {"a": main.union(tiny)}, {"a": 1},
        image_rank_function="{quality}", min_polygon_area=1, extrude_distance=5,
        do_postprocess_footprints=postprocess,
        postprocess_footprints_params={"hole_edge_distance": 0, "hole_to_hole_distance": 0,
            "simplify_smoothing_radius": 0, "simplify_tolerance": 0, "filter_area_rank": None})
    assert result.loc["a"].geometry.equals(main)
