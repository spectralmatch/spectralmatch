"""Initial ranked coverage shares create_footprints with raw raster footprints."""
from pathlib import Path

import geopandas as gpd
import pytest
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

from spectralmatch import create_footprints
from .test_dask_execution import _install_fake_dask
from .utils_test import create_dummy_raster


def write_coverage(tmp_path, geometries, scores, **kwargs):
    paths = [f"/images/{name}.tif" for name in geometries]
    output = str(tmp_path / "coverage.gpkg")
    create_footprints(
        paths, output,
        metadata_records=[{"quality": scores[name], "scene_id": name, "geometry": mapping(geometries[name])} for name in geometries],
        footprint_source="metadata", metadata_crs="EPSG:3857", epsg=3857, **kwargs,
    )
    return gpd.read_file(output).set_index("image")


@pytest.mark.parametrize("descending,owner", [(True, "a"), (False, "b")])
def test_ranking_direction_and_overlap_ownership(tmp_path, descending, owner):
    source = {"a": box(0, 0, 10, 10), "b": box(5, 0, 15, 10)}
    result = write_coverage(tmp_path, source, {"a": 9, "b": 2},
                            image_rank_function="{quality}", image_rank_descending=descending)
    assert result.loc[owner].geometry.equals(source[owner])
    assert result.geometry.iloc[0].intersection(result.geometry.iloc[1]).area == 0
    assert result.geometry.union_all().equals(box(0, 0, 15, 10))
    assert result.loc["a", "quality"] == 9
    assert result.loc["b", "weighted_score"] == 2


def test_outline_and_extrusion_preserve_overlap_and_source_bounds(tmp_path):
    outline_path = str(tmp_path / "outline.gpkg")
    outline = box(2, 2, 13, 8)
    gpd.GeoDataFrame(geometry=[outline], crs=3857).to_file(outline_path, layer="goal")
    source = {"a": box(0, 0, 10, 10), "b": box(5, 0, 15, 10), "hidden": box(3, 3, 5, 5), "outside": box(30, 30, 40, 40)}
    result = write_coverage(tmp_path, source, {"a": 10, "b": 5, "hidden": 0, "outside": 20},
                            outline_path=outline_path, outline_layer="goal",
                            image_rank_function="{quality}", extrude_distance=2)
    assert set(result.index) == {"a", "b"}
    for name, owned in (("a", box(2, 2, 10, 8)), ("b", box(10, 2, 13, 8))):
        assert result.loc[name].geometry.symmetric_difference(owned.buffer(2).intersection(source[name])).area < 1e-10
        assert source[name].covers(result.loc[name].geometry)
    assert result.loc["b"].geometry.bounds[0] == 8  # Margin under the better image.
    assert result.geometry.union_all().covers(outline)
    assert result.geometry.union_all().difference(outline).area > 0


def test_extrusion_never_changes_ownership_for_subsequent_images(tmp_path):
    source = {"a": box(0, 0, 10, 10), "b": box(5, 0, 20, 10), "c": box(15, 0, 25, 10)}
    result = write_coverage(tmp_path, source, {"a": 3, "b": 2, "c": 1},
                            image_rank_function="{quality}", extrude_distance=2)
    assert result.loc["b"].geometry.bounds == (8, 0, 20, 10)
    assert result.loc["c"].geometry.bounds == (18, 0, 25, 10)


def test_ties_use_basename_order_independently_of_worker_completion(tmp_path):
    source = {"z": box(0, 0, 10, 10), "a": box(0, 0, 10, 10)}
    result = write_coverage(tmp_path, source, {"z": 1, "a": 1}, image_rank_function="{quality}")
    assert list(result.index) == ["a"]


def test_outline_without_ranking_preserves_overlaps_and_reprojects(tmp_path):
    outline_path = str(tmp_path / "outline.gpkg")
    goal = box(0, 0, 0.00005, 0.00005)
    gpd.GeoDataFrame(geometry=[goal], crs=4326).to_file(outline_path)
    result = write_coverage(tmp_path, {"a": box(0, 0, 10, 10), "b": box(0, 0, 10, 10)},
                            {"a": 1, "b": 2}, outline_path=outline_path)
    expected = transform(Transformer.from_crs(4326, 3857, always_xy=True).transform, goal)
    assert all(geometry.equals(expected) for geometry in result.geometry)


def test_changed_ranks_recompute_joint_coverage_even_with_reuse(tmp_path):
    source = {"a": box(0, 0, 10, 10), "b": box(0, 0, 10, 10)}
    first = write_coverage(tmp_path, source, {"a": 2, "b": 1}, image_rank_function="{quality}", resume_from_outputs="validate")
    second = write_coverage(tmp_path, source, {"a": 1, "b": 2}, image_rank_function="{quality}", resume_from_outputs="validate")
    assert list(first.index) == ["a"]
    assert list(second.index) == ["b"]


@pytest.mark.parametrize("kwargs,match", [
    ({"extrude_distance": -1}, "extrude_distance"),
    ({"extrude_distance": float("nan")}, "extrude_distance"),
    ({"extrude_distance": float("inf")}, "extrude_distance"),
    ({"extrude_distance": True}, "extrude_distance"),
    ({"image_rank_function": ""}, "image_rank_function"),
    ({"image_rank_descending": "false"}, "image_rank_descending"),
    ({"outline_layer": "missing"}, "outline_path"),
])
def test_invalid_coverage_options_do_not_replace_output(tmp_path, kwargs, match):
    output = tmp_path / "coverage.gpkg"
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match=match):
        write_coverage(tmp_path, {"a": box(0, 0, 1, 1)}, {"a": 1}, **kwargs)
    assert output.read_bytes() == b"previous output"


def test_failed_ranking_preserves_previous_output(tmp_path):
    source = {"a": box(0, 0, 10, 10)}
    before = write_coverage(tmp_path, source, {"a": 1})
    with pytest.raises(ValueError, match="missing"):
        write_coverage(tmp_path, source, {"a": 1}, image_rank_function="{missing}")
    assert gpd.read_file(tmp_path / "coverage.gpkg").geometry.iloc[0].equals(before.geometry.iloc[0])
    assert not list(tmp_path.glob(".coverage-*"))


def test_buffer_rejects_geographic_crs_before_reading_rasters(tmp_path):
    with pytest.raises(ValueError, match="projected"):
        create_footprints(["missing.tif"], str(tmp_path / "out.gpkg"),
                          metadata_records=[{}], footprint_source="metadata", extrude_distance=1)


def test_empty_outline_coverage_preserves_existing_output(tmp_path):
    source = {"a": box(0, 0, 10, 10)}
    before = write_coverage(tmp_path, source, {"a": 1})
    outline_path = str(tmp_path / "outline.gpkg")
    gpd.GeoDataFrame(geometry=[box(20, 20, 30, 30)], crs=3857).to_file(outline_path)
    with pytest.raises(ValueError, match="No image coverage"):
        write_coverage(tmp_path, source, {"a": 1}, outline_path=outline_path)
    assert gpd.read_file(tmp_path / "coverage.gpkg").geometry.iloc[0].equals(before.geometry.iloc[0])


@pytest.mark.parametrize("mode", ["serial", "thread", "dask"])
def test_metadata_mode_never_opens_rasters_and_preserves_attributes(tmp_path, monkeypatch, mode):
    def unexpected_read(*args, **kwargs):
        pytest.fail("Metadata geometry must not read raster pixels.")

    monkeypatch.setattr("spectralmatch.seamline.footprints._footprint_from_image", unexpected_read)
    options = {}
    if mode == "thread":
        options["image_threads"] = 2
    elif mode == "dask":
        _install_fake_dask(monkeypatch, {})
        options.update(concurrent_processing_backend="dask", dask_scheduler=("address", "tcp://localhost:8786"))
    source = box(-156, 19, -155, 20)
    output = create_footprints(
        ["not_on_disk_a.tif", "not_on_disk_b.tif"], str(tmp_path / "metadata.gpkg"),
        footprint_source="metadata", epsg=6635,
        metadata_records=[{"geometry": mapping(source), "quality": 2, "note": "first"},
                          {"geometry": mapping(source), "quality": None, "note": "second"}],
        **options,
    )
    frame = gpd.read_file(output).set_index("image")
    expected = transform(Transformer.from_crs(4326, 6635, always_xy=True).transform, source)
    assert frame.crs.to_epsg() == 6635
    assert all(geometry.equals(expected) for geometry in frame.geometry)
    assert frame.loc["not_on_disk_a", "quality"] == 2
    assert frame.loc["not_on_disk_b", "note"] == "second"


@pytest.mark.parametrize("options,message", [
    ({"metadata_records": []}, "one dictionary"),
    ({"metadata_records": [{"image_path": "x"}]}, "conflict"),
    ({"metadata_records": [{"Quality": 2, "quality": 3}]}, "unique ignoring case"),
    ({"metadata_records": [{"quality": [2]}]}, "scalar"),
    ({"metadata_records": [{"quality": float("nan")}]}, "finite"),
    ({"metadata_records": [{}], "metadata_csv": "metadata.csv"}, "mutually exclusive"),
    ({"footprint_source": "metadata"}, "requires metadata_records"),
    ({"footprint_source": "metadata", "metadata_records": [{}]}, "GeoJSON polygon"),
    ({"footprint_source": "metadata", "metadata_records": [{"geometry": mapping(box(0, 0, 1, 1).boundary)}]}, "Polygon"),
    ({"footprint_source": "unknown"}, "footprint_source"),
    ({"epsg": True}, "epsg"),
])
def test_metadata_validation_preserves_existing_outputs(tmp_path, options, message):
    output = tmp_path / "output.gpkg"
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match=message):
        create_footprints(["missing.tif"], str(output), **options)
    assert output.read_bytes() == b"previous output"


def test_raster_mode_supports_outline_ranking_and_output_projection(tmp_path):
    image = str(tmp_path / "image.tif")
    create_dummy_raster(image, width=4, height=4, count=1, crs="EPSG:4326", transform=(0, 1, 0, 4, 0, -1), fill_value=1)
    outline = str(tmp_path / "outline.gpkg")
    gpd.GeoDataFrame(geometry=[box(1, 1, 3, 3)], crs=4326).to_file(outline)
    output = create_footprints([image], str(tmp_path / "projected.gpkg"),
                               metadata_records=[{"quality": 4}], epsg=3857,
                               outline_path=outline, image_rank_function="{quality}")
    result = gpd.read_file(output)
    expected = transform(Transformer.from_crs(4326, 3857, always_xy=True).transform, box(1, 1, 3, 3))
    assert result.crs.to_epsg() == 3857
    assert result.geometry.iloc[0].equals(expected)
    assert result.weighted_score.iloc[0] == 4


