import inspect
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from spectralmatch import (
    Seamline,
    create_footprints,
    markov_triangles,
    pipeline,
    postprocess_footprints,
)
from spectralmatch.seamline.footprints import _read_image_metadata
from .test_dask_execution import _install_fake_dask
from .utils_test import create_dummy_raster


@pytest.fixture
def images(tmp_path):
    paths = [str(tmp_path / f"scene_{index}.tif") for index in range(2)]
    for index, path in enumerate(paths):
        create_dummy_raster(
            path,
            width=16,
            height=16,
            count=1,
            crs="EPSG:32604",
            transform=(index * 8, 1, 0, 16, 0, -1),
            fill_value=100,
        )
    return paths


@pytest.fixture
def metadata(tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text(
        "source,quality_score,cloud_cover,sun_elevation,off_nadir_angle,note,optional\n"
        "archive/SCENE_1.tif,90,5,63.5,8,preferred,\n"
        "archive/scene_0.tif,50,15,60.5,18,secondary,2\n"
        "unused.tif,20,90,40,30,unused,3\n"
    )
    return str(path)


@pytest.mark.parametrize("mode", ["serial", "thread", "dask"])
def test_csv_metadata_survives_parallel_footprints_postprocessing_and_ranking(
    tmp_path, monkeypatch, images, metadata, mode
):
    options = {}
    if mode == "thread":
        options["image_threads"] = 2
    elif mode == "dask":
        _install_fake_dask(monkeypatch, {})
        options.update(
            concurrent_processing_backend="dask",
            dask_scheduler=("address", "tcp://localhost:8786"),
        )
    source = create_footprints(
        images,
        str(tmp_path / "raw.gpkg"),
        metadata_csv=metadata,
        metadata_image_field_name="source",
        image_field_name="scene",
        **options,
    )
    original = gpd.read_file(source, layer="footprints").set_index("scene")
    assert "source" not in original
    assert original.loc["scene_1", "quality_score"] == 90
    assert original.loc["scene_0", "cloud_cover"] == 15
    assert original.loc["scene_1", "note"] == "preferred"
    assert pd.isna(original.loc["scene_1", "optional"])
    assert original.loc["scene_0", "optional"] == 2
    assert pd.api.types.is_integer_dtype(original.quality_score)
    assert pd.api.types.is_float_dtype(original.sun_elevation)
    processed = postprocess_footprints(
        source,
        str(tmp_path / "processed.gpkg"),
        simplify_smoothing_radius=0,
        simplify_tolerance=0,
        **options,
    )
    actual = gpd.read_file(processed, layer="footprints").set_index("scene")
    pd.testing.assert_frame_equal(
        original.drop(columns="geometry").sort_index(),
        actual.drop(columns="geometry").sort_index(),
    )
    result = Seamline.weighted(
        processed,
        str(tmp_path / "weighted.gpkg"),
        image_field_name="scene",
        rank_function="{quality_score} - {cloud_cover} + {sun_elevation} - {off_nadir_angle}",
    )
    ranked = gpd.read_file(result, layer="seamlines").set_index("scene")
    assert ranked.loc["scene_1", "weighted_score"] == 140.5
    assert ranked.loc["scene_0", "weighted_score"] == 77.5
    assert ranked.loc["scene_1"].geometry.area == 256
    assert ranked.loc["scene_0"].geometry.area == 128


def test_csv_metadata_drives_markov_image_quality(tmp_path, images, metadata):
    pytest.importorskip("maxflow")
    source = create_footprints(
        images,
        str(tmp_path / "raw.gpkg"),
        metadata_csv=metadata,
        metadata_image_field_name="source",
    )
    result = markov_triangles(
        images,
        str(tmp_path / "markov.gpkg"),
        input_polygons=source,
        image_rank_function="{quality_score} - {cloud_cover}",
        edge_variables_weights_gsds_bands=[],
        mesh_spacing=4,
    )
    ranked = gpd.read_file(result, layer="seamlines").set_index("image")
    assert ranked.loc["scene_1", "quality_score"] == 85
    assert ranked.loc["scene_0", "quality_score"] == 35
    assert ranked.loc["scene_1"].geometry.area == 256
    assert ranked.loc["scene_0"].geometry.area == 128


@pytest.mark.parametrize(
    "contents,message",
    [
        ("image,quality\nscene_0,5\n", "scene_1.*found 0"),
        ("image,quality\nscene_0,5\nother_scene_0,6\nscene_1,7\n", "scene_0.*found 2"),
        ("filename,quality\nscene_0,5\n", "no join column"),
        ("image,quality\n,5\n", "empty values"),
        ("image,quality\n  ,5\n", "empty values"),
        ("image,image_path\nscene_0,x\n", "conflict with output fields"),
        ("image,Geometry\nscene_0,x\n", "conflict with output fields"),
        ("image,fid\nscene_0,1\n", "conflict with output fields"),
        ("image,quality,Quality\nscene_0,5,6\n", "unique ignoring case"),
        ("image,\nscene_0,5\n", "nonempty column names"),
        ("", "nonempty column names"),
    ],
)
def test_csv_validation_precedes_pixel_reads_and_output_changes(
    tmp_path, monkeypatch, images, contents, message
):
    path = tmp_path / "invalid.csv"
    path.write_text(contents)
    output = tmp_path / "untouched.gpkg"
    output.write_bytes(b"existing output")

    def unexpected_read(*args, **kwargs):
        pytest.fail("Raster workers must not run for invalid metadata.")

    monkeypatch.setattr(
        "spectralmatch.seamline.footprints._footprint_from_image", unexpected_read
    )
    with pytest.raises(ValueError, match=message):
        create_footprints(images, str(output), metadata_csv=str(path))
    assert output.read_bytes() == b"existing output"
    assert not Path(str(output) + ".incomplete").exists()


def test_metadata_joins_preserve_numeric_names_and_match_literals(tmp_path):
    path = tmp_path / "names.csv"
    path.write_text("image,quality\n001,4\nPREFIX_A[1].TIF,5\nNA,6\n")
    schema, rows = _read_image_metadata(
        str(path), "image", ["001", "a[1]", "NA"], "image"
    )
    assert schema == {"quality": "int64"}
    assert rows == [{"quality": 4}, {"quality": 5}, {"quality": 6}]


@pytest.mark.parametrize("extension", ["", ".tif"])
def test_csv_field_must_contain_full_processed_basename(tmp_path, extension):
    name = "Worldview_20160922_Coregistered_Global_Local"
    path = tmp_path / "metadata.csv"
    path.write_text(f"image,quality_score\n{ name.lower() }{extension},90\n")
    assert _read_image_metadata(str(path), "image", [name], "image")[1] == [
        {"quality_score": 90}
    ]
    path.write_text(f"image,quality_score\nWorldview_20160922{extension},90\n")
    with pytest.raises(
        ValueError, match="full current raster basename.*processing suffixes"
    ):
        _read_image_metadata(str(path), "image", [name], "image")


def test_worldview_csv_matches_example_localmatch_outputs():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/examples/data_worldview/Input/ImageMetadata.csv"
    )
    names = [
        f"Worldview_{date}_Coregistered_Global_Local"
        for date in (20160922, 20160923, 20160930)
    ]
    _, rows = _read_image_metadata(str(path), "image", names, "image")
    assert [row["quality_score"] for row in rows] == [90, 85, 95]


def test_metadata_cannot_overwrite_custom_image_field(tmp_path):
    path = tmp_path / "names.csv"
    path.write_text("source,scene\nA.tif,4\n")
    with pytest.raises(ValueError, match="conflict with output fields"):
        _read_image_metadata(str(path), "source", ["A"], "scene")


def test_metadata_with_only_join_column_is_valid(tmp_path):
    path = tmp_path / "names.csv"
    path.write_text("image\nA.tif\n")
    assert _read_image_metadata(str(path), "image", ["A"], "image") == ({}, [{}])


@pytest.mark.parametrize("step", ["weighted_seamline", "voronoi_center_seamline"])
def test_pipeline_forwards_csv_for_explicit_and_implicit_footprints(
    tmp_path, images, metadata, step
):
    steps = (
        ("create_footprints", "postprocess_footprints", step)
        if step == "weighted_seamline"
        else (step,)
    )
    result = pipeline(
        shared_input_images=images,
        shared_output_image_path=str(tmp_path / "seamlines.gpkg"),
        shared_temp_dir=str(tmp_path / "work"),
        delete_temp_dir=False,
        steps=steps,
        shared_image_threads=None,
        create_footprints_metadata_csv=metadata,
        create_footprints_metadata_image_field_name="source",
        create_footprints_output_layer="raw",
        postprocess_footprints_output_layer="processed",
        postprocess_footprints_simplify_smoothing_radius=0,
        postprocess_footprints_simplify_tolerance=0,
        weighted_seamline_rank_function="{quality_score} - {cloud_cover}",
    )
    footprints = gpd.read_file(result["create_footprints"], layer="raw").set_index(
        "image"
    )
    assert footprints.loc["scene_1", "quality_score"] == 90
    if step == "weighted_seamline":
        ranked = gpd.read_file(result["output"], layer="seamlines").set_index("image")
        assert ranked.loc["scene_1", "weighted_score"] == 85


@pytest.mark.parametrize(
    "method", [Seamline.postprocess_footprints, Seamline.weighted, Seamline.voronoi]
)
@pytest.mark.parametrize("input_layer", ["footprints", None])
def test_default_layer_selects_footprints_and_none_selects_first(
    tmp_path, method, input_layer
):
    source = str(tmp_path / "multilayer.gpkg")
    for layer, name in (("first", "first_image"), ("footprints", "expected_image")):
        gpd.GeoDataFrame(
            {"image": [name], "quality": [1]},
            geometry=[box(0, 0, 10, 10)],
            crs=32604,
        ).to_file(source, layer=layer, driver="GPKG")
    options = {} if input_layer == "footprints" else {"input_layer": None}
    if method is Seamline.weighted:
        options["rank_function"] = "{quality}"
    elif method is Seamline.postprocess_footprints:
        options.update(simplify_smoothing_radius=0, simplify_tolerance=0)
    result = method(source, str(tmp_path / "output.gpkg"), **options)
    expected = "expected_image" if input_layer == "footprints" else "first_image"
    assert gpd.read_file(result).image.tolist() == [expected]


def test_public_footprint_field_and_layer_defaults_are_consistent():
    for method in (
        Seamline.create_footprints,
        Seamline.postprocess_footprints,
        Seamline.weighted,
        Seamline.voronoi,
        Seamline.markov_triangles,
    ):
        parameters = inspect.signature(method).parameters
        if "input_layer" in parameters:
            assert parameters["input_layer"].default == "footprints"
        if "image_field_name" in parameters:
            assert parameters["image_field_name"].default == "image"
