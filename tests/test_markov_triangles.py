"""Paper-method checks: graph cuts, radiometry, coverage and hierarchy execution."""

import importlib
import inspect
import math
import multiprocessing as mp
from collections import Counter
from itertools import product
import os

import geopandas as gpd
import numpy as np
import pytest
import pandas as pd
from osgeo import gdal, osr
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from spectralmatch import Seamline, create_footprints, markov_triangles
from spectralmatch.cli import _build_cli
from spectralmatch.seamline.markov_triangles import _alpha_expansion
from .utils_test import create_dummy_raster
from .test_dask_execution import _install_fake_dask

pytest.importorskip("maxflow")
pytest.importorskip("shapely", minversion="2.1")
module = importlib.import_module("spectralmatch.seamline.markov_triangles")


def _inputs(tmp_path, offsets=(0, 8), *, field="image", crs="EPSG:32604"):
    paths = []
    for index, offset in enumerate(offsets):
        path = tmp_path / f"image{index}.tif"
        create_dummy_raster(
            path,
            width=16,
            height=16,
            count=1,
            crs=crs,
            transform=(offset, 1, 0, 16, 0, -1),
            fill_value=100 + index * 20,
        )
        paths.append(str(path))
    return create_footprints(
        paths, str(tmp_path / "footprints.gpkg"), image_field_name=field
    )


def _run(source, output, **kwargs):
    paths = gpd.read_file(source).image_path.drop_duplicates().tolist()
    return markov_triangles(paths, output, input_polygons=source, **kwargs)


def _assert_partition(source, result, field="image"):
    original = gpd.read_file(source).dissolve(by=field).geometry
    output = gpd.read_file(result)
    assert output.crs == original.crs
    assert output.geometry.is_valid.all()
    assert output[field].is_unique
    assert (
        unary_union(output.geometry).symmetric_difference(unary_union(original)).area
        < 1e-7
    )
    for row in output.itertuples(index=False):
        assert row.geometry.difference(original.loc[getattr(row, field)]).area < 1e-7
    assert sum(output.geometry.area) - unary_union(output.geometry).area < 1e-7
    return output


def _sample_forked_costs(connection, paths):
    """Report raster costs from a child inheriting an initialized GDAL thread pool."""
    try:
        cost, _, _ = module._edge_costs(
            [{"left": box(0, 0, 16, 16)}, {"right": box(8, 0, 24, 16)}],
            dict(zip(("left", "right"), paths)),
            [LineString([(10, 4), (10, 12)])],
            module._variables([["laplacian_difference", 1, [1], [1]]]),
            "EPSG:32604",
        )
        connection.send(("ok", cost(np.array([0]), np.array([1])).tolist()))
    except BaseException as exc:
        connection.send(("error", repr(exc)))
    finally:
        connection.close()


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="Requires fork")
def test_pixel_sampling_after_parent_multithreaded_gdal_does_not_deadlock(tmp_path):
    source = _inputs(tmp_path)
    paths = gpd.read_file(source).image_path.tolist()
    context = mp.get_context("fork")
    receiving, sending = context.Pipe(duplex=False)
    worker = context.Process(target=_sample_forked_costs, args=(sending, paths))
    with gdal.config_option("GDAL_NUM_THREADS", "2", thread_local=False):
        # Initialize GDAL's process-wide worker pool before creating a child.
        warmed = gdal.Warp(
            "", paths[0], format="MEM", width=1024, height=1024, resampleAlg="bilinear"
        )
        assert warmed.ReadAsArray().shape == (1024, 1024)
        warmed = None
        worker.start()
        sending.close()
        try:
            assert receiving.poll(
                20
            ), "Forked raster sampling deadlocked after GDAL threading."
            assert receiving.recv() == ("ok", [0.0])
            worker.join(5)
            assert worker.exitcode == 0
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
            receiving.close()


@pytest.mark.parametrize("fail", [False, True])
def test_raster_sampling_restores_gdal_thread_settings(monkeypatch, fail):
    def sample(*args):
        assert gdal.GetConfigOption("GDAL_NUM_THREADS") == "1"
        if fail:
            raise ValueError("sample failure")
        return np.zeros((1, 1))

    monkeypatch.setattr(module, "_laplacian", sample)
    terms = module._variables([["laplacian_difference", 1, [1], [1]]])
    plans = {1: (np.array([0]), np.array([0]), np.array([[0.5, 0.5]]))}
    with gdal.config_option("GDAL_NUM_THREADS", "3"):
        if fail:
            with pytest.raises(ValueError, match="sample failure"):
                module._shared_samples([{}], {}, plans, terms, "EPSG:32604")
        else:
            module._shared_samples([{}], {}, plans, terms, "EPSG:32604")
        assert gdal.GetConfigOption("GDAL_NUM_THREADS") == "3"


@pytest.mark.parametrize("debug_logs", [False, True])
def test_node_progress_reports_stages(tmp_path, capsys, debug_logs):
    source = _inputs(tmp_path)
    _run(source, str(tmp_path / "out.gpkg"), mesh_spacing=4, debug_logs=debug_logs)
    progress = capsys.readouterr().out
    for stage in (
        "building mesh",
        "finding shared edges",
        "sampling raster costs",
        "optimizing labels",
        "merging labeled triangles",
        "completed 2 image polygons",
    ):
        assert (stage in progress) == debug_logs


@pytest.mark.parametrize("nlabels", [2, 3, 4])
def test_expansion_against_exhaustive_moves(nlabels):
    rng = np.random.default_rng(931)
    edges = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 4)])
    for _ in range(8):
        signatures = rng.normal(size=(len(edges), nlabels, 3))
        pairwise = np.abs(signatures[:, :, None] - signatures[:, None, :]).max(axis=3)
        visible = rng.random((5, nlabels)) > 0.4
        visible[np.arange(5), rng.integers(nlabels, size=5)] = True
        cost = lambda a, b: pairwise[np.arange(len(edges)), a, b]
        result = _alpha_expansion(
            visible, edges, cost, pairwise.max(axis=(1, 2)).sum(), 50
        )
        energy = cost(result[edges[:, 0]], result[edges[:, 1]]).sum()
        assert visible[np.arange(5), result].all()
        for alpha in range(nlabels):
            for switches in product([False, True], repeat=5):
                proposal = np.where(switches, alpha, result)
                if visible[np.arange(5), proposal].all():
                    assert (
                        cost(proposal[edges[:, 0]], proposal[edges[:, 1]]).sum()
                        >= energy - 1e-10
                    )
        if nlabels == 2:
            feasible = [
                np.array(p)
                for p in product(range(2), repeat=5)
                if visible[np.arange(5), p].all()
            ]
            assert energy == pytest.approx(
                min(cost(p[edges[:, 0]], p[edges[:, 1]]).sum() for p in feasible)
            )


@pytest.mark.parametrize("workers", [None, 2])
def test_markov_basic_api_and_parallel_hierarchy(tmp_path, workers):
    source = _inputs(tmp_path, offsets=(0, 10, 20, 30), field="source")
    output = str(tmp_path / "nested" / "seamlines.gpkg")
    assert Seamline.markov_triangles is markov_triangles
    assert _build_cli().markov_triangles is markov_triangles
    assert (
        markov_triangles(
            input_images=list(gpd.read_file(source).image_path),
            input_polygons=source,
            input_layer="footprints",
            output_mask=output,
            image_field_name="source",
            mesh_spacing=4,
            quadtree_max_images_per_leaf=2,
            image_threads=workers,
            debug_logs=True,
        )
        == output
    )
    _assert_partition(source, output, "source")
    assert not os.path.exists(output + ".incomplete")


def test_hierarchy_matches_across_serial_process_and_dask(tmp_path, monkeypatch):

    source = _inputs(tmp_path, offsets=(0, 10, 20, 30))
    outputs = []
    calls = {}
    _install_fake_dask(monkeypatch, calls)
    for mode in (
        {},
        {"image_threads": 2},
        {
            "concurrent_processing_backend": "dask",
            "dask_scheduler": ("address", "tcp://scheduler:8786"),
        },
    ):
        path = str(tmp_path / f"out{len(outputs)}.gpkg")
        _run(source, path, mesh_spacing=4, quadtree_max_images_per_leaf=2, **mode)
        outputs.append(_assert_partition(source, path).set_index("image"))
    for output in outputs[1:]:
        assert set(output.index) == set(outputs[0].index)
        for name in output.index:
            assert (
                output.loc[name]
                .geometry.symmetric_difference(outputs[0].loc[name].geometry)
                .area
                < 1e-7
            )
    assert calls["closed"]


def test_narrow_islands_holes_and_duplicate_image_ids(tmp_path):
    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    first = Polygon(
        box(0, 0, 16, 16).exterior.coords, [box(6.1, 6.1, 6.3, 6.3).exterior.coords]
    )
    second = MultiPolygon([box(8, 0, 12, 16), box(20, 2, 20.1, 2.2)])
    frame.geometry = [first, second]
    duplicate = frame.iloc[[0]].copy()
    duplicate.geometry = [box(1, 1, 2, 2)]

    gpd.GeoDataFrame(pd.concat([frame, duplicate]), crs=frame.crs).to_file(
        source, layer="footprints"
    )
    output = _run(source, str(tmp_path / "out.gpkg"), mesh_spacing=5)
    _assert_partition(source, output)


def test_divergence_and_low_cost_seam(tmp_path):
    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    # A constant first image; second image has high texture except a quiet
    # vertical corridor in the overlap. The cut should move into the corridor.
    grid = np.indices((16, 16)).sum(axis=0)
    pixels = np.where(grid % 2, 130, 70).astype("float32")
    pixels[:, :6] = 100
    create_dummy_raster(
        frame.image_path.iloc[1],
        band_data=pixels,
        nodata=None,
        crs="EPSG:32604",
        transform=(8, 1, 0, 16, 0, -1),
    )
    output = _run(source, str(tmp_path / "out.gpkg"), mesh_spacing=1)
    result = _assert_partition(source, output).set_index("image")
    seam = result.loc["image0"].geometry.intersection(result.loc["image1"].geometry)
    assert seam.length > 0
    assert 8 <= seam.bounds[0] <= seam.bounds[2] <= 13

    quadratic = (np.indices((16, 16))[1] ** 2).astype("float32")
    create_dummy_raster(
        frame.image_path.iloc[0],
        band_data=quadratic,
        nodata=None,
        crs="EPSG:32604",
        transform=(0, 1, 0, 16, 0, -1),
    )
    divergence = module._laplacian(
        {"A": box(0, 0, 16, 16)},
        {"A": frame.image_path.iloc[0]},
        np.array([[5.5, 5.5], [8.5, 8.5]]),
        (1,),
        1,
        module._RasterSampler("EPSG:32604"),
    )
    assert np.allclose(divergence, 2)


def test_foreground_protection_and_infeasible_constraints(tmp_path):
    source = _inputs(tmp_path)
    mask = str(tmp_path / "objects.gpkg")
    protected = box(10, 1, 18, 15)
    gpd.GeoDataFrame(geometry=[protected], crs=32604).to_file(mask)
    output = _run(
        source, str(tmp_path / "out.gpkg"), mesh_spacing=2, foreground_path=mask
    )
    result = _assert_partition(source, output).set_index("image")
    assert result.loc["image1"].geometry.covers(protected)
    gpd.GeoDataFrame(geometry=[box(0, 0, 24, 16)], crs=32604).to_file(mask)
    with pytest.raises(ValueError, match="infeasible"):
        _run(source, str(tmp_path / "bad.gpkg"), mesh_spacing=2, foreground_path=mask)


def test_resume_failure_marker_and_input_protection(tmp_path, monkeypatch):
    source = _inputs(tmp_path)
    output = str(tmp_path / "out.gpkg")
    _run(source, output, mesh_spacing=4)
    actual = module._solve_node
    monkeypatch.setattr(
        module,
        "_solve_node",
        lambda *args: (_ for _ in ()).throw(RuntimeError("worker failed")),
    )
    assert _run(source, output, resume_from_outputs="validate") == output
    with pytest.raises(RuntimeError, match="worker failed"):
        _run(source, output)
    assert os.path.exists(output + ".incomplete")
    with pytest.raises(RuntimeError, match="worker failed"):
        _run(source, output, resume_from_outputs="yes")
    monkeypatch.setattr(module, "_solve_node", actual)
    _run(source, output, mesh_spacing=4, resume_from_outputs="yes")
    assert not os.path.exists(output + ".incomplete")
    with pytest.raises(ValueError, match="different files"):
        _run(source, source)


@pytest.mark.parametrize(
    "params",
    [
        {"mesh_spacing": 0},
        {"mesh_spacing": float("nan")},
        {"image_quality_weight": -1},
        {"image_quality_weight": float("inf")},
        {"quadtree_overlap": 0.5},
        {"quadtree_overlap": 0},
        {"quadtree_max_images_per_leaf": False},
        {"quadtree_max_depth": -1},
        {"solver_max_iterations": 0},
        {"image_threads": True},
        {"resume_from_outputs": "maybe"},
        {"foreground_layer": "objects"},
    ],
)
def test_parameter_validation(tmp_path, params):
    with pytest.raises(ValueError):
        markov_triangles("unused.tif", str(tmp_path / "out.gpkg"), **params)


def test_sources_use_inputs_not_path_attributes(tmp_path, monkeypatch):
    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    paths = frame.image_path.tolist()
    frame.image_path = "does-not-exist.tif"
    frame.image = ["some/path/image0.tif", "prefix_image1_suffix"]
    frame.to_file(source, layer="footprints")
    footprints = importlib.import_module("spectralmatch.seamline.footprints")
    monkeypatch.setattr(
        footprints,
        "_footprint_from_image",
        lambda *args: pytest.fail("Supplied footprints must not be recalculated"),
    )
    output = markov_triangles(
        paths, str(tmp_path / "matched.gpkg"), input_polygons=source, mesh_spacing=4
    )
    result = gpd.read_file(output)
    assert set(result.image) == {"image0", "image1"}
    assert set(result.image_path) == set(paths)
    with pytest.raises(ValueError, match="bands"):
        markov_triangles(
            paths,
            str(tmp_path / "bands.gpkg"),
            input_polygons=source,
            edge_variables_weights_gsds_bands=[["laplacian_difference", 1, [1], [2]]],
        )
    frame.iloc[:1].to_file(source, layer="footprints")
    with pytest.raises(ValueError, match="No footprint.*image1"):
        markov_triangles(paths, str(tmp_path / "missing.gpkg"), input_polygons=source)
    frame.set_crs(4326, allow_override=True).to_file(source, layer="footprints")
    with pytest.raises(ValueError, match="projected CRS"):
        markov_triangles(
            paths, str(tmp_path / "geographic.gpkg"), input_polygons=source
        )


def test_real_nodata_hole_boundary_and_rotated_rasters(tmp_path):

    source = _inputs(tmp_path, offsets=(0, 4))
    paths = list(gpd.read_file(source).image_path)
    with gdal.Open(paths[0], gdal.GA_Update) as dataset:
        pixels = dataset.GetRasterBand(1).ReadAsArray()
        pixels[4:8, 6:8] = 0
        dataset.GetRasterBand(1).WriteArray(pixels)
    create_footprints(paths, source)
    output = _run(source, str(tmp_path / "hole.gpkg"), mesh_spacing=2)
    _assert_partition(source, output)
    for index, path in enumerate(paths):
        create_dummy_raster(
            path,
            width=12,
            height=12,
            count=1,
            crs="EPSG:32604",
            transform=(index * 8, 2, 1, 30, 1, -2),
        )
    create_footprints(paths, source)
    output = _run(source, str(tmp_path / "rotated.gpkg"), mesh_spacing=3)
    _assert_partition(source, output)


def test_missing_mask_samples_are_not_silently_used(tmp_path):

    source = _inputs(tmp_path)
    path = gpd.read_file(source).image_path.iloc[0]
    with gdal.Open(path, gdal.GA_Update) as dataset:
        dataset.GetRasterBand(1).Fill(0)
    with pytest.raises(ValueError, match="nodata"):
        _run(source, str(tmp_path / "out.gpkg"), mesh_spacing=2)


def test_raster_crs_and_footprint_extent_validation(tmp_path):

    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(32605)
    with gdal.Open(frame.image_path.iloc[0], gdal.GA_Update) as dataset:
        dataset.SetProjection(crs.ExportToWkt())
    with pytest.raises(ValueError, match="same CRS"):
        _run(source, str(tmp_path / "out.gpkg"))
    crs.ImportFromEPSG(32604)
    with gdal.Open(frame.image_path.iloc[0], gdal.GA_Update) as dataset:
        dataset.SetProjection(crs.ExportToWkt())
    frame.geometry = [box(-5, -5, 20, 20), frame.geometry.iloc[1]]
    frame.to_file(source, layer="footprints")
    with pytest.raises(ValueError, match="outside its raster"):
        _run(source, str(tmp_path / "out.gpkg"))


def test_auto_footprints_single_and_disjoint(tmp_path):
    source = _inputs(tmp_path, offsets=(0, 30))
    frame = gpd.read_file(source)
    result = markov_triangles(str(tmp_path / "*.tif"), str(tmp_path / "disjoint.gpkg"))
    _assert_partition(source, result)
    frame.iloc[:1].to_file(source, layer="footprints")
    result = markov_triangles([frame.image_path.iloc[0]], str(tmp_path / "single.gpkg"))
    _assert_partition(source, result)


def test_validate_resume_recomputes_wrong_layer(tmp_path):
    source = _inputs(tmp_path)
    output = str(tmp_path / "out.gpkg")
    gpd.GeoDataFrame({"wrong": ["A"]}, geometry=[box(0, 0, 1, 1)], crs=32604).to_file(
        output, layer="unrelated"
    )
    _run(source, output, mesh_spacing=4, resume_from_outputs="validate")
    result = gpd.read_file(output, layer="seamlines")
    assert "image" in result and sum(result.geometry.area) == pytest.approx(384)


def test_default_band_and_explicit_band_selection(tmp_path, monkeypatch):
    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    for row in frame.itertuples():
        create_dummy_raster(
            row.image_path,
            width=16,
            height=16,
            count=3,
            crs="EPSG:32604",
            transform=(row.geometry.bounds[0], 1, 0, 16, 0, -1),
        )
    actual = module._laplacian
    selected = []

    def capture(candidate, paths, coordinates, bands, spacing, sampler):
        selected.append(bands)
        return actual(candidate, paths, coordinates, bands, spacing, sampler)

    monkeypatch.setattr(module, "_laplacian", capture)
    _run(source, str(tmp_path / "rgb.gpkg"), mesh_spacing=4)
    assert selected and set(selected) == {(1,)}
    selected.clear()
    _run(
        source,
        str(tmp_path / "red.gpkg"),
        mesh_spacing=4,
        edge_variables_weights_gsds_bands=[["laplacian_difference", 1, [1], [2]]],
    )
    assert selected and set(selected) == {(2,)}


def test_validate_resume_replaces_corrupt_output(tmp_path):
    source = _inputs(tmp_path)
    output = tmp_path / "corrupt.gpkg"
    output.write_text("incomplete file")
    _run(source, str(output), mesh_spacing=4, resume_from_outputs="validate")
    _assert_partition(source, str(output))


@pytest.mark.parametrize("scale_reducer", ["average", "largest", "smallest"])
@pytest.mark.parametrize("band_reducer", ["average", "largest", "smallest"])
def test_variable_reducers_and_multipliers(monkeypatch, scale_reducer, band_reducer):
    values = np.array([[[2, -6], [1, 2]], [[10, 3], [4, 8]]], dtype=float)

    def laplacian(candidate, paths, coordinates, bands, gsd, sampler):
        label = int(next(iter(candidate)))
        return np.broadcast_to(values[int(gsd) - 1, label], (len(coordinates), 2))

    monkeypatch.setattr(module, "_laplacian", laplacian)
    candidates = [{str(i): box(0, 0, 10, 10)} for i in range(2)]
    difference = [
        "laplacian_difference",
        2,
        {scale_reducer: [1, 2]},
        {band_reducer: [1, 2]},
    ]
    magnitude = [
        "laplacian_magnitude",
        0.5,
        {scale_reducer: [1, 2]},
        {band_reducer: [1, 2]},
    ]
    cost, bound, swap = module._edge_costs(
        candidates,
        {},
        [LineString([(1, 1), (2, 1)])],
        module._variables([difference, magnitude]),
        "EPSG:32604",
    )
    reduce_scale = {"average": np.mean, "largest": np.max, "smallest": np.min}[
        scale_reducer
    ]
    reduce_band = {"average": np.mean, "largest": np.max, "smallest": np.min}[
        band_reducer
    ]
    expected_difference = reduce_scale(
        reduce_band(np.abs(values[:, 0] - values[:, 1]), axis=1)
    )
    expected_magnitude = reduce_scale(
        reduce_band(np.abs(values), axis=2), axis=0
    ).mean()
    expected = 2 * expected_difference + 0.5 * expected_magnitude
    assert cost(np.array([0]), np.array([1]))[0] == pytest.approx(expected)
    assert cost(np.array([1]), np.array([0]))[0] == pytest.approx(expected)
    assert cost(np.array([0]), np.array([0]))[0] == 0
    assert bound >= expected
    assert swap == ("smallest" in (scale_reducer, band_reducer))


@pytest.mark.parametrize(
    "variable",
    [
        ["laplacian_difference", -1, [1], [1]],
        ["laplacian_difference", float("nan"), [1], [1]],
        ["laplacian_difference", True, [1], [1]],
        ["laplacian_difference", 1, [0], [1]],
        ["laplacian_difference", 1, [float("inf")], [1]],
        ["laplacian_difference", 1, [1], [0]],
        ["laplacian_difference", 1, [1], [True]],
        ["laplacian_difference", 1, [1], {"average": [1, 1]}],
        ["laplacian_difference", 1, {"median": [1]}, [1]],
        ["laplacian_difference", 1, {"average": []}, [1]],
        ["laplacian_difference", 1, {"average": [1], "largest": [2]}, [1]],
        ["laplacian_difference", 1],
    ],
)
def test_invalid_variable_specs(tmp_path, variable):
    with pytest.raises(ValueError):
        markov_triangles(
            "unused.tif",
            str(tmp_path / "out.gpkg"),
            edge_variables_weights_gsds_bands=[variable],
        )


def test_zero_weights_skip_reads_and_match_difference_only(tmp_path, monkeypatch):
    source = _inputs(tmp_path)
    disabled = [
        ["does-not-exist.tif", 0, [1], [1]],
        ["laplacian_magnitude", 0, [1], [1]],
    ]
    actual = module._RasterSampler.sample
    monkeypatch.setattr(
        module._RasterSampler,
        "sample",
        lambda *args, **kwargs: pytest.fail("Disabled terms must not read pixels"),
    )
    result = _run(
        source, str(tmp_path / "none.gpkg"), edge_variables_weights_gsds_bands=disabled
    )
    _assert_partition(source, result)
    monkeypatch.setattr(module._RasterSampler, "sample", actual)
    difference = [["laplacian_difference", 1, [1], [1]]]
    a = gpd.read_file(
        _run(
            source,
            str(tmp_path / "difference.gpkg"),
            edge_variables_weights_gsds_bands=difference,
        )
    ).set_index("image")
    b = gpd.read_file(
        _run(
            source,
            str(tmp_path / "disabled.gpkg"),
            edge_variables_weights_gsds_bands=difference + disabled,
        )
    ).set_index("image")
    assert all(a.loc[name].geometry.equals(b.loc[name].geometry) for name in a.index)


@pytest.mark.parametrize("descending", [True, False])
def test_quality_prefers_more_area_and_saves_expression(tmp_path, descending):
    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    frame["quality"] = [10, 90]
    frame["clouds"] = [2, 4]
    frame.to_file(source, layer="footprints")
    output = _run(
        source,
        str(tmp_path / "quality.gpkg"),
        edge_variables_weights_gsds_bands=[],
        image_rank_function="{quality} - {clouds}",
        image_rank_descending=descending,
        image_quality_weight=2,
        mesh_spacing=3,
    )
    result = _assert_partition(source, output).set_index("image")
    favored = "image1" if descending else "image0"
    assert result.loc[favored].geometry.area == pytest.approx(256)
    assert result.quality_score.to_dict() == {"image0": 8, "image1": 86}
    weighted = Seamline.weighted(
        source,
        str(tmp_path / "weighted.gpkg"),
        rank_function="{quality} - {clouds}",
        rank_descending=descending,
    )
    scores = gpd.read_file(weighted).set_index("image").weighted_score.to_dict()
    assert scores == result.quality_score.to_dict()
    zero = _run(
        source,
        str(tmp_path / "zero.gpkg"),
        edge_variables_weights_gsds_bands=[],
        image_rank_function="{quality}",
        image_quality_weight=0,
        mesh_spacing=3,
    )
    assert gpd.read_file(zero).set_index("image").loc[
        "image0"
    ].geometry.area == pytest.approx(256)


@pytest.mark.parametrize("swap", [False, True])
def test_graph_quality_unaries_against_exhaustive_solutions(swap):
    rng = np.random.default_rng(52)
    edges = np.array([(0, 1), (0, 2), (1, 3), (2, 3)])
    nlabels = 3 if swap else 2
    for _ in range(8):
        features = rng.normal(size=(len(edges), nlabels, 3))
        distances = np.abs(features[:, :, None] - features[:, None, :])
        pairwise = distances.min(axis=3) if swap else distances.max(axis=3)
        cost = lambda a, b: pairwise[np.arange(len(edges)), a, b]
        visible = rng.random((4, nlabels)) > 0.3
        visible[:, 0] = True
        unary = rng.random((4, nlabels)) * 2
        labels = module._alpha_expansion(
            visible,
            edges,
            cost,
            pairwise.max(axis=(1, 2)).sum(),
            50,
            unary=unary,
            swap=swap,
        )
        energy = (
            lambda p: unary[np.arange(4), p].sum()
            + cost(p[edges[:, 0]], p[edges[:, 1]]).sum()
        )
        for proposed in product(range(nlabels), repeat=4):
            proposed = np.array(proposed)
            if not visible[np.arange(4), proposed].all():
                continue
            changed_labels = set(labels[labels != proposed]) | set(
                proposed[labels != proposed]
            )
            if swap and len(changed_labels) > 2:
                continue
            assert energy(proposed) >= energy(labels) - 1e-10


def test_gsd_uses_absolute_units_and_gdal_average(tmp_path):
    path = str(tmp_path / "values.tif")
    pixels = np.arange(64, dtype="float32").reshape(8, 8)
    create_dummy_raster(
        path,
        band_data=np.array([pixels, pixels * 10]),
        nodata=None,
        crs="EPSG:32604",
        transform=(0, 0.5, 0, 4, 0, -0.5),
    )
    sampler = module._RasterSampler("EPSG:32604")
    try:
        fine = sampler.sample(path, np.array([[0.5, 3.5]]), (2, 1), 0.5)
        coarse = sampler.sample(path, np.array([[0.5, 3.5]]), (2, 1), 1)
        assert fine[0].tolist() == [90, 9]
        assert coarse[0].tolist() == [45, 4.5]
        assert sampler.datasets[(path, 1)].RasterXSize == 4
        assert sampler.datasets[(path, 1)].GetDriver().ShortName == "VRT"
        assert sampler.datasets[(path, 1)].GetGeoTransform()[1] == 1
        assert len(sampler.tiles) == 4
    finally:
        sampler.close()


def test_direct_external_costs_route_seam_and_validate_nodata(tmp_path):
    source = _inputs(tmp_path)
    cost_path = str(tmp_path / "costs.tif")
    costs = np.full((16, 24), 100, dtype="float32")
    costs[:, 11:13] = 1
    create_dummy_raster(
        cost_path,
        band_data=costs,
        nodata=None,
        crs="EPSG:32604",
        transform=(0, 1, 0, 16, 0, -1),
    )
    variable = [[cost_path, 2, [1], [1]]]
    result = _assert_partition(
        source,
        _run(
            source,
            str(tmp_path / "cost.gpkg"),
            mesh_spacing=1,
            edge_variables_weights_gsds_bands=variable,
        ),
    ).set_index("image")
    seam = result.loc["image0"].geometry.intersection(result.loc["image1"].geometry)
    assert seam.length > 0 and 11 <= seam.bounds[0] <= seam.bounds[2] <= 13
    cost, bound, _ = module._edge_costs(
        [{}, {}],
        {},
        [LineString([(11.1, 8), (11.9, 8)])],
        module._variables(variable),
        "EPSG:32604",
    )
    assert cost(np.array([0]), np.array([1])).tolist() == [2]
    assert bound == 2  # A constant raster's Laplacian would be zero.
    with gdal.Open(cost_path, gdal.GA_Update) as dataset:
        dataset.GetRasterBand(1).Fill(-1)
    with pytest.raises(ValueError, match="nonnegative"):
        _run(
            source,
            str(tmp_path / "negative.gpkg"),
            edge_variables_weights_gsds_bands=variable,
        )
    with gdal.Open(cost_path, gdal.GA_Update) as dataset:
        dataset.GetRasterBand(1).Fill(-99)
        dataset.GetRasterBand(1).SetNoDataValue(-99)
    with pytest.raises(ValueError, match="finite"):
        _run(
            source,
            str(tmp_path / "nodata.gpkg"),
            edge_variables_weights_gsds_bands=variable,
        )


@pytest.mark.parametrize("method", ["weighted", "voronoi"])
def test_existing_seamline_methods_match_substrings(tmp_path, method):
    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    paths = frame.image_path.tolist()
    frame.image = ["folder/image0.tif", "prefix-image1-suffix"]
    frame["quality"] = [1, 2]
    frame.to_file(source, layer="footprints")
    kwargs = {"rank_function": "{quality}"} if method == "weighted" else {}
    output = getattr(Seamline, method)(
        source,
        str(tmp_path / "output.gpkg"),
        input_images=paths,
        input_layer="footprints",
        **kwargs,
    )
    assert set(gpd.read_file(output).image) == {"image0", "image1"}
    frame.iloc[:1].to_file(source, layer="footprints")
    with pytest.raises(ValueError, match="No footprint"):
        getattr(Seamline, method)(
            source, str(tmp_path / "missing.gpkg"), input_images=paths, **kwargs
        )


def test_scales_are_independent_of_mesh_and_shared_between_terms(tmp_path, monkeypatch):
    source = _inputs(tmp_path)
    actual = module._laplacian
    calls = []

    def capture(candidate, paths, coordinates, bands, gsd, sampler):
        calls.append((next(iter(candidate)), gsd))
        return actual(candidate, paths, coordinates, bands, gsd, sampler)

    monkeypatch.setattr(module, "_laplacian", capture)
    variables = [
        ["laplacian_difference", 1, {"largest": [1, 3]}, [1]],
        ["laplacian_magnitude", 0.25, {"largest": [1, 3]}, [1]],
    ]
    for spacing in [2, 7]:
        calls.clear()
        _run(
            source,
            str(tmp_path / f"mesh{spacing}.gpkg"),
            mesh_spacing=spacing,
            edge_variables_weights_gsds_bands=variables,
        )
        assert sorted(calls) == [
            ("image0", 1),
            ("image0", 3),
            ("image1", 1),
            ("image1", 3),
        ]


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_quality_rejected(tmp_path, score):
    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    frame["quality"] = [1, score]
    frame.to_file(source, layer="footprints")
    with pytest.raises(ValueError, match="finite numeric"):
        _run(source, str(tmp_path / "quality.gpkg"), image_rank_function="{quality}")


def test_quality_foreground_hierarchy_preserves_original_scores(tmp_path):
    source = _inputs(tmp_path, offsets=(0, 10, 20, 30))
    frame = gpd.read_file(source)
    frame["quality"] = [1, 2, 3, 4]
    frame.to_file(source, layer="footprints")
    output = _run(
        source,
        str(tmp_path / "quality.gpkg"),
        edge_variables_weights_gsds_bands=[],
        image_rank_function="{quality}",
        quadtree_max_images_per_leaf=2,
        mesh_spacing=2,
    )
    result = _assert_partition(source, output).set_index("image")
    assert result.quality_score.to_dict() == dict(zip(frame.image, frame.quality))
    assert result.loc["image3"].geometry.area == 256


@pytest.mark.parametrize("gsd", [1.0, 0.5, -1, 0, True, None, "auto", "Native", "1"])
def test_gsd_requires_positive_integer_or_native(gsd):
    with pytest.raises(ValueError, match="positive integers or native"):
        module._variables([["laplacian_difference", 1, [gsd], [1]]])


def test_native_defaults_and_mixed_image_resolutions(tmp_path, monkeypatch):
    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    for index, (path, resolution) in enumerate(zip(frame.image_path, [0.5, 2])):
        create_dummy_raster(
            path,
            width=int(16 / resolution),
            height=int(16 / resolution),
            count=1,
            crs="EPSG:32604",
            transform=(index * 8, resolution, 0, 16, 0, -resolution),
        )
    default = (
        inspect.signature(markov_triangles)
        .parameters["edge_variables_weights_gsds_bands"]
        .default
    )
    assert default == [
        ["laplacian_difference", 1.0, ["native"], [1]],
        ["laplacian_magnitude", 0.25, [100], [1]],
    ]
    _, _, terms = module._sources(frame, "image", default)
    assert [term[3] for term in terms] == [(0.5,), (100,)]
    actual, calls = module._laplacian, []

    def capture(candidate, paths, coordinates, bands, gsd, sampler):
        calls.append((next(iter(candidate)), gsd, bands))
        return actual(candidate, paths, coordinates, bands, gsd, sampler)

    monkeypatch.setattr(module, "_laplacian", capture)
    result = _run(source, str(tmp_path / "native.gpkg"), mesh_spacing=4)
    _assert_partition(source, result)
    assert calls == [
        ("image0", 0.5, (1,)),
        ("image1", 0.5, (1,)),
        ("image0", 100, (1,)),
        ("image1", 100, (1,)),
    ]
    assert default[0][2] == ["native"]


def test_native_external_resolution_and_reprojection(tmp_path):
    source = _inputs(tmp_path)
    frame = gpd.read_file(source)
    path = str(tmp_path / "cost.tif")
    create_dummy_raster(
        path,
        width=10,
        height=8,
        count=1,
        nodata=None,
        crs="EPSG:32604",
        transform=(0, 2.5, 0, 20, 0, -2.5),
    )
    variable = [path, 1, {"average": ["native", 4]}, [1]]
    _, _, terms = module._sources(frame, "image", [variable])
    assert terms[0][3] == (2.5, 4)
    result = _run(
        source,
        str(tmp_path / "external.gpkg"),
        edge_variables_weights_gsds_bands=[variable],
    )
    _assert_partition(source, result)
    create_dummy_raster(
        path,
        count=1,
        nodata=None,
        crs="EPSG:4326",
        transform=(-159, 0.0001, 0, 21, 0, -0.0001),
    )
    _, _, terms = module._sources(frame, "image", [variable])
    with gdal.Open(path) as dataset:
        warped = gdal.AutoCreateWarpedVRT(dataset, None, frame.crs.to_wkt())
        expected = abs(warped.GetGeoTransform()[1])
    assert terms[0][3] == (expected, 4)
    assert 5 < expected < 20  # Metres in the working CRS, not degrees.


def test_native_alias_reuses_matching_explicit_resolution(tmp_path, monkeypatch):
    source = _inputs(tmp_path)
    actual, calls = module._laplacian, []

    def capture(candidate, paths, coordinates, bands, gsd, sampler):
        calls.append((next(iter(candidate)), gsd))
        return actual(candidate, paths, coordinates, bands, gsd, sampler)

    monkeypatch.setattr(module, "_laplacian", capture)
    _run(
        source,
        str(tmp_path / "aliases.gpkg"),
        edge_variables_weights_gsds_bands=[
            ["laplacian_difference", 1, ["native"], [1]],
            ["laplacian_magnitude", 0.25, [1], [1]],
            ["missing.tif", 0, ["native"], [1]],
        ],
    )
    assert calls == [("image0", 1), ("image1", 1)]


@pytest.mark.parametrize("external", [False, True])
def test_overlapping_bands_and_scale_sets_read_once_beyond_cache(
    tmp_path, monkeypatch, external
):
    width = 70 * 256
    paths, candidates = {}, []
    for i in range(1 if external else 2):
        path = str(tmp_path / f"source{i}.tif")
        values = np.indices((8, width)).sum(axis=0).astype("float32") % 13 + 1 + i
        create_dummy_raster(
            path,
            band_data=np.array([values, values * 3, values * 7]),
            nodata=None,
            crs="EPSG:32604",
            transform=(0, 1, 0, 8, 0, -1),
        )
        paths[str(i)] = path
        candidates.append({str(i): box(0, 0, width, 8)})
    if external:
        candidates = [{}, {}]
    segments = [LineString([(i * 256 + 20, 4), (i * 256 + 26, 4)]) for i in range(70)]
    terms = module._variables(
        [
            [
                paths["0"] if external else "laplacian_difference",
                1,
                [2],
                {"largest": [2, 1]},
            ],
            [
                paths["0"] if external else "laplacian_magnitude",
                0.5,
                {"largest": [1, 2]},
                {"smallest": [3, 2]},
            ],
        ]
    )
    reads = Counter()
    original = gdal.Band.ReadAsArray

    def read(band, *args, **kwargs):
        dataset = band.GetDataset()
        if dataset is not None:
            key = (
                tuple(dataset.GetFileList()),
                dataset.GetGeoTransform()[1],
                band.GetBand(),
                args,
            )
            reads[key] += 1
        return original(band, *args, **kwargs)

    monkeypatch.setattr(gdal.Band, "ReadAsArray", read)
    combined, _, _ = module._edge_costs(
        candidates, paths, segments, terms, "EPSG:32604"
    )
    assert len(reads) > 64
    assert max(reads.values()) == 1
    monkeypatch.setattr(gdal.Band, "ReadAsArray", original)
    left = np.zeros(len(segments), dtype=int)
    right = np.arange(len(segments)) % 2
    separate = [
        module._edge_costs(candidates, paths, segments, [term], "EPSG:32604")[0](
            left, right
        )
        for term in terms
    ]
    assert combined(left, right) == pytest.approx(sum(separate))


def test_band_reuse_preserves_valid_boundary_values(monkeypatch):
    class Sampler:
        def sample(self, path, coordinates, bands, gsd, native_fallback=False):
            x, y = coordinates.T
            data = {1: x + y, 2: np.where(x == 0, np.nan, x * 2 + y)}
            return np.column_stack([data[band] for band in bands])

    candidate = {"image": box(0, 0, 8, 8)}
    points = np.array([[0.0, 4.0], [1.0, 3.0]])
    alone = module._laplacian(
        candidate, {"image": "unused"}, points, (1,), 1, Sampler()
    )
    combined = module._laplacian(
        candidate, {"image": "unused"}, points, (1, 2), 1, Sampler()
    )
    assert np.array_equal(alone[:, 0], combined[:, 0])


def test_native_rotated_pixel_size():
    dataset = gdal.GetDriverByName("MEM").Create("", 8, 8, 1)
    dataset.SetProjection(CRS.from_epsg(32604).to_wkt())
    dataset.SetGeoTransform((0, 2, 1, 30, 1, -2))
    assert module._native_resolution(dataset, CRS.from_epsg(32604)) == pytest.approx(
        math.sqrt(5)
    )


@pytest.mark.parametrize("reducer", ["average", "largest", "smallest"])
@pytest.mark.parametrize(
    "position,kind,value", [(2, "GSDs", "native"), (3, "bands", 1)]
)
def test_single_value_cannot_use_reducer(tmp_path, reducer, position, kind, value):
    term = ["laplacian_difference", 1, ["native"], [1]]
    term[position] = {reducer: [value]}
    with pytest.raises(
        ValueError, match=f"{kind} aggregation requires at least two values"
    ):
        markov_triangles(
            "unused.tif",
            str(tmp_path / "output.gpkg"),
            edge_variables_weights_gsds_bands=[term],
        )


@pytest.mark.parametrize("position,kind", [(2, "GSDs"), (3, "bands")])
def test_multiple_values_require_reducer(position, kind):
    term = ["laplacian_difference", 1, ["native"], [1]]
    term[position] = [1, 2]
    with pytest.raises(ValueError, match=f"Multiple {kind} require"):
        module._variables([term])


def test_single_selection_is_not_aggregated(monkeypatch):
    for name in ["mean", "min", "max"]:
        monkeypatch.setattr(
            module.np,
            name,
            lambda *args, **kwargs: pytest.fail("Single selections must not aggregate"),
        )
    term = module._variables([["laplacian_difference", 1, ["native"], [1]]])[0]
    assert term[2] is None and term[4] is None
    assert module._reduce(np.array([[7, 9]]), term[2]).tolist() == [7, 9]
    assert module._reduce(np.array([[7], [9]]), term[4], axis=1).tolist() == [7, 9]
