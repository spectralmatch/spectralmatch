import os
import pytest
import geopandas as gpd
from shapely.geometry import box

from spectralmatch import Seamline
from .test_utils import create_dummy_raster


# voronoi_center_seamline
@pytest.mark.parametrize("image_prefix, fill_value", [("A", 100), ("B", 120)])
def test_voronoi_center_seamline_all_params(tmp_path, image_prefix, fill_value):
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
            fill_value=fill_value if name == image_prefix else fill_value + 50,
        )
        input_paths.append(str(path))

    output_mask = str(output_dir / "seamlines.gpkg")
    debug_vectors = str(output_dir / "debug_vectors.gpkg")

    Seamline.voronoi(
        input_images=input_paths,
        output_mask=output_mask,
        image_field_name="source",
        debug_logs=True,
        debug_vectors_path=debug_vectors,
    )

    assert os.path.exists(output_mask)
    assert os.path.exists(debug_vectors)


def test_voronoi_center_seamline_minimal(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    paths = []
    for name in ["X", "Y"]:
        path = input_dir / f"{name}.tif"
        create_dummy_raster(path, 16, 16, count=1, fill_value=75 if name == "X" else 85)
        paths.append(str(path))

    out_path = str(output_dir / "seamlines.gpkg")

    Seamline.voronoi(input_images=paths, output_mask=out_path)

    assert os.path.exists(out_path)


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
