import numpy as np
import os
import pytest
import geopandas as gpd

from osgeo import gdal

from spectralmatch import (
    band_math,
    create_cloud_mask_with_omnicloudmask,
    process_raster_values_to_vector_polygons,
)
from .utils_test import create_dummy_raster


@pytest.fixture
def dummy_multiband_raster(tmp_path):
    path = tmp_path / "input.tif"
    create_dummy_raster(path, width=64, height=64, count=2, fill_value=10)
    return path


@pytest.fixture
def dummy_rgbn_raster(tmp_path):
    path = tmp_path / "rgbn.tif"
    create_dummy_raster(path, width=128, height=128, count=3, fill_value=100)
    return path


@pytest.fixture
def dummy_red_nir_raster(tmp_path):
    path = tmp_path / "rgbn.tif"
    create_dummy_raster(
        path,
        dtype="uint16",
        nodata=0,
        transform=(0, 1, 0, 32, 0, -1),
        band_data=[
            np.full((32, 32), 1000, dtype="uint16"),
            np.full((32, 32), 500, dtype="uint16"),
        ],
    )
    return path


@pytest.fixture
def dummy_raster_for_vector(tmp_path):
    path = tmp_path / "input.tif"
    width, height = 16, 16
    data = np.zeros((height, width), dtype="uint8")
    data[2:6, 2:6] = 1
    data[10:14, 10:14] = 2

    create_dummy_raster(
        path,
        dtype="uint8",
        nodata=0,
        transform=(0, 1, 0, 16, 0, -1),
        band_data=data,
    )
    return path


@pytest.fixture
def dummy_gradient_raster(tmp_path):
    path = tmp_path / "input.tif"
    data = np.tile(np.arange(16, dtype="uint8"), (16, 1))  # Horizontal gradient
    create_dummy_raster(
        path,
        dtype="uint8",
        nodata=0,
        transform=(0, 1, 0, 16, 0, -1),
        band_data=data,
    )

    return path


# band_math
def test_band_math_basic(dummy_multiband_raster, tmp_path):
    input_path = str(dummy_multiband_raster)
    output_path = str(tmp_path / "output.tif")
    band_math(
        input_images=[input_path], output_images=[output_path], threshold_math="B1 + B2"
    )

    ds = gdal.Open(output_path)
    result = ds.GetRasterBand(1).ReadAsArray()
    assert result.shape == (64, 64)
    assert np.all(result == 20)
    ds = None


def test_band_math_dtype(dummy_multiband_raster, tmp_path):
    input_path = str(dummy_multiband_raster)
    output_path = str(tmp_path / "typed.tif")
    band_math(
        input_images=[input_path],
        output_images=[output_path],
        threshold_math="B1 * 2",
        custom_output_dtype="uint16",
    )

    ds = gdal.Open(output_path)
    assert gdal.GetDataTypeName(ds.GetRasterBand(1).DataType) == "UInt16"
    ds = None


def test_band_math_nodata(dummy_multiband_raster, tmp_path):
    input_path = str(dummy_multiband_raster)
    output_path = str(tmp_path / "nodata.tif")
    band_math(
        input_images=[input_path],
        output_images=[output_path],
        threshold_math="B1 + B2",
        custom_nodata_value=99,
    )

    ds = gdal.Open(output_path)
    assert ds.GetRasterBand(1).GetNoDataValue() == 99
    ds = None


# create_cloud_mask_with_omnicloudmask
def test_create_cloud_mask(dummy_rgbn_raster, tmp_path):
    input_path = str(dummy_rgbn_raster)
    output_path = str(tmp_path / "cloud_mask.tif")

    create_cloud_mask_with_omnicloudmask(
        input_images=[input_path],
        output_images=[output_path],
        red_band_index=1,
        green_band_index=2,
        nir_band_index=3,
        debug_logs=True,
        omnicloud_kwargs={"patch_size": 50, "patch_overlap": 20},
    )

    assert os.path.exists(output_path)
    ds = gdal.Open(output_path)
    assert ds.GetRasterBand(1).ReadAsArray().shape == (128, 128)
    ds = None


# process_raster_values_to_vector_polygons
def test_process_raster_values_to_polygons_basic(dummy_raster_for_vector, tmp_path):
    input_path = str(dummy_raster_for_vector)
    output_path = str(tmp_path / "out.gpkg")

    process_raster_values_to_vector_polygons(
        input_images=[input_path],
        output_vectors=[output_path],
        extraction_expression="B1 > 0",
    )

    assert tmp_path.joinpath("out.gpkg").exists()
    gdf = gpd.read_file(output_path)
    assert not gdf.empty
    assert gdf.geometry.iloc[0].is_valid


def test_process_polygons_with_value_mapping_and_filter(
    dummy_raster_for_vector, tmp_path
):
    input_path = str(dummy_raster_for_vector)
    output_path = str(tmp_path / "filtered.gpkg")

    process_raster_values_to_vector_polygons(
        input_images=[input_path],
        output_vectors=[output_path],
        extraction_expression="B1 >= 2",
        filter_by_polygon_size="<50%",
        value_mapping={2: 5},
    )

    gdf = gpd.read_file(output_path)
    assert all(gdf.area > 4)


def test_process_polygons_with_buffer(dummy_raster_for_vector, tmp_path):
    input_path = str(dummy_raster_for_vector)
    output_path = str(tmp_path / "buffered.gpkg")

    process_raster_values_to_vector_polygons(
        input_images=[input_path],
        output_vectors=[output_path],
        extraction_expression="B1 == 1",
        polygon_buffer=0.5,
    )

    gdf = gpd.read_file(output_path)
    assert gdf.geometry.iloc[0].buffer(-0.5).area < gdf.geometry.iloc[0].area


def test_invalid_filter_by_polygon_size_raises(dummy_raster_for_vector, tmp_path):
    input_path = str(dummy_raster_for_vector)
    output_path = str(tmp_path / "error.gpkg")

    with pytest.raises(ValueError):
        process_raster_values_to_vector_polygons(
            input_images=[input_path],
            output_vectors=[output_path],
            extraction_expression="B1 > 0",
            filter_by_polygon_size="50%",  # Invalid
        )
