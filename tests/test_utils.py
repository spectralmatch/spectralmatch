import pytest
import os
import geopandas as gpd
from osgeo import gdal

from .utils_test import create_dummy_raster, create_dummy_vector
from spectralmatch import merge_rasters, merge_vectors, align_rasters, mask_rasters


@pytest.fixture
def basic_raster_set(tmp_path):
    """
    Creates 3 dummy rasters in an input folder and returns their paths + output folder.
    Used for merging, masking, aligning, etc.

    Returns:
        Tuple[List[str], str]: List of input raster paths, and output folder path.
    """
    input_dir = os.path.join(tmp_path, "input")
    output_dir = os.path.join(tmp_path, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    raster_paths = []
    for i, value in enumerate([50, 100, 150]):
        raster_path = os.path.join(input_dir, f"raster_{i}.tif")
        create_dummy_raster(raster_path, width=32, height=32, count=1, fill_value=value)
        raster_paths.append(raster_path)

    return raster_paths, output_dir


@pytest.fixture
def basic_vector_set(tmp_path):
    """
    Creates 2 dummy vector files for testing vector merging.

    Returns:
        Tuple[List[str], str]: List of vector file paths and output directory.
    """
    input_dir = os.path.join(tmp_path, "input")
    output_dir = os.path.join(tmp_path, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    vector_paths = []
    for i in range(2):
        vector_path = os.path.join(input_dir, f"vector_{i}.gpkg")
        create_dummy_vector(vector_path, bounds=(i, i, i + 5, i + 5))
        vector_paths.append(vector_path)

    return vector_paths, output_dir


@pytest.fixture
def raster_and_vector_mask(tmp_path):
    """
    Creates one dummy raster and one vector mask covering part of the raster.

    Returns:
        Tuple[List[str], List[str], Tuple]: (input raster paths, output raster paths, vector mask tuple)
    """
    input_dir = os.path.join(tmp_path, "input")
    output_dir = os.path.join(tmp_path, "output")
    vector_path = os.path.join(tmp_path, "mask.gpkg")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_path = os.path.join(input_dir, "masked_input.tif")
    output_path = os.path.join(output_dir, "masked_output.tif")

    create_dummy_raster(input_path, width=32, height=32, count=1, fill_value=100)
    create_dummy_vector(vector_path, bounds=(8, 8, 24, 24))

    return [input_path], [output_path], ("exclude", vector_path)


@pytest.fixture
def misaligned_raster_set(tmp_path):
    """
    Creates two dummy rasters with different resolutions for alignment testing.

    Returns:
        Tuple[List[str], List[str]]: List of input raster paths, list of output raster paths.
    """
    input_dir = os.path.join(tmp_path, "input")
    output_dir = os.path.join(tmp_path, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_paths = []
    output_paths = []

    # Raster 1: 1x1 meter
    path1 = os.path.join(input_dir, "raster1.tif")
    create_dummy_raster(path1, width=16, height=16, transform=(0, 1, 0, 16, 0, -1))
    input_paths.append(path1)
    output_paths.append(os.path.join(output_dir, "aligned1.tif"))

    # Raster 2: 2x2 meter
    path2 = os.path.join(input_dir, "raster2.tif")
    create_dummy_raster(path2, width=8, height=8, transform=(0, 2, 0, 16, 0, -2))
    input_paths.append(path2)
    output_paths.append(os.path.join(output_dir, "aligned2.tif"))

    return input_paths, output_paths


# merge_rasters
def test_merge_rasters_minimal(basic_raster_set):
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {
                "io_threads": 2,
                "tile_threads": 2,
                "debug_logs": True,
                "output_dtype": "uint16",
                "custom_nodata_value": 9999,
            },
        ],
    )
    def test_merge_rasters_parametrized(basic_raster_set, kwargs):
        input_rasters, output_dir = basic_raster_set
        name = "minimal" if not kwargs else "all"
        output_path = os.path.join(output_dir, f"merged_{name}.tif")

        merge_rasters(
            input_images=input_rasters,
            output_image_path=output_path,
            **kwargs,
        )

        assert os.path.exists(output_path)


# mask_rasters
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "debug_logs": True,
            "window_size": 16,
            "io_threads": 2,
            "tile_threads": 2,
            "include_touched_pixels": True,
            "custom_nodata_value": 99,
        },
    ],
)
def test_mask_rasters_parametrized(raster_and_vector_mask, kwargs):
    input_rasters, output_rasters, vector_mask = raster_and_vector_mask

    mask_rasters(
        input_images=input_rasters,
        output_images=output_rasters,
        vector_mask=vector_mask,
        **kwargs,
    )

    assert os.path.exists(output_rasters[0])

    ds = gdal.Open(output_rasters[0])
    expected_nodata = kwargs.get("custom_nodata_value", 0)
    assert ds.GetRasterBand(1).GetNoDataValue() == expected_nodata
    data = ds.GetRasterBand(1).ReadAsArray()
    assert data.shape == (32, 32)
    ds = None


# merge_vectors
def test_merge_vectors_union_minimal(basic_vector_set):
    input_vectors, output_dir = basic_vector_set
    output_path = os.path.join(output_dir, "merged_union.gpkg")

    merge_vectors(
        input_vectors=input_vectors, merged_vector_path=output_path, method="union"
    )

    assert os.path.exists(output_path)
    merged = gpd.read_file(output_path)
    assert not merged.empty


def test_merge_vectors_keep_with_names(basic_vector_set):
    input_vectors, output_dir = basic_vector_set
    output_path = os.path.join(output_dir, "merged_keep.gpkg")

    merge_vectors(
        input_vectors=input_vectors,
        merged_vector_path=output_path,
        method="keep",
        debug_logs=True,
        create_name_attribute=("src", "_"),
    )

    assert os.path.exists(output_path)
    merged = gpd.read_file(output_path)
    assert "src" in merged.columns
    assert len(merged) == 2


# align_rasters
def test_align_rasters_minimal(misaligned_raster_set):
    input_paths, output_paths = misaligned_raster_set

    align_rasters(input_images=input_paths, output_images=output_paths)

    for out_path, expected_resolution in zip(output_paths, (1.0, 2.0)):
        assert os.path.exists(out_path)
        ds = gdal.Open(out_path)
        assert ds.RasterXSize > 0 and ds.RasterYSize > 0
        assert abs(ds.GetGeoTransform()[1]) == pytest.approx(expected_resolution)
        assert abs(ds.GetGeoTransform()[5]) == pytest.approx(expected_resolution)
        ds = None


def test_align_rasters_all_options(misaligned_raster_set):
    input_paths, output_paths = misaligned_raster_set

    align_rasters(
        input_images=input_paths,
        output_images=output_paths,
        resampling_method="nearest",
        tap=True,
        resolution="average",
        debug_logs=True,
        window_size=16,
        io_threads=2,
        image_threads=2,
        )

    for out_path in output_paths:
        assert os.path.exists(out_path)
        ds = gdal.Open(out_path)
        assert ds.RasterXSize > 0 and ds.RasterYSize > 0
        ds = None


@pytest.mark.parametrize("resolution", [2, 1.5])
def test_align_rasters_accepts_numeric_resolution(misaligned_raster_set, resolution):
    input_paths, output_paths = misaligned_raster_set

    align_rasters(
        input_images=input_paths,
        output_images=output_paths,
        resolution=resolution,
        tap=True,
    )

    for out_path in output_paths:
        transform = gdal.Open(out_path).GetGeoTransform()
        assert abs(transform[1]) == pytest.approx(resolution)
        assert abs(transform[5]) == pytest.approx(resolution)
        assert transform[0] / resolution == pytest.approx(round(transform[0] / resolution))
        assert transform[3] / resolution == pytest.approx(round(transform[3] / resolution))
