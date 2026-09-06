import pytest
import os
import numpy as np
from osgeo import gdal, osr

from spectralmatch import (
    compare_before_after_all_images,
    compare_image_spectral_profiles_pairs,
    compare_spatial_spectral_difference_band_average,
)
from .test_utils import create_dummy_raster
from spectralmatch.statistics import _projected_mean_spectral_difference


@pytest.fixture
def spectral_test_rasters(tmp_path):
    """
    Creates two dummy rasters in an input directory and an empty output directory.

    Returns:
        Tuple[dict, str]: Dictionary with labels as keys and raster paths as values, and the output directory path.
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    image_dict = {}
    for label, value in [("Image A", 80), ("Image B", 160)]:
        raster_path = input_dir / f"{label.replace(' ', '_')}.tif"
        create_dummy_raster(raster_path, width=16, height=16, count=5, fill_value=value)
        image_dict[label] = str(raster_path)

    return image_dict, str(output_dir)


# compare_image_spectral_profiles
def test_compare_image_spectral_profiles_pairs(spectral_test_rasters):
    image_dict, output_dir = spectral_test_rasters
    pair_dict = {}

    for label, before_path in image_dict.items():
        after_filename = os.path.basename(before_path).replace(".tif", "_after.tif")
        after_path = os.path.join(os.path.dirname(before_path), after_filename)
        create_dummy_raster(after_path, width=16, height=16, count=5, fill_value=200)
        pair_dict[label] = [before_path, after_path]

    output_path = os.path.join(output_dir, "paired_profiles.png")

    compare_image_spectral_profiles_pairs(
        image_groups_dict=pair_dict,
        output_figure_path=output_path,
        title="Before vs After Comparison",
        xlabel="Band",
        ylabel="Mean Value",
    )

    assert os.path.exists(output_path)


# compare_spatial_spectral_difference_band_average
def test_compare_spatial_spectral_difference_band_average(spectral_test_rasters):
    image_dict, output_dir = spectral_test_rasters
    images = list(image_dict.values())
    assert len(images) >= 2, "Need at least two images for difference comparison"

    output_path = os.path.join(output_dir, "spatial_diff.png")

    compare_spatial_spectral_difference_band_average(
        input_images=[images[0], images[1]],
        output_figure_path=output_path,
        title="Difference Map",
        diff_label="Mean Band Abs Diff",
        subtitle="Test difference between A and B",
    )

    assert os.path.exists(output_path)


def test_compare_before_after_all_images(spectral_test_rasters):
    image_dict, output_dir = spectral_test_rasters
    input_images_1 = list(image_dict.values())
    input_images_2 = []

    for before_path in input_images_1:
        after_filename = os.path.basename(before_path).replace(".tif", "_after.tif")
        after_path = os.path.join(os.path.dirname(before_path), after_filename)
        create_dummy_raster(after_path, width=16, height=16, count=3, fill_value=180)
        input_images_2.append(after_path)

    image_names = [os.path.splitext(os.path.basename(p))[0] for p in input_images_1]
    output_path = os.path.join(output_dir, "compare_before_after_all_images.png")

    compare_before_after_all_images(
        input_images_1=input_images_1,
        input_images_2=input_images_2,
        image_names=image_names,
        output_figure_path=output_path,
        title="Before vs After Grid",
        ylabel_1="Original",
        ylabel_2="Processed",
    )

    assert os.path.exists(output_path)


@pytest.mark.parametrize("after_width", [5, 6])
@pytest.mark.parametrize("rotated", [False, True])
def test_spatial_difference_compares_locations_instead_of_array_indices(tmp_path, after_width, rotated):
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    before = np.tile(np.arange(6, dtype=np.float32) + 10, (4, 1))
    after = np.tile(np.arange(after_width, dtype=np.float32) + 19, (4, 1))
    transform = (100, 1, 0.2 if rotated else 0, 200, 0.1 if rotated else 0, -1)
    x, y = gdal.ApplyGeoTransform(transform, 2, 0)
    after_transform = (x, transform[1], transform[2], y, transform[4], transform[5])
    create_dummy_raster(before_path, band_data=before, transform=transform, crs="EPSG:3857")
    create_dummy_raster(after_path, band_data=after, transform=after_transform, crs="EPSG:3857")
    diff = _projected_mean_spectral_difference(gdal.Open(str(before_path)), gdal.Open(str(after_path)), "nearest")
    assert diff.shape == (4, 6)
    assert np.isnan(diff[:, :2]).all()
    np.testing.assert_allclose(diff[:, 2:], 7)


def test_spatial_difference_resamples_different_resolutions(tmp_path):
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    before = 100 + np.arange(8)[None, :] + 0.5 + 2 * (7.5 - np.arange(8)[:, None])
    after = 105.0 + (2 * np.arange(4)[None, :] + 1) + 2 * (7 - 2 * np.arange(4)[:, None])
    create_dummy_raster(before_path, band_data=before, transform=(0, 1, 0, 8, 0, -1))
    create_dummy_raster(after_path, band_data=after, transform=(0, 2, 0, 8, 0, -2))
    first, second = gdal.Open(str(before_path)), gdal.Open(str(after_path))
    diff = _projected_mean_spectral_difference(first, second, "bilinear")
    np.testing.assert_allclose(diff[1:-1, 1:-1], 5)
    nearest = _projected_mean_spectral_difference(first, second, "nearest")
    assert np.any(nearest[1:-1, 1:-1] != diff[1:-1, 1:-1])


def test_spatial_difference_reprojects_different_crs(tmp_path):
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    geographic, projected = osr.SpatialReference(), osr.SpatialReference()
    geographic.ImportFromEPSG(4326)
    geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    projected.ImportFromEPSG(3857)
    x, y, _ = osr.CoordinateTransformation(geographic, projected).TransformPoint(0.04, 0.04)
    create_dummy_raster(before_path, width=4, height=4, count=1, fill_value=100, transform=(0, 0.01, 0, 0.04, 0, -0.01), crs="EPSG:4326")
    create_dummy_raster(after_path, width=6, height=6, count=1, fill_value=112, transform=(0, x / 6, 0, y, 0, -y / 6), crs="EPSG:3857")
    diff = _projected_mean_spectral_difference(gdal.Open(str(before_path)), gdal.Open(str(after_path)), "bilinear")
    np.testing.assert_allclose(diff, 12)


def test_spatial_difference_masks_each_images_nodata_and_mask(tmp_path):
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    before = np.stack([np.full((4, 6), 100.0), np.full((4, 6), 200.0)])
    after = np.stack([np.full((4, 6), 110.0), np.full((4, 6), 230.0)])
    before[0, 0, 0] = -99
    after[1, 0, 1] = -999
    before[1, 0, 2] = np.nan
    after[0, 0, 3] = np.nan
    create_dummy_raster(before_path, band_data=before, nodata=-99)
    create_dummy_raster(after_path, band_data=after, nodata=-999)
    for path, col in [(before_path, 4), (after_path, 5)]:
        dataset = gdal.Open(str(path), gdal.GA_Update)
        dataset.CreateMaskBand(gdal.GMF_PER_DATASET)
        mask = np.full((4, 6), 255, dtype=np.uint8)
        mask[0, col] = 0
        dataset.GetRasterBand(1).GetMaskBand().WriteArray(mask)
        dataset = None
    diff = _projected_mean_spectral_difference(gdal.Open(str(before_path)), gdal.Open(str(after_path)), "nearest")
    assert np.isnan(diff[0]).all()
    np.testing.assert_allclose(diff[1:], 20)


def test_spatial_difference_honors_band_specific_nodata():
    before = gdal.GetDriverByName("MEM").Create("", 3, 3, 2, gdal.GDT_Float64)
    after = gdal.GetDriverByName("MEM").Create("", 3, 3, 2, gdal.GDT_Float64)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    for dataset, values, nodata in [(before, (100, 200), (-99, -88)), (after, (110, 230), (-999, -888))]:
        dataset.SetProjection(srs.ExportToWkt())
        dataset.SetGeoTransform((0, 1, 0, 3, 0, -1))
        for index, (value, missing) in enumerate(zip(values, nodata), 1):
            data = np.full((3, 3), value, dtype=np.float64)
            data[index - 1, 0 if dataset is before else 1] = missing
            band = dataset.GetRasterBand(index)
            band.SetNoDataValue(missing)
            band.WriteArray(data)
    diff = _projected_mean_spectral_difference(before, after, "nearest")
    assert np.isnan(diff[:2, :2]).all()
    np.testing.assert_allclose(diff[:, 2], 20)
    np.testing.assert_allclose(diff[2], 20)


@pytest.mark.parametrize("all_nodata", [False, True])
def test_spatial_difference_rejects_no_shared_valid_coverage(tmp_path, all_nodata):
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    create_dummy_raster(before_path, count=1)
    create_dummy_raster(after_path, count=1, fill_value=0 if all_nodata else 100, transform=(0 if all_nodata else 100, 1, 0, 10, 0, -1))
    with pytest.raises(ValueError, match="no overlapping pixels valid"):
        _projected_mean_spectral_difference(gdal.Open(str(before_path)), gdal.Open(str(after_path)), "nearest")


def test_spatial_difference_rejects_mismatched_bands_and_missing_georeferencing(tmp_path):
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    create_dummy_raster(before_path, count=1)
    create_dummy_raster(after_path, count=2)
    with pytest.raises(ValueError, match="number of bands"):
        _projected_mean_spectral_difference(gdal.Open(str(before_path)), gdal.Open(str(after_path)), "nearest")
    create_dummy_raster(after_path, count=1, crs=None)
    with pytest.raises(ValueError, match="CRS and affine geotransform"):
        _projected_mean_spectral_difference(gdal.Open(str(before_path)), gdal.Open(str(after_path)), "nearest")


def test_spatial_difference_plot_supports_changed_dimensions(tmp_path):
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    create_dummy_raster(before_path, width=8, height=6, count=2, fill_value=100)
    create_dummy_raster(after_path, width=5, height=4, count=2, fill_value=110, transform=(2, 1, 0, 9, 0, -1))
    output = tmp_path / "new_folder" / "projected_difference.png"
    compare_spatial_spectral_difference_band_average([str(before_path), str(after_path)], str(output), "Projected difference", "After minus before", "Shared valid coverage", scale=(-20, 20))
    assert output.is_file() and output.stat().st_size > 0
