"""Seamline clipping must survive merging independently sampled raster grids."""
import numpy as np
import geopandas as gpd
import pytest
from osgeo import gdal
from shapely.geometry import box, LineString
from shapely.ops import split
from spectralmatch import pipeline
from .utils_test import create_dummy_raster


@pytest.mark.parametrize('seperate_multipolygons', [False, True])
@pytest.mark.parametrize('resolution,shift,slope', [(1., .31, 1.17), (1.07, 0., .37), (1.4, .31, 1.17)])
def test_pipeline_seamline_merge_preserves_coverage_and_source_nodata(tmp_path, resolution, shift, slope, seperate_multipolygons):
    paths = []
    for i, (offset, pixel_size) in enumerate([(0., 1.), (shift, resolution)]):
        rows, cols = np.indices((128, 128))
        x = 200000 + offset + (cols + .5) * pixel_size
        y = 2200128 + offset - (rows + .5) * pixel_size
        values = np.full((128, 128), 100 * (i + 1), dtype=np.int16)
        # Genuine missing source data must stay missing; this is not gap filling.
        values[(x > 200025) & (x < 200035) & (y > 2200025) & (y < 2200035)] = -9999
        path = str(tmp_path / f'image{i}.tif')
        create_dummy_raster(path, width=128, height=128, count=1, dtype='int16', nodata=-9999,
                            crs='EPSG:6635', transform=(200000 + offset, pixel_size, 0, 2200128 + offset, 0, -pixel_size), band_data=values)
        paths.append(path)
    extent = box(200000, 2200000, 200128, 2200128)
    seam = LineString([(199000, 2200064 - slope * 1064), (201000, 2200064 + slope * 936)])
    polygons = list(split(extent, seam).geoms)
    vector = str(tmp_path / 'seams.gpkg')
    gpd.GeoDataFrame({'image': ['image0', 'image1']}, geometry=polygons, crs=6635).to_file(vector)
    output = str(tmp_path / 'mosaic.tif')
    pipeline(paths, output, steps=('mask', 'merge'), shared_temp_dir=str(tmp_path / 'work'),
             shared_cache=None, shared_image_threads=None, shared_io_threads=1, shared_tile_threads=1,
             shared_custom_nodata_value=-9999, shared_window_size=16,
             mask_rasters_vector_mask=('include', vector, 'image'),
             mask_rasters_seperate_multipolygons=seperate_multipolygons)
    ds = gdal.Open(output); values = ds.ReadAsArray(); gt = ds.GetGeoTransform()
    rows, cols = np.indices(values.shape)
    x = gt[0] + (cols + .5) * gt[1]; y = gt[3] + (rows + .5) * gt[5]
    interior = (x > 200010) & (x < 200118) & (y > 2200010) & (y < 2200118)
    hole_margin = (x > 200022) & (x < 200038) & (y > 2200022) & (y < 2200038)
    assert np.all(values[interior & ~hole_margin] != -9999)
    hole_core = (x > 200028) & (x < 200032) & (y > 2200028) & (y < 2200032)
    assert np.all(values[hole_core] == -9999)
    assert set(np.unique(values)).issubset({-9999, 100, 200})
