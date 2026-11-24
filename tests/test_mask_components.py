"""Mask components produce independent, cropped rasters with stable names."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal, osr
from shapely.geometry import MultiPolygon, Polygon, box

from spectralmatch import mask_rasters


def raster(path):
    ds = gdal.GetDriverByName('GTiff').Create(str(path), 20, 20, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((250000, 1, 0, 2215820, 0, -1))
    srs = osr.SpatialReference(); srs.ImportFromEPSG(32605)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.GetRasterBand(1).WriteArray(np.full((20, 20), 7, dtype='uint8'))
    ds = None
    return str(path)


def geometry():
    ring = Polygon(box(250002, 2215802, 250008, 2215808).exterior.coords,
                   [box(250004, 2215804, 250006, 2215806).exterior.coords])
    return MultiPolygon([box(250012, 2215812, 250018, 2215818), ring])


def mask(path, geometries, *, crs=32605, names=None):
    frame = gpd.GeoDataFrame({'image': names or ['scene'] * len(geometries)}, geometry=geometries, crs=32605)
    frame.to_crs(crs).to_file(path, layer='mask', driver='GPKG')
    return str(path)


def valid_pixels(path):
    with gdal.Open(str(path)) as ds:
        data = ds.ReadAsArray()
        return data.shape, np.count_nonzero(data), ds.GetGeoTransform()


@pytest.mark.parametrize('crs', [32605, 4326])
def test_split_preserves_pixels_holes_and_numbering(tmp_path, crs):
    source = raster(tmp_path / 'scene_coverage.tif')
    vector = mask(tmp_path / 'mask.gpkg', [geometry()], crs=crs)
    base = str(tmp_path / 'result.tif')
    kwargs = dict(vector_mask=('include', vector, 'image'))
    ordinary = mask_rasters([source], [base], **kwargs)
    with ThreadPoolExecutor(2) as executor:
        outputs = mask_rasters([source], [base], seperate_multipolygons=True, executor=executor,
                               validity_writing_postfix='_writing', **kwargs)
    assert ordinary == [base]
    assert outputs == [str(tmp_path / f'result_{i}.tif') for i in (1, 2)]
    assert [valid_pixels(p)[1] for p in outputs] == [32, 36]
    assert sum(valid_pixels(p)[1] for p in outputs) == valid_pixels(base)[1] == 68
    assert valid_pixels(outputs[0])[0] == (6, 6)
    assert valid_pixels(outputs[1])[0] == (6, 6)
    assert not list(tmp_path.glob('*_writing*'))


def test_split_filters_by_source_and_dissolves_touching_features(tmp_path):
    sources = [raster(tmp_path / f'{name}_coverage.tif') for name in ('1111', '2222')]
    geoms = [geometry(), box(250018, 2215812, 250022, 2215818), box(250001, 2215811, 250004, 2215814), box(250030, 2215830, 250040, 2215840)]
    vector = mask(tmp_path / 'mask.gpkg', geoms, names=['1111', '1111', '2222', '1111'])
    outputs = mask_rasters(sources, str(tmp_path / '$.masked.tif'),
                           vector_mask=('include', vector, 'image'), seperate_multipolygons=True)
    assert [Path(p).name for p in outputs] == ['1111_coverage.masked_1.tif', '1111_coverage.masked_2.tif', '2222_coverage.masked_1.tif']
    assert [valid_pixels(p)[1] for p in outputs] == [32, 48, 9]
    assert not list(tmp_path.glob('*.masked.tif'))


def test_exclude_splits_remaining_area(tmp_path):
    source = raster(tmp_path / 'scene.tif')
    vector = mask(tmp_path / 'mask.gpkg', [box(250008, 2215799, 250012, 2215821)])
    outputs = mask_rasters([source], [str(tmp_path / 'out.tif')], vector_mask=('exclude', vector),
                           seperate_multipolygons=True, custom_nodata_value=99)
    assert [valid_pixels(p)[0] for p in outputs] == [(20, 8), (20, 8)]
    for p in outputs:
        with gdal.Open(p) as ds:
            assert ds.GetRasterBand(1).GetNoDataValue() == 99
            assert (ds.ReadAsArray() == 7).all()


def test_resume_reuses_each_numbered_output(tmp_path):
    source = raster(tmp_path / 'scene.tif')
    vector = mask(tmp_path / 'mask.gpkg', [geometry()])
    kwargs = dict(vector_mask=('include', vector), seperate_multipolygons=True, resume_from_outputs='validate')
    outputs = mask_rasters([source], [str(tmp_path / 'out.tif')], **kwargs)
    first_mtime = Path(outputs[0]).stat().st_mtime_ns
    Path(outputs[1]).unlink()
    assert mask_rasters([source], [str(tmp_path / 'out.tif')], **kwargs) == outputs
    assert Path(outputs[0]).stat().st_mtime_ns == first_mtime
    assert valid_pixels(outputs[1])[1] == 36


def test_empty_intersection_has_no_outputs(tmp_path):
    source = raster(tmp_path / 'scene.tif')
    vector = mask(tmp_path / 'mask.gpkg', [box(250030, 2215830, 250040, 2215840)])
    outputs = mask_rasters([source], [str(tmp_path / 'out.tif')], vector_mask=('include', vector), seperate_multipolygons=True)
    assert outputs == []
    assert not list(tmp_path.glob('out*.tif'))


@pytest.mark.parametrize('vector_mode', ['none', 'no_matching_image'])
def test_unmasked_input_gets_one_numbered_file(tmp_path, vector_mode):
    source = raster(tmp_path / 'scene.tif')
    vector = mask(tmp_path / 'mask.gpkg', [geometry()], names=['other'])
    outputs = mask_rasters([source], str(tmp_path / 'outputs'),
                          vector_mask=None if vector_mode == 'none' else ('include', vector, 'image'),
                          seperate_multipolygons=True)
    assert [Path(p).name for p in outputs] == ['scene_Mask_1.tif']
    assert valid_pixels(outputs[0])[:2] == ((20, 20), 400)


@pytest.mark.parametrize('value', [1, 'true', None])
def test_split_requires_boolean(tmp_path, value):
    source = raster(tmp_path / 'scene.tif')
    with pytest.raises(ValueError, match='seperate_multipolygons must be a boolean'):
        mask_rasters([source], [str(tmp_path / 'out.tif')], seperate_multipolygons=value)


def test_numbered_outputs_cannot_overwrite_inputs(tmp_path):
    source = raster(tmp_path / 'out_1.tif')
    with pytest.raises(ValueError, match='must differ from inputs'):
        mask_rasters([source], [str(tmp_path / 'out.tif')], seperate_multipolygons=True)
    assert valid_pixels(source)[1] == 400


def test_pipeline_returns_all_numbered_outputs(tmp_path):
    from spectralmatch import pipeline

    source = raster(tmp_path / 'scene.tif')
    vector = mask(tmp_path / 'mask.gpkg', [geometry()])
    result = pipeline(
        shared_input_images=[source], shared_output_image_path=str(tmp_path / 'outputs'),
        shared_temp_dir=str(tmp_path / 'temp'), steps=('mask',),
        mask_rasters_vector_mask=('include', vector), mask_rasters_seperate_multipolygons=True,
        shared_image_threads=None, shared_io_threads=1, shared_tile_threads=1,
    )
    assert len(result['output']) == 2
    assert result['output'] == result['mask_rasters']
    assert [valid_pixels(p)[1] for p in result['output']] == [32, 36]
