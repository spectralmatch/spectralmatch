import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr
from spectralmatch import align_rasters, compute_overviews, merge_rasters
from spectralmatch.utils_compression import compression_context, current_compression, raster_creation_options
from spectralmatch.utils_multiprocessing import _run_image_tasks


def _write_probe(path):
    ds=gdal.GetDriverByName('GTiff').Create(path,64,64,1,gdal.GDT_Byte,options=raster_creation_options(['TILED=YES','COMPRESS=DEFLATE']))
    ds.GetRasterBand(1).Fill(7);ds=None
    ds=gdal.Open(path)
    return current_compression(),ds.GetMetadata('IMAGE_STRUCTURE').get('COMPRESSION','NONE')


def test_compression_context_reaches_spawn_workers_and_resets(tmp_path):
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as executor:
        for enabled in [False,True]:
            path=str(tmp_path/f'{enabled}.tif')
            with compression_context(enabled):
                result=_run_image_tasks(_write_probe,[(path,)],input_paths=[path],output_paths=[path],executor=executor)
            assert result==[(enabled,'DEFLATE' if enabled else 'NONE')]
    assert current_compression() is True


def raster(tmp_path):
    path=str(tmp_path/'input.tif');s=osr.SpatialReference();s.ImportFromEPSG(32605)
    d=gdal.GetDriverByName('GTiff').Create(path,128,128,1,gdal.GDT_Int16)
    d.SetGeoTransform((200000,1,0,2200000,0,-1));d.SetProjection(s.ExportToWkt());d.GetRasterBand(1).SetNoDataValue(-9999)
    d.GetRasterBand(1).WriteArray(np.arange(128*128,dtype='int16').reshape(128,128));d=None;return path


@pytest.mark.parametrize('enabled',[True,False])
def test_align_merge_and_overviews_obey_compression(tmp_path,enabled):
    source=raster(tmp_path);aligned=str(tmp_path/'aligned.tif');mosaic=str(tmp_path/'mosaic.tif')
    align_rasters([source],[aligned],compression=enabled)
    merge_rasters([aligned],mosaic,compression=enabled,build_overviews=False)
    compute_overviews(mosaic,window_scales=(2,4),compression=enabled)
    for path in [aligned,mosaic]:
        d=gdal.Open(path);codec=d.GetMetadata('IMAGE_STRUCTURE').get('COMPRESSION','NONE')
        assert (codec!='NONE')==enabled
        np.testing.assert_array_equal(d.ReadAsArray(),gdal.Open(source).ReadAsArray())
    import tifffile
    with tifffile.TiffFile(mosaic) as tif:
        assert len(tif.pages)>=3
        assert all((int(page.compression)!=1)==enabled for page in tif.pages)


def test_options_remove_codec_settings_and_do_not_touch_other_drivers():
    with compression_context(False):
        assert raster_creation_options(['COMPRESS=ZSTD','LEVEL=9','PREDICTOR=2','BLOCKSIZE=256'],'COG')==['BLOCKSIZE=256','COMPRESS=NONE','OVERVIEW_COMPRESS=NONE']
        assert raster_creation_options([], 'VRT')==[]
    with pytest.raises(ValueError,match='compression'):
        with compression_context('false'):pass


@pytest.mark.parametrize("cog", [False, True])
def test_global_local_compression_does_not_change_pixels(tmp_path, cog):
    from spectralmatch import global_regression, local_block_adjustment
    from .test_footprint_reads import raster as sparse_raster
    a, _, _ = sparse_raster(tmp_path, 'a')
    b, _, _ = sparse_raster(tmp_path, 'b')
    for function, options in [
        (global_regression, dict(pif_method='entire', estimate_stats=False)),
        (local_block_adjustment, dict(number_of_blocks=(8, 8), correction_method='offset')),
    ]:
        values = []
        for enabled in (True, False):
            paths = [str(tmp_path / f'{function.__name__}_{enabled}_{i}.tif') for i in (0, 1)]
            function([a, b], paths, compression=enabled, save_as_cog=cog, window_size=128, **options)
            images = []
            for path in paths:
                ds = gdal.Open(path)
                assert (ds.GetMetadata('IMAGE_STRUCTURE').get('COMPRESSION', 'NONE') != 'NONE') == enabled
                images.append(ds.ReadAsArray())
            values.append(images)
        for left, right in zip(*values):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('enabled', [True, False])
def test_pif_intermediate_compression(tmp_path, enabled):
    from .test_pif_batched import sources
    from spectralmatch.pif import pif
    paths, w, h, gt, crs = sources(tmp_path, size=(128, 128))
    with compression_context(enabled):
        valid = pif._build_valid_mask_raster(*paths, w, h, gt, crs, str(tmp_path))
        seeds = pif._build_seed_mask_raster(np.array([[70, 90]]), w, h, gt, crs, 15, str(tmp_path))
        score = str(tmp_path / 'inz.tif')
        mask, count = pif._seed_pif_mask(*paths, valid, seeds, w, h, gt, crs, 3, 1, 3, .2, .9, str(tmp_path), False, score)
    for path in [valid, seeds, score, mask]:
        ds = gdal.Open(path)
        assert (ds.GetMetadata('IMAGE_STRUCTURE').get('COMPRESSION', 'NONE') != 'NONE') == enabled


@pytest.mark.parametrize('enabled', [True, False])
def test_mask_and_overview_copies_honor_compression(tmp_path, enabled):
    from spectralmatch import mask_rasters
    source = raster(tmp_path)
    masked = str(tmp_path / 'masked.tif')
    copied = str(tmp_path / 'copied.tif')
    mask_rasters([source], [masked], compression=enabled)
    compute_overviews([masked], output_image_paths=[copied], window_scales=(2,), compression=enabled)
    for path in [masked, copied]:
        ds = gdal.Open(path)
        assert (ds.GetMetadata('IMAGE_STRUCTURE').get('COMPRESSION', 'NONE') != 'NONE') == enabled
        np.testing.assert_array_equal(ds.ReadAsArray(), gdal.Open(source).ReadAsArray())
