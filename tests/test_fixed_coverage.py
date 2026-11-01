"""Fixed coverage policies use valid data and preserve moving-to-moving ties."""
import hashlib
import importlib
import json

import numpy as np
import pytest
from osgeo import gdal, osr

from spectralmatch import joint_coregistration
from spectralmatch.joint_coregistration.fixed_coverage import fully_covered_images

module = importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')


def raster(path, *, x=0, valid=None, mask_kind='nodata', transform=None):
    valid = np.ones((24, 24), dtype=bool) if valid is None else valid
    ds = gdal.GetDriverByName('GTiff').Create(str(path), valid.shape[1], valid.shape[0],
                                           2 if mask_kind == 'alpha' else 1,
                                           gdal.GDT_Byte if mask_kind == 'alpha' else gdal.GDT_Int16)
    crs = osr.SpatialReference(); crs.ImportFromEPSG(3857)
    ds.SetProjection(crs.ExportToWkt())
    ds.SetGeoTransform(transform or (x, 1, 0, 24, 0, -1))
    band = ds.GetRasterBand(1)
    band.WriteArray(np.where(valid, 100, -9999) if mask_kind == 'nodata' else np.full(valid.shape, 100))
    if mask_kind == 'nodata':
        band.SetNoDataValue(-9999)
    elif mask_kind == 'mask':
        band.CreateMaskBand(gdal.GMF_PER_DATASET)
        band.GetMaskBand().WriteArray(valid.astype('uint8')*255)
    else:
        band.SetColorInterpretation(gdal.GCI_GrayIndex)
        ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_AlphaBand)
        ds.GetRasterBand(2).WriteArray(valid.astype('uint8')*255)
    band = None; ds = None
    return str(path)


def infos(paths):
    from pathlib import Path
    return {Path(path).stem: module._read_image_info(Path(path).stem, path) for path in paths}


@pytest.mark.parametrize('mask_kind', ['nodata', 'mask', 'alpha'])
def test_coverage_uses_valid_reference_data_not_just_bounds(tmp_path, mask_kind):
    valid = np.ones((24,24), dtype=bool); valid[8:16,8:16] = False
    fixed = raster(tmp_path/'fixed.tif', valid=valid, mask_kind=mask_kind)
    moving = raster(tmp_path/'moving.tif')
    assert fully_covered_images(infos([fixed,moving]), {'fixed'}) == set()
    # Moving NoData over the reference hole does not require coverage.
    raster(tmp_path/'moving.tif', valid=valid)
    assert fully_covered_images(infos([fixed,moving]), {'fixed'}) == {'moving'}


def test_coverage_uses_union_of_fixed_images(tmp_path):
    left=np.zeros((24,24),dtype=bool); left[:,:12]=True
    right=~left
    paths=[raster(tmp_path/'left.tif',valid=left),raster(tmp_path/'right.tif',valid=right),
           raster(tmp_path/'moving.tif')]
    data=infos(paths)
    assert fully_covered_images(data, {'left'}) == set()
    assert fully_covered_images(data, {'left','right'}) == {'moving'}


@pytest.mark.parametrize('origin', [2,100])
def test_partial_or_missing_extent_is_not_full_coverage(tmp_path,origin):
    paths=[raster(tmp_path/'fixed.tif'),raster(tmp_path/'moving.tif',x=origin)]
    assert fully_covered_images(infos(paths),{'fixed'}) == set()


def test_rotated_grids_use_actual_pixel_centres(tmp_path):
    gt=(0,1,.2,24,.1,-1)
    paths=[raster(tmp_path/'fixed.tif',transform=gt),raster(tmp_path/'moving.tif',transform=gt)]
    assert fully_covered_images(infos(paths),{'fixed'}) == {'moving'}


def network(tmp_path):
    valid=np.ones((24,24),dtype=bool); valid[8:16,8:16]=False
    partial=raster(tmp_path/'a_partial.tif')
    full_valid=np.zeros((24,24),dtype=bool); full_valid[:8,:]=True
    full=raster(tmp_path/'b_full.tif',valid=full_valid)
    fixed=raster(tmp_path/'z_fixed.tif',valid=valid)
    ties={('a_partial','b_full'):np.array([[6.,4.,5.,4.]]),
          ('a_partial','z_fixed'):np.array([[6.,4.,4.,4.],[10.,10.,10.,10.]]),
          ('b_full','z_fixed'):np.array([[5.,4.,4.,4.]])}
    cache=tmp_path/'input_ties.json';module._save_ties(str(cache),ties)
    return [partial,full,fixed],str(cache)


@pytest.mark.parametrize('model',['none','piecewise_affine'])
def test_default_partial_filters_cached_nodata_ties_and_preserves_fixed_file(tmp_path,model):
    paths,cache=network(tmp_path)
    before=hashlib.sha256(open(paths[2],'rb').read()).digest()
    outputs=[str(tmp_path/f'out_{i}.tif') for i in range(3)]
    saved=tmp_path/'selected.json'
    joint_coregistration(paths[:2],outputs[:2],fixed_images=[paths[2]],tie_load_path=cache,
                         tie_save_path=str(saved),local_model=model,local_grid_spacing=12)
    selected=module._load_ties(str(saved))
    assert len(selected[('a_partial','z_fixed')])==1
    assert len(selected[('a_partial','b_full')])==1
    np.testing.assert_allclose(gdal.Open(outputs[0]).GetGeoTransform()[0],-2,atol=.01)
    assert hashlib.sha256(open(paths[2],'rb').read()).digest()==before
    assert gdal.Open(paths[2]).GetGeoTransform()==(0,1,0,24,0,-1)


def test_no_influence_removes_only_direct_fixed_pairs_and_still_allows_indirect_anchor(tmp_path):
    paths,cache=network(tmp_path)
    saved=tmp_path/'selected.json';outputs=[str(tmp_path/f'out_{i}.tif') for i in range(3)]
    joint_coregistration(paths[:2],outputs[:2],fixed_images=[paths[2]],tie_load_path=cache,
                         tie_save_path=str(saved),local_model='none',
                         fixed_on_no_or_partial_overlap='no_influence')
    selected=module._load_ties(str(saved))
    assert set(selected)=={('a_partial','b_full'),('b_full','z_fixed')}
    # The partial image still receives the fixed reference alignment indirectly.
    assert gdal.Open(outputs[0]).GetGeoTransform()[0]==pytest.approx(-2,abs=.01)


def test_error_policy_fails_before_matching_for_reference_holes(tmp_path,monkeypatch):
    paths,cache=network(tmp_path)
    def forbidden(*args,**kwargs):
        pytest.fail('Coverage failure must precede tie matching')
    monkeypatch.setattr(module,'_collect_ties',forbidden)
    outputs=[str(tmp_path/f'out_{i}.tif') for i in range(3)]
    with pytest.raises(ValueError,match='not fully covered.*a_partial'):
        joint_coregistration(paths[:2],outputs[:2],fixed_images=[paths[2]],
                             fixed_on_no_or_partial_overlap='error')
    from pathlib import Path
    assert not any(Path(path).exists() for path in outputs)


def test_error_accepts_full_coverage_and_rejects_invalid_policy(tmp_path):
    paths=[raster(tmp_path/'a.tif'),raster(tmp_path/'b.tif')]
    cache=tmp_path/'ties.json';module._save_ties(str(cache),{('a','b'):np.array([[5.,5.,5.,5.]])})
    outputs=[str(tmp_path/'out_a.tif'),str(tmp_path/'out_b.tif')]
    joint_coregistration(paths[1:],outputs[1:],fixed_images=[paths[0]],tie_load_path=str(cache),
                         fixed_on_no_or_partial_overlap='error',local_model='none')
    with pytest.raises(ValueError,match='fixed_on_no_or_partial_overlap must be'):
        joint_coregistration(paths,outputs,fixed_on_no_or_partial_overlap='typo')


def test_disconnected_moving_network_is_not_forced_to_reference(tmp_path):
    paths=[raster(tmp_path/'a.tif'),raster(tmp_path/'b.tif',x=2),raster(tmp_path/'fixed.tif',x=100)]
    cache=tmp_path/'ties.json';module._save_ties(str(cache),{('a','b'):np.array([[5.,5.,5.,5.]])})
    outputs=[str(tmp_path/f'out_{i}.tif') for i in range(3)]
    joint_coregistration(paths[:2],outputs[:2],fixed_images=[paths[2]],tie_load_path=str(cache),
                         fixed_on_no_or_partial_overlap='no_influence',local_model='none')
    assert gdal.Open(outputs[0]).GetGeoTransform()[0]==pytest.approx(1,abs=.01)
    assert gdal.Open(outputs[1]).GetGeoTransform()[0]==pytest.approx(1,abs=.01)
    assert gdal.Open(paths[2]).GetGeoTransform()[0]==100


@pytest.mark.parametrize('mode',['partial_influence','no_influence','error'])
def test_pipeline_forwards_coverage_policy(tmp_path,monkeypatch,mode):
    from spectralmatch import chain
    paths=[raster(tmp_path/'a.tif'),raster(tmp_path/'fixed.tif')]
    calls=[]
    def capture(**kwargs):
        calls.append(kwargs);return [str(tmp_path/'aligned.tif')]
    monkeypatch.setattr(chain,'joint_coregistration',capture)
    chain.pipeline(shared_input_images=paths[:1],shared_output_image_path=str(tmp_path/'outputs'),
                   shared_temp_dir=str(tmp_path/'work'),steps=('joint_coregistration',),
                   joint_coregistration_fixed_images=[paths[1]],
                   joint_coregistration_fixed_on_no_or_partial_overlap=mode)
    assert calls[0]['fixed_on_no_or_partial_overlap']==mode
