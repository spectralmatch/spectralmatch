"""PIF residual alignment is optional and uses the fixed-reference joint model."""
import importlib
import json

import numpy as np
import pytest
from osgeo import gdal, osr
from scipy.ndimage import gaussian_filter, map_coordinates

from spectralmatch import Match, pipeline
from spectralmatch.pif import pif
from spectralmatch.types_and_validation import Match as MatchValidation
from .test_pif_windows import rasters, ties

joint = importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')


def write(path, values, pixel_size=1):
    srs = osr.SpatialReference(); srs.ImportFromEPSG(3857)
    h, w = values.shape
    ds = gdal.GetDriverByName('GTiff').Create(str(path), w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((200000, pixel_size, 0, 2200000, 0, -pixel_size))
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).SetNoDataValue(-9999)
    ds.GetRasterBand(1).WriteArray(values)
    ds = None
    return str(path)


@pytest.mark.parametrize('geometry', ['collinear', 'nearly_collinear', 'horizontal', 'vertical', 'distributed'])
def test_pair_translation_handles_degenerate_geometry_without_tps(tmp_path, monkeypatch, geometry):
    values = gaussian_filter(np.random.default_rng(5).normal(size=(128, 128)), 1).astype('float32') + 100
    moving = np.full_like(values, -9999)
    moving[2:, 3:] = values[:-2, :-3]
    ref = write(tmp_path/'reference.tif', values)
    src = write(tmp_path/'moving.tif', moving)
    points = {'collinear': [(20, 20), (50, 50), (90, 90)],
              'nearly_collinear': [(20, 20), (50, 50.01), (90, 90)],
              'horizontal': [(50, 20), (50, 50), (50, 90)],
              'vertical': [(20, 50), (50, 50), (90, 50)],
              'distributed': [(20, 20), (90, 20), (20, 90), (90, 90)]}[geometry]
    pairs = [(r, c, r+2, c+3) for r, c in points]
    original = gdal.WarpOptions
    def options(*args, **kwargs):
        assert not kwargs.get('tps', False)
        return original(*args, **kwargs)
    monkeypatch.setattr(gdal, 'WarpOptions', options)
    out, used = joint._coregister_overlap(ref, src, str(tmp_path/'corrected.tif'), tie_pairs=pairs)
    ds = gdal.Open(out)
    np.testing.assert_allclose(ds.GetRasterBand(1).ReadAsArray()[8:-8,8:-8], values[8:-8,8:-8], atol=1e-4)
    assert ds.GetGeoTransform() == gdal.Open(ref).GetGeoTransform()
    assert ds.GetRasterBand(ds.RasterCount).GetColorInterpretation() == gdal.GCI_AlphaBand
    assert np.all(ds.GetRasterBand(ds.RasterCount).ReadAsArray()[-2:, :] == 0)
    assert len(used) == len(pairs)
    np.testing.assert_array_equal(gdal.Open(src).ReadAsArray(), moving)


def test_local_residual_model_and_window_use_shared_joint_solver(tmp_path, monkeypatch):
    values = gaussian_filter(np.random.default_rng(7).normal(size=(128,128)), 2).astype('float32')+100
    rr, cc = np.indices(values.shape, dtype=float)
    # Moving features drift horizontally from 1 to 5 pixels across rows.
    shift = 1 + rr/32
    moving = map_coordinates(values, [rr, cc-shift], order=1, mode='constant', cval=-9999)
    ref=write(tmp_path/'ref.tif',values,10);src=write(tmp_path/'moving.tif',moving,10)
    pairs=[(r,c,r,c+1+r/32) for r in (16,48,80,112) for c in (20,60,100)]
    called=[];solve=joint._solve_local_alignment
    def local(*args,**kwargs):
        result=solve(*args,**kwargs)
        assert np.all(result['reference'].displacement == 0)
        assert np.any(np.abs(result['moving'].displacement)>0.01)
        called.append(True)
        return result
    monkeypatch.setattr(joint,'_solve_local_alignment',local)
    out,_=joint._coregister_overlap(ref,src,str(tmp_path/'corrected.tif'),tie_pairs=pairs)
    ds=gdal.Open(out);actual=ds.GetRasterBand(1).ReadAsArray();ds=None
    sl=np.s_[12:-12,12:-12]
    before=np.mean((values[sl]-moving[sl])**2);after=np.mean((values[sl]-actual[sl])**2)
    assert after < before * .5, (before,after)
    assert called == [True]
    win,_=joint._coregister_overlap(ref,src,str(tmp_path/'window.tif'),tie_pairs=pairs,output_window=(20,30,40,35))
    ds=gdal.Open(win)
    np.testing.assert_allclose(ds.GetRasterBand(1).ReadAsArray(),actual[30:65,20:60],atol=1e-4)
    assert ds.GetGeoTransform() == (200200,10,0,2199700,0,-10)


@pytest.mark.parametrize('scope',['overlap','matched_windows'])
@pytest.mark.parametrize('enabled',[False,True])
def test_both_modes_keep_pif_statistics_and_only_optionally_warp(tmp_path,monkeypatch,scope,enabled):
    paths=rasters(tmp_path)
    original=pif._coregister_overlap;calls=[]
    def warp(*args,**kwargs):
        assert enabled, 'Disabled PIF alignment must not fit or warp tie corrections'
        calls.append(kwargs.get('output_window'))
        return original(*args,**kwargs)
    monkeypatch.setattr(pif,'_coregister_overlap',warp)
    params,stats,_=pif.Pif.flood_from_match_points(
        paths,load_ties=ties(tmp_path),tie_sampling_method='grid',coregister=enabled,
        stats_scope=scope,window_size=33,estimate_stats=False,inz_threshold=10,
        _return_statistics=True)
    assert np.isfinite(params).all()
    assert stats['A']['B'][0]['size'] == stats['B']['A'][0]['size'] > 0
    assert stats['B']['A'][0]['mean'] == pytest.approx(1.1*stats['A']['B'][0]['mean']+7,rel=1e-6)
    assert len(calls) == ((1 if scope == 'overlap' else 4) if enabled else 0)


def test_default_global_and_pipeline_skip_residual_warp(tmp_path,monkeypatch):
    paths=rasters(tmp_path)
    def forbidden(*a,**kw):raise AssertionError('Residual coregistration must default to off')
    monkeypatch.setattr(pif,'_coregister_overlap',forbidden)
    model=tmp_path/'model.json'
    outputs=Match.global_regression(paths,[str(tmp_path/f'{i}_out.tif') for i in range(2)],
        pif_load_ties=ties(tmp_path),tie_sampling_method='grid',pif_inz_threshold=10,
        estimate_stats=False,save_adjustments=str(model))
    assert len(outputs)==2 and json.loads(model.read_text())['A']['overlap_stats']['B']['band_0']['size']>0
    pipeline(paths,str(tmp_path/'pipeline'/'$.tif'),steps=('global_regression',),
        global_regression_pif_load_ties=ties(tmp_path),global_regression_tie_sampling_method='grid',
        global_regression_pif_inz_threshold=10,shared_image_threads=None,shared_tile_threads=1,
        shared_io_threads=1,shared_temp_dir=str(tmp_path/'work'))


def test_overlap_vrts_share_exact_grid_at_fractional_bounds(tmp_path):
    paths=rasters(tmp_path)
    ds=gdal.Open(paths[1],gdal.GA_Update);ds.SetGeoTransform((.31,1,0,127.73,0,-1));ds=None
    ref,src,w,h,gt,_=pif._build_overlap_vrts(*paths,str(tmp_path))
    a,b=gdal.Open(ref),gdal.Open(src)
    assert a.GetGeoTransform()==b.GetGeoTransform()==gt
    assert (a.RasterXSize,a.RasterYSize)==(b.RasterXSize,b.RasterYSize)==(w,h)


@pytest.mark.parametrize('value',[None,1,'false'])
def test_coregistration_option_requires_bool(value):
    with pytest.raises(ValueError,match='pif_coregister'):
        MatchValidation._validate_global_regression(pif_coregister=value)
    with pytest.raises(ValueError,match='coregister'):
        pif.Pif.flood_from_match_points([],coregister=value)


def test_small_local_correction_does_not_shift_pixel_centers(tmp_path):
    rr, cc = np.indices((128, 128))
    values = (50 + cc * .1 + rr * .2).astype('float32')
    ref = write(tmp_path / 'ref.tif', values, 10)
    moving = write(tmp_path / 'moving.tif', values, 10)
    pairs = [(r, c, r, c + .001 * r / 128)
             for r in (16, 48, 80, 112) for c in (20, 60, 100)]
    out, _ = joint._coregister_overlap(ref, moving, str(tmp_path / 'corrected.tif'), tie_pairs=pairs)
    ds = gdal.Open(out)
    np.testing.assert_allclose(ds.GetRasterBand(1).ReadAsArray()[10:-10, 10:-10],
                               values[10:-10, 10:-10], atol=.001)
