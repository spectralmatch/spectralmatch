"""References are read-only paths; each image role has explicit matching bands."""
import hashlib
import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from spectralmatch import joint_coregistration
from .test_joint_coregistration import _translation_fixture
from .utils_test import create_dummy_raster

module=importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')


@pytest.mark.parametrize('name',['input_images_bands','fixed_images_bands'])
@pytest.mark.parametrize('value',[None,[],(1,),[0],[-1],[True],[1,1],[1,2.5],'1'])
def test_matching_bands_require_nonempty_unique_one_based_integer_lists(tmp_path,name,value):
    paths,_=_translation_fixture(tmp_path)
    with pytest.raises(ValueError,match=name+' must be'):
        joint_coregistration(paths,[str(tmp_path/'out_a.tif'),str(tmp_path/'out_b.tif')],**{name:value})


def test_defaults_and_separate_reference_paths_write_only_moving_output(tmp_path):
    paths,cache=_translation_fixture(tmp_path)
    parameters=inspect.signature(joint_coregistration).parameters
    assert parameters['input_images_bands'].default==parameters['fixed_images_bands'].default==[1]
    original=Path(paths[0]).read_bytes()
    output=str(tmp_path/'moving.tif')
    result=joint_coregistration(paths[1:],[output],fixed_images=[paths[0]],tie_load_path=cache,local_model='none')
    assert result==[output]
    assert gdal.Open(output).GetGeoTransform()[0]==pytest.approx(0,abs=1e-6)
    assert Path(paths[0]).read_bytes()==original
    assert not list(tmp_path.glob('*.vrt'))
    assert parameters['input_images_bands'].default==parameters['fixed_images_bands'].default==[1]


@pytest.mark.parametrize('role',['input_images_bands','fixed_images_bands'])
def test_original_band_count_is_checked_before_temporary_alpha_is_added(tmp_path,monkeypatch,role):
    paths,_=_translation_fixture(tmp_path)
    def forbidden(*args,**kwargs):pytest.fail('Band error must be detected before matching')
    monkeypatch.setattr(module,'_collect_ties',forbidden)
    with pytest.raises(ValueError,match='exceed the 1 bands'):
        joint_coregistration(paths[1:],[str(tmp_path/'out.tif')],fixed_images=[paths[0]],**{role:[2]})


def test_reference_cannot_be_an_output_even_through_symlink(tmp_path):
    paths,cache=_translation_fixture(tmp_path)
    alias=tmp_path/'ref_alias.tif';alias.symlink_to(paths[0])
    with pytest.raises(ValueError,match='does not support in-place'):
        joint_coregistration(paths[1:],[str(alias)],fixed_images=[paths[0]],tie_load_path=cache)


def test_band_average_precedes_percentile_stretch(tmp_path):
    path=str(tmp_path/'bands.tif')
    create_dummy_raster(path,width=64,height=64,count=4,dtype='float32',nodata=-9999,
                        crs='EPSG:3857',transform=(0,1,0,64,0,-1))
    yy,xx=np.mgrid[:64,:64]
    a=(xx**2).astype('float32');b=(yy*17).astype('float32')
    with gdal.Open(path,gdal.GA_Update) as ds:
        ds.GetRasterBand(1).Fill(999)
        ds.GetRasterBand(2).WriteArray(a)
        ds.GetRasterBand(3).Fill(0)
        ds.GetRasterBand(4).WriteArray(b)
    valid=np.ones((64,64),dtype=bool)
    actual=module._read_uint8_gray(path,valid,bands=[2,4])
    mean=(a+b)/2;low,high=np.percentile(mean,[2,98])
    expected=(np.clip((mean-low)/(high-low),0,1)*255).astype('uint8')
    np.testing.assert_array_equal(actual,expected)


def test_selected_band_nodata_and_nonfinite_pixels_are_excluded(tmp_path):
    path=str(tmp_path/'bands.tif')
    create_dummy_raster(path,width=32,height=32,count=3,dtype='float32',nodata=-9999,
                        crs='EPSG:3857',transform=(0,1,0,32,0,-1))
    values=np.ones((32,32),dtype='float32');values[4,5]=-9999;values[8,9]=np.nan
    with gdal.Open(path,gdal.GA_Update) as ds:
        ds.GetRasterBand(2).WriteArray(values)
    with gdal.Open(path) as ds:
        valid=module._build_overlap_valid_mask(ds,ds,(0,0,32,32),(2,),(2,))
    assert not valid[4,5] and not valid[8,9] and valid.sum()==1022


def test_real_orb_matching_uses_selected_bands_for_each_role(tmp_path):
    from scipy.ndimage import gaussian_filter,shift
    base=gaussian_filter(np.random.default_rng(716).normal(size=(192,192)),1.5).astype('float32')
    moving=str(tmp_path/'a_moving.tif');reference=str(tmp_path/'z_reference.tif')
    for path,count in [(moving,4),(reference,3)]:
        create_dummy_raster(path,width=192,height=192,count=count,dtype='float32',nodata=-9999,
                            crs='EPSG:3857',transform=(0,1,0,192,0,-1))
        with gdal.Open(path,gdal.GA_Update) as ds:
            for band in range(1,count+1):ds.GetRasterBand(band).Fill(0)
    with gdal.Open(moving,gdal.GA_Update) as ds:
        data=shift(base,(2,4),mode='nearest')
        ds.GetRasterBand(2).WriteArray(data)
        ds.GetRasterBand(4).WriteArray(data*2)
    with gdal.Open(reference,gdal.GA_Update) as ds:ds.GetRasterBand(3).WriteArray(base)
    before=hashlib.sha256(Path(reference).read_bytes()).digest()
    output=str(tmp_path/'aligned.tif')
    joint_coregistration([moving],[output],fixed_images=[reference],input_images_bands=[2,4],
                         fixed_images_bands=[3],fixed_tie_method='orb',
                         tie_grid_spacing=96,tie_search_radius='48px',tie_orb_max_features=300,
                         local_model='none')
    with gdal.Open(output) as ds:
        assert ds.RasterCount==4
        np.testing.assert_allclose([ds.GetGeoTransform()[0],ds.GetGeoTransform()[3]],[-4,194],atol=.75)
    assert hashlib.sha256(Path(reference).read_bytes()).digest()==before


def test_pipeline_forwards_both_band_lists(tmp_path,monkeypatch):
    from spectralmatch import chain
    paths,_=_translation_fixture(tmp_path);calls=[]
    monkeypatch.setattr(chain,'joint_coregistration',lambda **kwargs:calls.append(kwargs) or ['aligned.tif'])
    chain.pipeline(shared_input_images=paths[1:],shared_output_image_path=str(tmp_path/'outputs'),
                   shared_temp_dir=str(tmp_path/'work'),steps=('joint_coregistration',),
                   joint_coregistration_fixed_images=[paths[0]],
                   joint_coregistration_input_images_bands=[2,4],joint_coregistration_fixed_images_bands=[3])
    assert calls[0]['fixed_images']==[paths[0]]
    assert calls[0]['input_images_bands']==[2,4]
    assert calls[0]['fixed_images_bands']==[3]


@pytest.mark.parametrize('parameter',['tie_save_path','tie_save_crs_path'])
def test_tie_files_cannot_overwrite_reference_inputs(tmp_path,parameter):
    paths,cache=_translation_fixture(tmp_path)
    original=Path(paths[0]).read_bytes()
    with pytest.raises(ValueError,match=parameter+' must differ'):
        joint_coregistration(paths[1:],[str(tmp_path/'out.tif')],fixed_images=[paths[0]],
                             tie_load_path=cache,**{parameter:paths[0]})
    assert Path(paths[0]).read_bytes()==original


@pytest.mark.parametrize('mode',['partial_influence','no_influence','error'])
def test_reference_validity_and_coverage_use_selected_bands(tmp_path,mode):
    moving=str(tmp_path/'a_moving.tif');reference=str(tmp_path/'z_reference.tif')
    for path,count in [(moving,1),(reference,2)]:
        create_dummy_raster(path,width=24,height=24,count=count,dtype='float32',nodata=-9999,
                            crs='EPSG:3857',transform=(0,1,0,24,0,-1))
    with gdal.Open(reference,gdal.GA_Update) as ds:
        ds.GetRasterBand(1).Fill(-9999)
        ds.GetRasterBand(2).Fill(100)
    cache=str(tmp_path/'ties.json')
    module._save_ties(cache,{('a_moving','z_reference'):np.array([[5.,5.,5.,5.]])})
    output=str(tmp_path/'out.tif')
    joint_coregistration([moving],[output],fixed_images=[reference],fixed_images_bands=[2],
                         fixed_on_no_or_partial_overlap=mode,tie_load_path=cache,local_model='none')
    assert gdal.Open(output).GetGeoTransform()==gdal.Open(moving).GetGeoTransform()
