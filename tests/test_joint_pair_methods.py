"""Matcher selection follows fixed-image membership, independently of basename order."""
from dataclasses import replace
import importlib
import numpy as np
import pytest

from spectralmatch import joint_coregistration
from .test_joint_coregistration import _translation_fixture
from .utils_test import create_dummy_raster

module=importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')


@pytest.mark.parametrize('fixed, input_method, fixed_method, expected', [
    ((), 'orb', 'elastix', {('a','b'):'orb', ('a','z'):'orb', ('b','z'):'orb'}),
    (('z',), 'orb', 'orb', {('a','b'):'orb', ('a','z'):'orb', ('b','z'):'orb'}),
    ((), 'elastix', 'elastix', {('a','b'):'elastix', ('a','z'):'elastix', ('b','z'):'elastix'}),
    (('z',), 'orb', 'elastix', {('a','b'):'orb', ('a','z'):'elastix', ('b','z'):'elastix'}),
    (('z',), 'elastix', 'orb', {('a','b'):'elastix', ('a','z'):'orb', ('b','z'):'orb'}),
    (('b','z'), 'orb', 'elastix', {('a','b'):'elastix', ('a','z'):'elastix', ('b','z'):'elastix'}),
])
def test_pair_methods_are_independent_and_use_processes_if_any_pair_needs_elastix(
        tmp_path, monkeypatch, fixed, input_method, fixed_method, expected):
    inputs,_=_translation_fixture(tmp_path)
    base=module._read_image_info('a',inputs[0])
    infos={name:replace(base,name=name) for name in ('a','b','z')}
    calls={}; executor_backends=[]
    def matches(i,j,method,*args):
        pair=module._canonical_pair(i.name,j.name)
        calls[pair]=method
        if fixed:
            assert set(args[-1])==set(fixed)
        return pair,np.array([[5.,5.,5.,5.]]),3.
    def run(function,args,**kwargs):
        executor_backends.append(kwargs['backend'])
        return [function(*arg) for arg in args]
    from spectralmatch.joint_coregistration import elastix
    required=[]
    monkeypatch.setattr(elastix,'require_elastix',lambda:required.append(True))
    monkeypatch.setattr(module,'_extract_pair_ties',matches)
    monkeypatch.setattr(module,'_run_image_tasks',run)
    ties,_=module._collect_ties(tuple(expected),infos,{},input_method,None,None,None,False,
                               fixed_images=fixed,fixed_feature_method=fixed_method)
    assert calls==expected and set(ties)==set(expected)
    assert bool(required)==('elastix' in expected.values())
    assert executor_backends==(['process'] if required else ['thread'])


@pytest.mark.parametrize('fixed', [None, []])
@pytest.mark.parametrize('method', ['orb', 'elastix'])
def test_fixed_matcher_is_unused_without_references(tmp_path,monkeypatch,fixed,method):
    inputs,ties=_translation_fixture(tmp_path)
    from spectralmatch.joint_coregistration import elastix
    monkeypatch.setattr(elastix,'require_elastix',lambda:pytest.fail('No unmatched fixed pairs'))
    outputs=[str(tmp_path/'out_a.tif'),str(tmp_path/'out_b.tif')]
    joint_coregistration(inputs,outputs,fixed_images=fixed,fixed_tie_method=method,
                         tie_load_path=ties,local_model='none')


def test_invalid_fixed_matcher_is_rejected(tmp_path):
    inputs,_=_translation_fixture(tmp_path)
    with pytest.raises(ValueError,match='fixed_tie_method must be'):
        joint_coregistration(inputs[1:],[str(tmp_path/'out-b.tif')],
                             fixed_images=[inputs[0]],fixed_tie_method='unknown')


def test_default_fixed_method_is_elastix_and_default_input_method_is_orb(tmp_path,monkeypatch):
    inputs,ties=_translation_fixture(tmp_path)
    original=module._collect_ties
    calls=[]
    def collect(*args,**kwargs):
        calls.append((args[3],kwargs['fixed_feature_method'],kwargs['fixed_images']))
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'_collect_ties',collect)
    joint_coregistration(inputs[1:],[str(tmp_path/'out.tif')],
                         fixed_images=[inputs[0]],tie_load_path=ties,local_model='none')
    assert calls==[('orb','elastix',{'a'})]


def test_fixed_image_is_matching_target_but_selected_tie_cache_stays_canonical(tmp_path,monkeypatch):
    inputs,_=_translation_fixture(tmp_path)
    infos=[module._read_image_info(name,path) for name,path in zip(('a_moving','z_fixed'),inputs)]
    for info in infos:
        create_dummy_raster(info.path,width=192,height=192,count=1,crs='EPSG:3857',transform=(0,1,0,192,0,-1))
    infos=[module._read_image_info(info.name,info.path) for info in infos]
    seen=[]
    def grid(reference,sensed,i,j,*args):
        seen.append((i.name,j.name))
        return np.array([[50.,50.,55.,52.]]),np.array([[50.,142.]])
    monkeypatch.setattr(module,'_extract_grid_pixel_matches',grid)
    pair,points,_=module._extract_pair_ties(*infos,'elastix',None,None,False,fixed_images=('z_fixed',))
    assert seen==[('z_fixed','a_moving')]
    assert pair==('a_moving','z_fixed')
    np.testing.assert_allclose(points,[[55.,52.,50.,50.]])


def test_loaded_pairs_do_not_require_the_unused_elastix_backend(tmp_path,monkeypatch):
    inputs,ties=_translation_fixture(tmp_path)
    from spectralmatch.joint_coregistration import elastix
    def unavailable():
        raise AssertionError('Loaded ties must not import an unused matcher')
    monkeypatch.setattr(elastix,'require_elastix',unavailable)
    joint_coregistration(inputs[1:],[str(tmp_path/'out.tif')],
                         fixed_images=[inputs[0]],tie_load_path=ties,local_model='none')


def test_real_mixed_method_network_with_multiband_inputs_and_fixed_name_sorting_last(tmp_path):
    pytest.importorskip('itk')
    from scipy.ndimage import gaussian_filter,shift
    from osgeo import gdal
    base=gaussian_filter(np.random.default_rng(716).normal(size=(192,192)),2).astype('float32')
    inputs=[]
    for name,displacement,count in [('a_moving',(2,4),8),('b_moving',(-3,1),8),('z_reference',(0,0),1)]:
        path=str(tmp_path/f'{name}.tif')
        create_dummy_raster(path,width=192,height=192,count=count,dtype='float32',nodata=-9999,
                            crs='EPSG:3857',transform=(0,1,0,192,0,-1))
        values=shift(base,displacement,mode='nearest')*100+500
        with gdal.Open(path,gdal.GA_Update) as ds:
            for band in range(1,count+1):
                ds.GetRasterBand(band).WriteArray(values+band)
        inputs.append(path)
    outputs=[str(tmp_path/'a_corrected.tif'),str(tmp_path/'b_corrected.tif')]
    reference_pixels=gdal.Open(inputs[2]).ReadAsArray().copy()
    joint_coregistration(inputs[:2],outputs,fixed_images=[inputs[2]],
                         tie_grid_spacing=96,tie_search_radius='48px',local_model='none',
                         tie_orb_max_features=300,image_threads=2)
    for output,expected in zip(outputs[:2],[(-4,194),(-1,189)]):
        with gdal.Open(output) as ds:
            gt=ds.GetGeoTransform()
            assert ds.RasterCount==8
            np.testing.assert_allclose([gt[0],gt[3]],expected,atol=.75)
    with gdal.Open(inputs[2]) as fixed:
        assert fixed.GetGeoTransform()==(0,1,0,192,0,-1)
        np.testing.assert_array_equal(fixed.ReadAsArray(),reference_pixels)


@pytest.mark.parametrize('fixed_method', ['elastix','orb'])
def test_pipeline_forwards_independent_matcher_settings(tmp_path,monkeypatch,fixed_method):
    from spectralmatch import chain
    inputs,_=_translation_fixture(tmp_path)
    calls=[]
    def capture(**kwargs):
        calls.append(kwargs)
        return [str(tmp_path/'aligned.tif')]
    monkeypatch.setattr(chain,'joint_coregistration',capture)
    chain.pipeline(shared_input_images=inputs[1:],shared_output_image_path=str(tmp_path/'outputs'),
                   shared_temp_dir=str(tmp_path/'work'),steps=('joint_coregistration',),
                   joint_coregistration_fixed_images=[inputs[0]],
                   joint_coregistration_moving_tie_method='elastix',
                   joint_coregistration_fixed_tie_method=fixed_method)
    assert len(calls)==1
    assert calls[0]['moving_tie_method']=='elastix'
    assert calls[0]['fixed_tie_method']==fixed_method
    assert calls[0]['fixed_images']==[inputs[0]]
