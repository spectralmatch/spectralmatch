import importlib
import numpy as np
import pytest
from osgeo import gdal
from spectralmatch import joint_coregistration
from .test_joint_coregistration import _translation_fixture, _write_ties, _pair_record
from .utils_test import create_dummy_raster

module = importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')


def test_fixed_reference_gets_full_global_correction(tmp_path):
    import hashlib
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path/'moving.tif')]
    before = hashlib.sha256(open(inputs[0],'rb').read()).digest()
    joint_coregistration(inputs[1:], outputs, fixed_images=[inputs[0]], tie_load_path=ties, local_model='none',
                         resolution=2, output_dtype='float32', custom_nodata_value=-123)
    assert hashlib.sha256(open(inputs[0],'rb').read()).digest()==before
    assert gdal.Open(outputs[0]).GetGeoTransform()[0] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("local_model", ["bilinear", "piecewise_affine"])
def test_multiple_fixed_images_and_network(local_model):
    def info(name):
        return module._ImageInfo(name, '', (0,1,0,100,0,-1), '', 100,100,(0,0,100,100),1,(50,50),100)
    infos = {name:info(name) for name in ('a','b','c','d')}
    ties = {('a','b'):np.array([[10,10,12,10]]), ('b','c'):np.array([[12,10,14,10]]),
            ('c','d'):np.array([[14,10,10,10]])}
    params = module._solve_global_alignment(infos, ties, 'translation', dict.fromkeys(infos,1),
                                            1, 'none', 1, False, fixed_images={'a','d'})
    np.testing.assert_array_equal(params['a'], [0,0])
    np.testing.assert_array_equal(params['d'], [0,0])
    np.testing.assert_allclose(params['b'], [-2,0], atol=1e-7)
    np.testing.assert_allclose(params['c'], [-4,0], atol=1e-7)
    meshes = module._solve_local_alignment(infos, ties, 'none', dict.fromkeys(infos,np.empty(0)),
                 local_model, dict.fromkeys(infos,1), 1, 50, .01, .01, 500, 'none', 1, False,
                 fixed_images={'a','d'})
    assert np.all(meshes['a'].displacement == 0)
    assert np.all(meshes['d'].displacement == 0)
    assert np.linalg.norm(meshes['b'].displacement) > 0
    assert np.linalg.norm(meshes['c'].displacement) > 0


def test_unknown_fixed_and_disconnected_fail(tmp_path):
    inputs, ties = _translation_fixture(tmp_path)
    outputs = [str(tmp_path/'a-out.tif'),str(tmp_path/'b-out.tif')]
    with pytest.raises(FileNotFoundError, match='Fixed reference raster does not exist'):
        joint_coregistration(inputs, outputs, fixed_images=['typo'])
    _write_ties(ties, [{'image_1':'a','image_2':'b','points':[]}])
    with pytest.raises(ValueError, match='No usable tie-point overlap pairs remain'):
        joint_coregistration(inputs[1:], outputs[1:], fixed_images=[inputs[0]], tie_load_path=ties)


def test_real_elastix_window():
    pytest.importorskip('itk')
    from scipy.ndimage import gaussian_filter, shift
    from spectralmatch.joint_coregistration.elastix import match_window
    data = gaussian_filter(np.random.default_rng(12).normal(size=(96,96)),1.5).astype('float32')
    points = match_window(data, shift(data,(3,5)), np.ones(data.shape,bool))
    assert points.shape == (1,4)
    np.testing.assert_allclose(points[0,2:]-points[0,:2], [5,3], atol=.75)
    assert match_window(data*0,data*0,np.ones(data.shape,bool)).shape == (0,4)
    assert match_window(data,data,np.zeros(data.shape,bool)).shape == (0,4)


def test_grid_elastix_roundtrip_and_original_pixels(tmp_path, monkeypatch):
    inputs, _ = _translation_fixture(tmp_path)
    # A larger valid grid is needed for 32px Elastix windows.
    for path in inputs:
        create_dummy_raster(path,width=192,height=192,count=1,crs='EPSG:3857',transform=(0,2,0,384,0,-2))
    from spectralmatch.joint_coregistration import elastix
    monkeypatch.setattr(elastix,'match_window',lambda a,b,m: np.array([[32.,32.,36.,30.]]))
    infos = [module._read_image_info(n,p) for n,p in zip(('a','b'),inputs)]
    pair, points, _ = module._extract_pair_ties(*infos,'elastix',100,None,False,100,'32px')
    assert pair == ('a','b')
    assert len(points) >= 3
    np.testing.assert_allclose(points[:,2:]-points[:,:2],np.tile([4,-2],(len(points),1)))


def test_elastix_process_executor_is_reused(tmp_path):
    pytest.importorskip('itk')
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from scipy.ndimage import gaussian_filter
    data = gaussian_filter(np.random.default_rng(99).normal(size=(192,192)),3).astype('float32')
    inputs = [str(tmp_path/'lidar.tif'), str(tmp_path/'optical.tif')]
    for path in inputs:
        create_dummy_raster(path,width=192,height=192,count=1,dtype='float32',nodata=-9999,crs='EPSG:3857',transform=(0,1,0,192,0,-1))
        ds = gdal.Open(path,gdal.GA_Update)
        ds.GetRasterBand(1).WriteArray(data)
        ds = None
    outputs = [str(tmp_path/'corrected.tif')]
    ties = str(tmp_path/'saved.json')
    with ProcessPoolExecutor(2,mp_context=get_context('spawn')) as executor:
        joint_coregistration(inputs[1:],outputs,fixed_images=[inputs[0]],
                             tie_grid_spacing=96,tie_search_radius='48px',local_model='none',
                             tie_save_path=ties,executor=executor)
        assert executor.submit(abs,-7).result() == 7
    assert gdal.Open(inputs[0]).GetGeoTransform() == (0,1,0,192,0,-1)
    import json
    assert len(json.loads(open(ties).read())['tie_points'][0]['points']) >= 3
    np.testing.assert_allclose(gdal.Open(outputs[0]).GetGeoTransform(),gdal.Open(inputs[1]).GetGeoTransform(),atol=.05)


def test_fixed_references_cannot_also_be_moving_inputs(tmp_path):
    inputs, _ = _translation_fixture(tmp_path)
    outputs = [str(tmp_path/'first.tif'), str(tmp_path/'second.tif')]
    with pytest.raises(ValueError, match='supplied only through fixed_images'):
        joint_coregistration(inputs,outputs,fixed_images=[inputs[0]])
