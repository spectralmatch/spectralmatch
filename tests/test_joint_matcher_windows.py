"""Both matchers share bounded, single-read band and mask windows."""
from collections import Counter
from dataclasses import replace
import importlib
import inspect

import numpy as np
import pytest
from osgeo import gdal

from spectralmatch import joint_coregistration
from .utils_test import create_dummy_raster
from .test_joint_coregistration import _translation_fixture

module = importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')


@pytest.mark.parametrize('method', ['orb', 'elastix'])
def test_selected_bands_and_shared_masks_are_read_once_per_window(tmp_path, monkeypatch, method):
    reads = []
    captures = []

    class Band:
        def __init__(self, band, role, index, mask=False):
            self.band, self.role, self.index, self.mask = band, role, index, mask
        def __getattr__(self, name):
            return getattr(self.band, name)
        def GetMaskBand(self):
            return Band(self.band.GetMaskBand(), self.role, self.index, True)
        def ReadAsArray(self, *window):
            assert len(window) == 4
            reads.append((self.role, self.mask, self.index, window))
            return self.band.ReadAsArray(*window)

    class Dataset:
        def __init__(self, dataset, role):
            self.dataset, self.role = dataset, role
        def __getattr__(self, name):
            return getattr(self.dataset, name)
        def GetRasterBand(self, index):
            return Band(self.dataset.GetRasterBand(index), self.role, index)

    paths = [str(tmp_path/f'{name}.tif') for name in ('reference','moving')]
    selected = [(2,4), (3,)]
    for path, bands in zip(paths, selected):
        create_dummy_raster(path, width=192, height=192, count=5, dtype='float32',
                            crs='EPSG:3857', transform=(0,1,0,192,0,-1), nodata=-9999)
        with gdal.Open(path, gdal.GA_Update) as ds:
            # Real shared dataset mask exercises reuse across selected bands.
            ds.CreateMaskBand(gdal.GMF_PER_DATASET)
            mask = np.full((192,192),255,dtype='uint8'); mask[70,70]=0
            ds.GetRasterBand(1).GetMaskBand().WriteArray(mask)
            yy,xx=np.mgrid[:192,:192]
            for band in bands:
                values=(xx**2+yy*band).astype('float32')
                values[75,75] = np.nan if band == bands[0] else -9999
                ds.GetRasterBand(band).WriteArray(values)
    infos = [replace(module._read_image_info(name,path),matching_bands=bands)
             for name,path,bands in zip(('reference','moving'),paths,selected)]
    datasets = [Dataset(gdal.Open(path), role) for role,path in zip(('reference','moving'),paths)]

    def match(reference, sensed, valid, *args):
        captures.append((reference.copy(),sensed.copy(),valid.copy()))
        assert np.isfinite(reference).all() and np.isfinite(sensed).all()
        # Return new storage, just as the real matchers do.
        y,x = np.argwhere(valid)[len(np.argwhere(valid))//2]
        return np.array([[x,y,x,y]],dtype=float)
    monkeypatch.setattr(module,'_match_orb_windows',match)
    from spectralmatch.joint_coregistration import elastix
    monkeypatch.setattr(elastix,'match_window',match)
    points,_ = module._extract_grid_pixel_matches(*datasets,*infos,96,'48px',method)
    assert len(points) == len(captures) == 4
    data_reads = Counter((role,index,window) for role,mask,index,window in reads if not mask)
    assert len(data_reads) == 4*3 and set(data_reads.values()) == {1}
    assert {(role,index) for role,index,window in data_reads} == {
        ('reference',2),('reference',4),('moving',3)}
    mask_reads = Counter((role,window) for role,mask,index,window in reads if mask)
    assert len(mask_reads) == 4*2 and set(mask_reads.values()) == {1}
    for reference,sensed,valid in captures:
        assert reference.dtype == sensed.dtype == np.uint8
        assert np.all(reference[~valid] == 0) and np.all(sensed[~valid] == 0)
    # The first window covers these source pixels and must exclude both kinds.
    assert not captures[0][2][70,70]
    assert not captures[0][2][75,75]


@pytest.mark.parametrize('name', ['moving_tie_method','fixed_tie_method'])
@pytest.mark.parametrize('invalid', [None,'unknown','ORB',1])
def test_both_matcher_parameters_validate_options(tmp_path, name, invalid):
    paths,_ = _translation_fixture(tmp_path)
    with pytest.raises(ValueError,match=name+' must be orb or elastix'):
        joint_coregistration(paths[1:],[str(tmp_path/'out.tif')],fixed_images=[paths[0]],
                             **{name:invalid})


def test_public_matcher_defaults_and_old_names_are_removed():
    parameters = inspect.signature(joint_coregistration).parameters
    assert parameters['moving_tie_method'].default == 'orb'
    assert parameters['fixed_tie_method'].default == 'elastix'
    assert 'tie_feature_method' not in parameters
    assert 'tie_fixed_feature_method' not in parameters
