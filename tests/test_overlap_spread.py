import importlib
import inspect
import itertools

import numpy as np
import pytest
from osgeo import gdal, osr

from spectralmatch import global_regression, joint_coregistration
from spectralmatch.overlap_sampling import (
    InsufficientTieSamples, parse_sample_quota, resolve_sample_limits,
    spread_centers, validate_sampling,
)

joint = importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')


def test_spread_centers_center_then_largest_area_with_stable_order():
    mask = np.ones((9, 17), dtype=bool)
    assert list(itertools.islice(spread_centers(mask), 3)) == [(4, 8), (4, 12), (4, 3)]
    assert list(itertools.islice(spread_centers(mask, (1, 10)), 2)) == [(4, 8), (6, 8)]


def test_spread_preserves_holes_islands_and_exhausts_without_duplicates():
    mask = np.zeros((19, 31), dtype=bool)
    mask[2:9, 3:11] = True
    mask[4:7, 5:9] = False
    mask[13:17, 24:29] = True
    points = list(spread_centers(mask))
    assert points == list(spread_centers(mask))
    assert len(points) == len(set(points)) == mask.sum()
    assert all(mask[row, col] for row, col in points)
    assert any(col >= 24 for _, col in points[:5])


def test_spread_empty_single_pixel_and_memmap(tmp_path):
    mask = np.memmap(tmp_path/'mask.bin', dtype=bool, mode='w+', shape=(10, 10))
    mask[:] = False
    assert list(spread_centers(mask)) == []
    mask[8, 2] = True
    assert list(spread_centers(mask)) == [(8, 2)]
    mask._mmap.close()


@pytest.mark.parametrize('value', [0, -1, True, 1.5, None, '1/0crs', '0/30crs', '1/30m', '1/NaNcrs', '1/1e999crs', 'spam'])
def test_invalid_quotas(value):
    with pytest.raises(ValueError, match='tie_max_samples'):
        parse_sample_quota(value, 'tie_max_samples')


def test_density_square_units_rounding_and_structural_minimum():
    assert resolve_sample_limits(1e6, '1/100crs', '1/500crs', 3) == (100, 4)
    assert resolve_sample_limits(19344.8508, '1/100crs', '1/500crs', 3) == (3, 3)
    assert resolve_sample_limits(19344.8508, '1/100crs', '1/500crs', 1) == (2, 1)
    assert resolve_sample_limits(19344.8508, '1/30crs', '1/500crs', 3) == (22, 3)
    assert resolve_sample_limits(18, '5/3crs', 1) == (10, 1)
    assert resolve_sample_limits(.1, '1/30crs', '1/500crs', 3) == (3, 3)
    assert resolve_sample_limits(1000, 12, 4) == (12, 4)
    with pytest.raises(ValueError, match='below'):
        resolve_sample_limits(1000, 2, 1, 3)
    with pytest.raises(ValueError, match='below'):
        resolve_sample_limits(1e6, 5, '1/30crs')


def test_sampling_validation_and_defaults():
    from spectralmatch import pipeline
    assert inspect.signature(pipeline).parameters['joint_coregistration_tie_sampling_method'].default == 'overlap_spread'
    for prefix in ('joint_coregistration', 'global_regression'):
        assert inspect.signature(pipeline).parameters[prefix + '_tie_max_samples'].default == '1/100crs'
    for f in (joint_coregistration, global_regression):
        params = inspect.signature(f).parameters
        assert params['tie_sampling_method'].default == 'overlap_spread'
        assert params['tie_max_samples'].default == '1/100crs'
        assert params['tie_min_samples'].default == '1/500crs'
    for args in [('bad', 5, 1, None), ('overlap_spread', 1, 2, None),
                 ('overlap_spread', '1/500crs', '1/30crs', None), ('grid', 10, 1, 0)]:
        with pytest.raises(ValueError):
            validate_sampling(*args)


def images(tmp_path, *, constant=False, empty=False):
    from scipy.ndimage import gaussian_filter
    data = gaussian_filter(np.random.default_rng(41).normal(size=(256, 256)), 1).astype('float32') * 100 + 1000
    if constant:
        data[:] = 1000
    valid = np.zeros(data.shape, dtype=bool)
    valid[40:240, 35:240] = True
    valid[120:145, 90:120] = False
    if empty:
        valid[:] = False
    data[~valid] = -9999
    srs = osr.SpatialReference();srs.ImportFromEPSG(3857)
    infos = []
    for name in ('a', 'b'):
        path = str(tmp_path/f'{name}.tif')
        ds = gdal.GetDriverByName('GTiff').Create(path, 256, 256, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((0, 1, 0, 256, 0, -1));ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).SetNoDataValue(-9999);ds.GetRasterBand(1).WriteArray(data);ds=None
        infos.append(joint._read_image_info(name, path))
    return infos, valid


def extract(infos, **kwargs):
    options = dict(sampling_method='overlap_spread', max_samples=5, min_samples=3,
                   max_attempts=40, search_radius='64px', tie_orb_max_features=500)
    options.update(kwargs)
    return joint._extract_pair_ties(*infos, 'orb', None, None, False, **options)[1]


def test_real_orb_reaches_accepted_goal_and_all_endpoints_are_valid(tmp_path):
    infos, valid = images(tmp_path)
    points = extract(infos)
    assert len(points) == 5
    assert len(np.unique(points.round(6), axis=0)) == 5
    for col, row, col2, row2 in points:
        assert valid[round(row), round(col)] and valid[round(row2), round(col2)]
    np.testing.assert_allclose(points[:, :2], points[:, 2:], atol=1)


def test_no_features_stops_at_attempt_limit_and_can_skip(tmp_path, capsys):
    infos, _ = images(tmp_path, constant=True)
    with pytest.raises(InsufficientTieSamples, match='0 found, 3 required.*attempt limit 4'):
        extract(infos, max_attempts=4)
    assert extract(infos, max_attempts=4, skip_overlap_when_not_enough_matches=True).shape == (0, 4)
    assert 'WARNING: Skipping overlap' in capsys.readouterr().out


@pytest.mark.parametrize('skip', [False, True])
def test_no_shared_valid_pixels_skip_without_enabling_match_failure_skip(tmp_path, capsys, skip):
    infos, _ = images(tmp_path, empty=True)
    assert extract(infos, skip_overlap_when_not_enough_matches=skip) is None
    assert 'no shared valid pixels' in capsys.readouterr().out


def test_cached_pairs_obey_validity_count_and_cap(tmp_path):
    infos, _ = images(tmp_path)
    points = np.array([[50, 50, 50, 50], [180, 50, 180, 50], [50, 200, 50, 200],
                       [200, 200, 200, 200], [0, 0, 0, 0], [50, 50, 50, 50]], dtype=float)
    selected = extract(infos, source_ties=points, max_samples=3)
    assert len(selected) == 3
    with pytest.raises(InsufficientTieSamples, match='2 found, 3 required'):
        extract(infos, source_ties=points[:2])


def test_alpha_is_honored_for_signed_integer_rasters(tmp_path):
    # GDAL may mark mask,1 ALL_VALID for a signed alpha band.
    ds = gdal.GetDriverByName('MEM').Create('', 12, 12, 2, gdal.GDT_Int16)
    ds.GetRasterBand(1).Fill(100)
    alpha = np.zeros((12, 12), dtype='int16');alpha[3:8, 2:10] = 32767
    ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_AlphaBand)
    ds.GetRasterBand(2).WriteArray(alpha)
    valid = joint._build_overlap_valid_mask(ds, ds, reference_bands=(1,), sensed_bands=(1,))
    np.testing.assert_array_equal(valid, alpha > 0)


def test_read_errors_are_not_misreported_as_insufficient_matches(tmp_path, monkeypatch):
    infos, _ = images(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError('Cannot read raster block')
    monkeypatch.setattr(joint, '_build_pair_overlap_vrts', fail)
    with pytest.raises(RuntimeError, match='Cannot read raster block'):
        extract(infos, skip_overlap_when_not_enough_matches=True)


def test_pif_mask_intersects_signed_alpha_not_only_implicit_masks(tmp_path):
    from spectralmatch.pif import pif
    paths=[]
    for name, region in [('a', (slice(1, 7), slice(2, 9))), ('b', (slice(4, 10), slice(5, 11)))]:
        path=str(tmp_path/f'{name}.tif')
        ds=gdal.GetDriverByName('GTiff').Create(path,12,12,2,gdal.GDT_Int16)
        ds.SetGeoTransform((0,1,0,12,0,-1));ds.GetRasterBand(1).Fill(100)
        alpha=np.zeros((12,12),dtype='int16');alpha[region]=32767
        ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_AlphaBand)
        ds.GetRasterBand(2).WriteArray(alpha);ds=None;paths.append(path)
    output=pif._build_valid_mask_raster(*paths,12,12,(0,1,0,12,0,-1),'',str(tmp_path))
    ds=gdal.Open(output);actual=ds.ReadAsArray()>0
    expected=np.zeros((12,12),dtype=bool);expected[4:7,5:9]=True
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('step', ['joint_coregistration','global_regression'])
def test_pipeline_forwards_placement_and_pair_quotas(tmp_path,monkeypatch,step):
    from spectralmatch import chain
    infos,_=images(tmp_path);calls=[]
    def capture(**kwargs):
        calls.append(kwargs);return ['result.tif']
    if step=='joint_coregistration':
        monkeypatch.setattr(chain,'joint_coregistration',capture)
    else:
        monkeypatch.setattr(chain.Match,'global_regression',capture)
    supplied={f'{step}_tie_sampling_method':'overlap_spread',f'{step}_tie_max_samples':'5/3crs',
              f'{step}_tie_min_samples':3,f'{step}_tie_max_attempts':19}
    chain.pipeline(shared_input_images=[i.path for i in infos],shared_output_image_path=str(tmp_path/'out'),
                   shared_temp_dir=str(tmp_path/'work'),steps=(step,),**supplied)
    assert len(calls)==1
    for name,value in supplied.items():
        assert calls[0][name.removeprefix(step+'_')]==value


def test_search_keeps_trying_past_minimum_and_accepts_short_of_goal_at_budget(tmp_path,monkeypatch):
    infos,valid=images(tmp_path)
    attempted=[]
    def at_center(reference,sensed,*args,centers,**kwargs):
        x,y=centers[0];gt=reference.GetGeoTransform()
        col=(x-gt[0])/gt[1]-.5;row=(y-gt[3])/gt[5]-.5
        attempted.append((row,col))
        return np.array([[col,row,col,row]]),np.array([[x,y]])
    monkeypatch.setattr(joint,'_extract_grid_pixel_matches',at_center)
    selected=extract(infos,max_samples=20,min_samples=3,max_attempts=8)
    assert len(attempted)==8
    assert 3<=len(selected)<20
    assert all(valid[int(row),int(col)] for row,col in attempted)


def test_joint_alignment_dispatches_to_spread_by_default(tmp_path, monkeypatch):
    infos, _ = images(tmp_path)
    original = joint._extract_pair_ties
    methods = []
    def record(*args, **kwargs):
        methods.append(args[11])
        return original(*args, **kwargs)
    monkeypatch.setattr(joint, '_extract_pair_ties', record)
    outputs = [str(tmp_path / (info.name + '_aligned.tif')) for info in infos]
    result = joint_coregistration(
        [info.path for info in infos], outputs, local_model='none',
        tie_max_samples=5, tie_min_samples=3, tie_max_attempts=40,
        tie_search_radius='64px', tie_orb_max_features=500,
    )
    assert methods == ['overlap_spread']
    assert result == outputs
    for output in outputs:
        with gdal.Open(output) as dataset:
            assert dataset.RasterCount == 1
            assert dataset.GetRasterBand(1).GetMaskBand().ReadAsArray().any()
