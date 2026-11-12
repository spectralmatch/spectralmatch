"""Opt-in pair exclusion must not remove images or hide unrelated failures."""
import importlib
import inspect
import json

import numpy as np
import pytest
from osgeo import gdal, osr

from spectralmatch import Match, joint_coregistration
from spectralmatch.pif import pif

joint = importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')


@pytest.fixture
def images(tmp_path):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    values = np.random.default_rng(51).uniform(50, 200, (128, 128)).astype('float32')
    paths = []
    for index, name in enumerate(('A', 'B', 'C')):
        path = str(tmp_path / f'{name}.tif')
        ds = gdal.GetDriverByName('GTiff').Create(path, 128, 128, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((0, 1, 0, 128, 0, -1))
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).SetNoDataValue(0)
        ds.GetRasterBand(1).WriteArray(values * (1 + .1 * index) + index * 7)
        ds = None
        paths.append(path)
    return paths


def selected_ties():
    return np.asarray([[30, 30, 30, 30], [90, 30, 90, 30],
                       [30, 90, 30, 90], [90, 90, 90, 90]], dtype=float)


def pair_arguments(images):
    return dict(reference_path=images[0], sensed_path=images[1],
                reference_name='A', sensed_name='B', num_bands=1, nodata_value=0,
                calculation_dtype='float32', red_band_index=None, nir_band_index=None,
                vegetation_threshold=.2, inz_threshold=10, region_radius=5,
                max_samples=1000, min_samples=10, feature_method='orb',
                source_ties=selected_ties(), cache=None, io_threads=None,
                tile_threads=None, save_inz_path=None, debug_logs=False,
                stats_scope='matched_windows', window_size=33, estimate_stats=False)


@pytest.mark.parametrize('skip', [False, True])
@pytest.mark.parametrize('shortage', ['ties', 'samples', 'band'])
def test_pair_shortages_raise_by_default_or_warn_and_return_no_statistics(
        images, monkeypatch, capsys, skip, shortage):
    args = pair_arguments(images)
    args['skip_overlap_when_not_enough_matches'] = skip
    if shortage == 'ties':
        args['source_ties'] = selected_ties()[:2]
        message = 'fewer than 3 usable points'
    elif shortage == 'samples':
        args['min_samples'] = 100000
        message = 'Not enough flood_from_match_points PIF samples'
    else:
        monkeypatch.setattr(pif, '_masked_band_stats', lambda *a, **kw: dict(mean=1., std=1., size=1))
        message = 'Band 1 has 1 flood_from_match_points PIF samples'
    if skip:
        assert pif._calculate_pair_pif_stats(**args) == ({}, {})
        warning = capsys.readouterr().out
        assert 'WARNING: Skipping overlap' in warning
        assert message in warning
    else:
        with pytest.raises(ValueError, match=message):
            pif._calculate_pair_pif_stats(**args)
        assert 'WARNING: Skipping overlap' not in capsys.readouterr().out


def test_skip_does_not_catch_unrelated_worker_errors(images, monkeypatch):
    def unreadable(*a, **kw):
        raise RuntimeError('Cannot read source raster block')
    monkeypatch.setattr(pif, '_build_overlap_vrts', unreadable)
    with pytest.raises(RuntimeError, match='Cannot read source raster block'):
        pif._calculate_pair_pif_stats(**pair_arguments(images), skip_overlap_when_not_enough_matches=True)


@pytest.mark.parametrize('scope', ['overlap', 'matched_windows'])
def test_global_regression_drops_pairs_from_solver_and_saved_model_but_keeps_images(
        images, tmp_path, monkeypatch, capsys, scope):
    ties = tmp_path / 'ties.json'
    joint._save_ties(str(ties), {pair:selected_ties() for pair in [('A','B'), ('A','C'), ('B','C')]})
    original = pif._calculate_pair_pif_stats
    signature = inspect.signature(original)
    def insufficient_c(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        if 'C' in (bound.arguments['reference_name'], bound.arguments['sensed_name']):
            bound.arguments['min_samples'] = 100000
        return original(*bound.args, **bound.kwargs)
    monkeypatch.setattr(pif, '_calculate_pair_pif_stats', insufficient_c)
    solve = pif._solve_pif_global_model
    def checked_solve(**kwargs):
        assert kwargs['overlapping_pairs'] == (('A', 'B'),)
        assert set(kwargs['all_overlap_stats']) == {'A', 'B'}
        return solve(**kwargs)
    monkeypatch.setattr(pif, '_solve_pif_global_model', checked_solve)
    outputs = [str(tmp_path / f'{name}_global.tif') for name in ('A','B','C')]
    model = tmp_path / 'adjustments.json'
    result = Match.global_regression(
        images, outputs, pif_load_ties=str(ties), pif_stats_scope=scope,
        pif_window_size=33, estimate_stats=False, pif_inz_threshold=10,
        skip_overlap_when_not_enough_matches=True, image_threads=2,
        save_adjustments=str(model),
    )
    assert result == outputs
    saved = json.loads(model.read_text())
    assert set(saved) == {'A','B','C'}
    assert set(saved['A']['overlap_stats']) == {'B'}
    assert set(saved['B']['overlap_stats']) == {'A'}
    assert saved['C']['overlap_stats'] == {}
    assert saved['C']['adjustments']['band_0'] == {'scale':1., 'offset':0.}
    with gdal.Open(outputs[2]) as actual, gdal.Open(images[2]) as source:
        np.testing.assert_array_equal(actual.ReadAsArray(), source.ReadAsArray())
    assert capsys.readouterr().out.count('WARNING: Skipping overlap') == 2


def test_all_skipped_pairs_still_require_a_valid_model(images, tmp_path, capsys):
    ties = tmp_path / 'ties.json'
    joint._save_ties(str(ties), {('A','B'):selected_ties()[:2]})
    with pytest.raises(ValueError, match='No valid flood_from_match_points PIF overlap pairs'):
        pif.Pif.flood_from_match_points(images[:2], load_ties=str(ties),
                                        skip_overlap_when_not_enough_matches=True)
    assert 'WARNING: Skipping overlap' in capsys.readouterr().out


@pytest.mark.parametrize('skip', [False, True])
def test_joint_registration_omits_empty_pairs_from_cache_and_keeps_all_outputs(
        images, tmp_path, monkeypatch, capsys, skip):
    def matches(a, b, *args):
        pair = joint._canonical_pair(a.name, b.name)
        return pair, (np.empty((0, 4)) if 'C' in pair else selected_ties()), 3.
    monkeypatch.setattr(joint, '_extract_pair_ties', matches)
    output = [str(tmp_path / f'{name}_coreg.tif') for name in ('A','B','C')]
    cache = tmp_path / 'selected.json'
    assert joint_coregistration(images, output, tie_save_path=str(cache), local_model='none',
                               skip_overlap_when_not_enough_matches=skip) == output
    cached = joint._load_ties(str(cache))
    assert set(cached) == ({('A','B')} if skip else {('A','B'), ('A','C'), ('B','C')})
    assert all(gdal.Open(path) is not None for path in output)
    assert ('WARNING: Skipping overlap' in capsys.readouterr().out) is skip


def test_empty_cached_joint_pairs_are_omitted_without_recalculating(images, monkeypatch, capsys):
    infos = {name:joint._read_image_info(name, path) for name,path in zip(('A','B'),images)}
    def unexpected(*a, **kw):
        raise AssertionError('Empty cached pair must not be recalculated')
    monkeypatch.setattr(joint, '_extract_pair_ties', unexpected)
    ties, thresholds = joint._collect_ties([('A','B')], infos, {('A','B'):np.empty((0,4))},
                                          'orb', None, None, None, False,
                                          skip_overlap_when_not_enough_matches=True)
    assert ties == {} and thresholds == []
    assert 'WARNING: Skipping overlap' in capsys.readouterr().out


@pytest.mark.parametrize('value', ['true', 1, None])
def test_skip_flag_requires_a_bool(value):
    for function in (joint_coregistration, Match.global_regression):
        with pytest.raises(ValueError, match='skip_overlap_when_not_enough_matches must be a bool'):
            function([], [], skip_overlap_when_not_enough_matches=value)
    with pytest.raises(ValueError, match='skip_overlap_when_not_enough_matches must be a bool'):
        pif.Pif.flood_from_match_points([], skip_overlap_when_not_enough_matches=value)


@pytest.mark.parametrize('no_spatial_overlap', [False, True])
def test_joint_registration_errors_if_no_usable_overlap_remains(images, tmp_path, monkeypatch, no_spatial_overlap):
    if no_spatial_overlap:
        with gdal.Open(images[1], gdal.GA_Update) as ds:
            ds.SetGeoTransform((1000, 1, 0, 128, 0, -1))
    def no_matches(a, b, *args):
        return joint._canonical_pair(a.name, b.name), np.empty((0, 4)), 3.
    monkeypatch.setattr(joint, '_extract_pair_ties', no_matches)
    outputs = [str(tmp_path / f'{name}_coreg.tif') for name in ('A', 'B')]
    with pytest.raises(ValueError, match='No usable tie-point overlap pairs remain'):
        joint_coregistration(images[:2], outputs, skip_overlap_when_not_enough_matches=True)
    assert not any(gdal.VSIStatL(path) for path in outputs)


def _split_valid_areas(paths):
    # Identical bounding boxes, but C shares no valid pixels with A or B.
    for index, path in enumerate(paths):
        with gdal.Open(path, gdal.GA_Update) as dataset:
            values = dataset.ReadAsArray()
            if index < 2:
                values[:, 60:] = 0
            else:
                values[:, :68] = 0
            dataset.GetRasterBand(1).WriteArray(values)
    return np.asarray([[15, 25, 15, 25], [45, 25, 45, 25],
                       [15, 95, 15, 95], [45, 95, 45, 95]], dtype=float)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_empty_valid_pif_overlap_skips_before_match_minimum(images, cached, reverse, capsys):
    points = _split_valid_areas(images)
    args = pair_arguments(images)
    args.update(reference_path=images[0], sensed_path=images[2],
                reference_name='A', sensed_name='C',
                source_ties=points[:1] if cached else None,
                tie_options={'tie_sampling_method':'overlap_spread'})
    if reverse:
        args.update(reference_path=images[2], sensed_path=images[0],
                    reference_name='C', sensed_name='A')
    assert pif._calculate_pair_pif_stats(**args) == ({}, {})
    assert 'no shared valid pixels' in capsys.readouterr().out


@pytest.mark.parametrize('scope', ['overlap', 'matched_windows'])
def test_empty_valid_overlaps_skip_by_default_and_isolated_image_is_preserved(images, tmp_path, scope):
    points = _split_valid_areas(images)
    ties = tmp_path/'split_ties.json'
    joint._save_ties(str(ties), {pair:points for pair in [('A','B'),('A','C'),('B','C')]})
    outputs = [str(tmp_path/f'{name}_matched.tif') for name in ('A','B','C')]
    model = tmp_path/'model.json'
    result = Match.global_regression(
        images, outputs, pif_load_ties=str(ties), pif_stats_scope=scope,
        pif_window_size=33, estimate_stats=False, pif_inz_threshold=10,
        save_adjustments=str(model), image_threads=2,
    )
    assert result == outputs
    saved = json.loads(model.read_text())
    assert set(saved) == {'A','B','C'}
    assert set(saved['A']['overlap_stats']) == {'B'}
    assert set(saved['B']['overlap_stats']) == {'A'}
    assert saved['C']['overlap_stats'] == {}
    assert saved['C']['adjustments']['band_0'] == {'scale':1., 'offset':0.}
    with gdal.Open(outputs[2]) as result_ds, gdal.Open(images[2]) as original:
        np.testing.assert_array_equal(result_ds.ReadAsArray(), original.ReadAsArray())


def test_empty_valid_alignment_pairs_are_omitted_from_saved_graph(images, tmp_path):
    points = _split_valid_areas(images)
    loaded, saved = tmp_path/'loaded.json', tmp_path/'selected.json'
    joint._save_ties(str(loaded), {pair:points for pair in [('A','B'),('A','C'),('B','C')]})
    outputs = [str(tmp_path/f'{name}_aligned.tif') for name in ('A','B','C')]
    assert joint_coregistration(images, outputs, tie_load_path=str(loaded),
                               tie_save_path=str(saved), local_model='none') == outputs
    assert set(joint._load_ties(str(saved))) == {('A','B')}
    assert all(gdal.Open(path) is not None for path in outputs)
    with gdal.Open(outputs[2]) as result_ds, gdal.Open(images[2]) as original:
        np.testing.assert_array_equal(result_ds.ReadAsArray(), original.ReadAsArray())
