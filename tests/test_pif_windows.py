import importlib
import json

import numpy as np
import pytest
from osgeo import gdal, osr

from spectralmatch import Match
from spectralmatch.pif import pif
from spectralmatch.types_and_validation import Match as MatchValidation

joint = importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')
match = importlib.import_module('spectralmatch.match.match')


def rasters(tmp_path, size=128):
    rng = np.random.default_rng(17)
    values = rng.uniform(50, 200, (size, size)).astype('float32')
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    paths = []
    for name, data in [('A', values), ('B', values * 1.1 + 7)]:
        path = str(tmp_path / f'{name}.tif')
        ds = gdal.GetDriverByName('GTiff').Create(path, size, size, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((0, 1, 0, size, 0, -1))
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).SetNoDataValue(0)
        ds.GetRasterBand(1).WriteArray(data)
        ds = None
        paths.append(path)
    return paths


def ties(tmp_path):
    path = tmp_path / 'ties.json'
    points = [[x, y, x, y] for x, y in [(30, 30), (90, 30), (30, 90), (90, 90)]]
    joint._save_ties(str(path), {('A', 'B'): np.asarray(points, dtype=float)})
    return str(path)


@pytest.mark.parametrize('scope', ['overlap', 'matched_windows'])
@pytest.mark.parametrize('estimate', [False, True])
def test_gdal_statistics_flag_reaches_difference_and_pif_statistics(tmp_path, monkeypatch, scope, estimate):
    paths = rasters(tmp_path)
    flags = []
    original = gdal.Band.GetStatistics

    def statistics(band, approx_ok, force, *args):
        flags.append(approx_ok)
        return original(band, approx_ok, force, *args)

    monkeypatch.setattr(gdal.Band, 'GetStatistics', statistics)
    parameters = pif.Pif.flood_from_match_points(
        paths, load_ties=ties(tmp_path), stats_scope=scope, window_size=33,
        estimate_stats=estimate, inz_threshold=10, region_radius=5,
    )
    assert parameters.shape == (1, 4, 1)
    assert np.isfinite(parameters).all()
    assert len(flags) >= 3  # Difference mean/std, reference PIF and sensed PIF.
    assert set(flags) == {int(estimate)}


def test_local_mode_never_materializes_whole_overlap_or_reads_preliminary_statistics(tmp_path, monkeypatch):
    paths = rasters(tmp_path)
    outputs = [str(tmp_path / f'{name}_global.tif') for name in ['A', 'B']]
    original = pif._coregister_overlap
    windows = []

    def bounded_warp(*args, **kwargs):
        window = kwargs.get('output_window')
        assert window is not None
        assert window[2] <= 33 and window[3] <= 33
        windows.append(window)
        return original(*args, **kwargs)

    def unexpected(*args, **kwargs):
        raise AssertionError('Full-image statistics or full-overlap mask were requested')

    monkeypatch.setattr(pif, '_coregister_overlap', bounded_warp)
    monkeypatch.setattr(joint, '_build_overlap_valid_mask', unexpected)
    monkeypatch.setattr(match, '_whole_stats_process_image', unexpected)
    monkeypatch.setattr(match, '_overlap_stats_process_image', unexpected)
    model = tmp_path / 'adjustments.json'
    result = Match.global_regression(
        paths, outputs, pif_load_ties=ties(tmp_path), pif_stats_scope='matched_windows',
        pif_window_size=33, estimate_stats=False, pif_inz_threshold=10,
        save_adjustments=str(model),
    )
    assert result == outputs
    assert len(windows) == 4
    for output in outputs:
        assert gdal.Open(output).RasterXSize == 128
    saved = json.loads(model.read_text())
    assert set(saved) == {'A', 'B'}
    assert saved['A']['whole_stats']['band_0']['size'] > 0
    assert saved['A']['overlap_stats']['B']['band_0']['size'] > 0


def test_matching_uses_joint_distributed_search_with_its_options(tmp_path, monkeypatch):
    paths = rasters(tmp_path, size=384)
    calls = []
    original = joint._extract_pair_ties

    def distributed(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    def unexpected(*args, **kwargs):
        raise AssertionError('Legacy whole-overlap feature extraction was called')

    monkeypatch.setattr(joint, '_extract_pair_ties', distributed)
    monkeypatch.setattr(joint, '_extract_overlap_pixel_matches', unexpected)
    result = pif.Pif.flood_from_match_points(
        paths, stats_scope='matched_windows', window_size=33,
        tie_grid_spacing=96, tie_search_radius='64px', tie_orb_max_features=500,
        tie_max_matches_per_window=1, inz_threshold=10,
    )
    assert result.shape == (1, 4, 1)
    assert len(calls) == 1
    assert calls[0][1]['grid_spacing'] == 96
    assert calls[0][1]['search_radius'] == '64px'
    assert calls[0][1]['tie_orb_max_features'] == 500


def test_window_stack_deduplicates_pixels_and_gdal_pools_between_window_variance(tmp_path):
    paths = rasters(tmp_path)
    # Identity warp with overlapping windows and distinct points.
    points = [(30, 30, 30, 30), (30, 40, 30, 40), (80, 30, 80, 30)]
    ref, sensed, width, height, gt, projection, seeds = pif._build_matched_window_stack(
        *paths, points, 1, 21, str(tmp_path),
    )
    ds = gdal.Open(ref)
    valid = ds.GetRasterBand(1).GetMaskBand().ReadAsArray() > 0
    values = ds.GetRasterBand(1).ReadAsArray()[valid]
    assert valid.sum() == 3 * 21 * 21 - 11 * 21
    mask = pif._build_valid_mask_raster(ref, sensed, width, height, gt, projection, str(tmp_path))
    stats = pif._masked_band_stats(ref, 1, mask, estimate_stats=False)
    assert stats['mean'] == pytest.approx(values.mean(), rel=1e-6)
    assert stats['std'] == pytest.approx(values.std(), rel=1e-6)
    seed_mask = pif._read_mask(seeds)
    # All three seed neighborhoods survive; the two close seeds share one column.
    assert np.count_nonzero(seed_mask & valid) == 3 * 11 * 11 - 11
    assert not valid[21].any()  # Separator between unrelated patches.


def test_window_scope_and_size_validation():
    for kwargs in [{'pif_stats_scope': 'unknown'}, {'pif_window_size': 0}, {'pif_window_size': 32}, {'pif_window_size': True}]:
        with pytest.raises(ValueError):
            MatchValidation._validate_global_regression(**kwargs)


@pytest.mark.parametrize('estimate,error,expected_calls', [
    (True, 'Failed to compute statistics, no valid pixels found in sampling.', [1, 0]),
    (False, 'Failed to compute statistics, no valid pixels found in sampling.', [0]),
    (True, 'Failed to read source raster block.', [1]),
])
def test_selected_pif_statistics_retry_only_empty_approximate_sampling(tmp_path, monkeypatch, estimate, error, expected_calls):
    paths = rasters(tmp_path)
    mask_path = str(tmp_path / 'pif_mask.tif')
    mask = gdal.GetDriverByName('GTiff').Create(mask_path, 128, 128, 1, gdal.GDT_Byte)
    mask.GetRasterBand(1).Fill(1)
    mask = None
    original = gdal.Band.GetStatistics
    calls = []

    def statistics(band, approx_ok, force, *args):
        calls.append(approx_ok)
        if len(calls) == 1:
            raise RuntimeError(error)
        return original(band, approx_ok, force, *args)

    monkeypatch.setattr(gdal.Band, 'GetStatistics', statistics)
    if expected_calls == [1, 0]:
        result = pif._masked_band_stats(paths[0], 1, mask_path, estimate_stats=estimate)
        source = gdal.Open(paths[0])
        values = source.ReadAsArray()
        assert result['mean'] == pytest.approx(values.mean(), rel=1e-6)
        assert result['std'] == pytest.approx(values.std(), rel=1e-6)
        assert result['size'] == values.size
    else:
        with pytest.raises(RuntimeError, match=error):
            pif._masked_band_stats(paths[0], 1, mask_path, estimate_stats=estimate)
    assert calls == expected_calls


def test_difference_statistics_do_not_retry_exact_sampling(tmp_path, monkeypatch):
    paths = rasters(tmp_path)
    calls = []

    def statistics(band, approx_ok, force, *args):
        calls.append(approx_ok)
        raise RuntimeError('Failed to compute statistics, no valid pixels found in sampling.')

    monkeypatch.setattr(gdal.Band, 'GetStatistics', statistics)
    with pytest.raises(RuntimeError, match='no valid pixels found in sampling'):
        pif.Pif.flood_from_match_points(
            paths, load_ties=ties(tmp_path), stats_scope='matched_windows',
            window_size=33, estimate_stats=True,
        )
    assert calls == [1]
