import importlib
import json
import numpy as np
import pytest
from osgeo import gdal, osr
from spectralmatch import Match


match = importlib.import_module('spectralmatch.match.match')
pif = importlib.import_module('spectralmatch.pif.pif')
joint = importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')

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





@pytest.mark.parametrize('save', [False, True])
def test_overlap_pif_only_calculates_needed_preliminary_statistics(tmp_path, monkeypatch, save):
    paths = rasters(tmp_path)
    calls = []
    whole = match._whole_stats_process_image
    pair_stats = pif._calculate_pair_pif_stats
    warps = []
    original_warp = pif._coregister_overlap
    def warp(*args, **kwargs):
        warps.append(kwargs.get('output_window'))
        return original_warp(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('PIF does not consume preliminary overlap statistics')
    def whole_stats(*args, **kwargs):
        calls.append(args[4])
        return whole(*args, **kwargs)
    monkeypatch.setattr(match, '_overlap_stats_process_image', forbidden)
    monkeypatch.setattr(match, '_whole_stats_process_image', whole_stats if save else forbidden)
    monkeypatch.setattr(pif, '_coregister_overlap', warp)
    outputs = [str(tmp_path/f'{name}_global.tif') for name in ('A','B')]
    result = Match.global_regression(
        paths, outputs, pif_load_ties=ties(tmp_path), tie_sampling_method='grid',
        pif_stats_scope='overlap', pif_inz_threshold=10, estimate_stats=False, pif_coregister=True,
        save_adjustments=str(tmp_path/'adjustments.json') if save else None,
    )
    assert result == outputs
    assert warps and all(window is None for window in warps)  # Whole-overlap semantics retained.
    if save:
        assert sorted(calls) == ['A','B']
        saved = json.loads((tmp_path/'adjustments.json').read_text())
        ds = gdal.Open(paths[0])
        values = ds.ReadAsArray()
        assert saved['A']['whole_stats']['band_0']['mean'] == pytest.approx(float(values.mean()), rel=1e-6)
        assert saved['A']['overlap_stats']['B']['band_0']['size'] > 0


def test_entire_mode_keeps_both_statistics_passes(tmp_path, monkeypatch):
    paths = rasters(tmp_path)
    calls = []
    for name in ('_whole_stats_process_image', '_overlap_stats_process_image'):
        original = getattr(match, name)
        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(match, name, wrapper)
    Match.global_regression(paths, [str(tmp_path/f'{n}_out.tif') for n in ('A','B')],
                            pif_method='entire', estimate_stats=False)
    assert calls.count('_whole_stats_process_image') == 2
    assert calls.count('_overlap_stats_process_image') == 1
