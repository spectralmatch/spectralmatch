import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
from osgeo import gdal, osr

from spectralmatch.pif import mask_windows as masks
from spectralmatch.pif import pif
from spectralmatch import utils
from spectralmatch.utils_thread_balance import tile_thread_budget


def raster(path, values):
    h, w = values.shape
    ds = gdal.GetDriverByName('GTiff').Create(str(path), w, h, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((100, 0.5, 0, 200, 0, -0.5))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32605)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(values)
    ds = None
    return str(path)


@pytest.mark.parametrize('radius', [0, 1, 5, 60])
def test_seed_json_matches_square_dilation_and_union(tmp_path, monkeypatch, radius):
    monkeypatch.setattr(masks, 'MAX_PIXELS', 64)
    points = np.array([[0, 0], [6, 7], [8, 9], [18, 22], [19, 23], [-1, 3]])
    expected = np.zeros((19, 23), dtype=np.uint8)
    for row, col in points:
        if 0 <= row < 19 and 0 <= col < 23:
            expected[row, col] = 1
    expected = cv2.dilate(expected, np.ones((2*radius+1, 2*radius+1), dtype=np.uint8))
    path = pif._build_seed_mask_raster(points, 23, 19, (0,1,0,19,0,-1), '', radius, str(tmp_path))
    ds = gdal.Open(path)
    np.testing.assert_array_equal(ds.ReadAsArray(), expected)
    ds = None
    records = json.loads((tmp_path/'seed_windows.json').read_text())
    assert len(records) == 5  # header and four in-bounds seeds
    restored = masks.load_mask_windows(tmp_path/'seed_windows.json', tmp_path/'restored.tif')
    ds = gdal.Open(restored)
    np.testing.assert_array_equal(ds.ReadAsArray(), expected)


@pytest.mark.parametrize('width', [17, 77])
@pytest.mark.parametrize('limit', [1, 2, 7, 53, 300, 5000])
def test_sample_matches_original_row_major_linspace(tmp_path, monkeypatch, width, limit):
    monkeypatch.setattr(masks, 'MAX_PIXELS', 64)
    values = (np.random.default_rng(32).random((55, width)) < .6).astype('uint8')
    source = raster(tmp_path/'source.tif', values)
    original = gdal.Band.ReadAsArray
    reads = []

    def bounded(band, xoff=0, yoff=0, win_xsize=None, win_ysize=None, *args, **kwargs):
        assert win_xsize is not None and win_ysize is not None
        assert win_xsize * win_ysize <= 64
        reads.append((win_xsize, win_ysize))
        return original(band, xoff, yoff, win_xsize, win_ysize, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(gdal.Band, 'ReadAsArray', bounded)
        output = masks.sample_mask(source, limit, tmp_path/'selected.json', tmp_path/'selected.tif')
    expected = np.zeros(values.size, dtype='uint8')
    indices = np.flatnonzero(values)
    positions = np.linspace(0, len(indices)-1, limit, dtype=int) if len(indices) > limit else np.arange(len(indices))
    expected[indices[positions]] = 1
    ds = gdal.Open(output)
    np.testing.assert_array_equal(ds.ReadAsArray(), expected.reshape(values.shape))
    assert reads
    if len(indices) > limit:
        assert json.loads((tmp_path/'selected.json').read_text())[0]['width'] == width


def test_arbitrary_binary_mask_json_roundtrip_and_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(masks, 'MAX_PIXELS', 64)
    data = np.zeros((45, 37), dtype='uint8')
    data[::3, ::5] = 1
    source = raster(tmp_path/'source.tif', data)
    saved = masks.save_mask_windows(source, tmp_path/'mask.json')
    # Normal JSON, including when another application reformats it.
    with open(saved) as stream:
        records = json.load(stream)
    with open(saved, 'w') as stream:
        json.dump(records, stream, indent=2)
    output = masks.load_mask_windows(saved, tmp_path/'restored.tif')
    a, b = gdal.Open(source), gdal.Open(output)
    np.testing.assert_array_equal(a.ReadAsArray(), b.ReadAsArray())
    assert a.GetGeoTransform() == b.GetGeoTransform()
    assert a.GetProjectionRef() == b.GetProjectionRef()
    a = b = None
    with open(saved, 'rb') as stream:
        truncated = stream.read()[:-20]
    (tmp_path/'broken.json').write_bytes(truncated)
    with pytest.raises(ValueError, match='JSON'):
        masks.load_mask_windows(tmp_path/'broken.json', tmp_path/'broken.tif')


@pytest.mark.parametrize('threads, expected', [(None, 1), (1, 1), (3, 3), ('cpu', os.cpu_count())])
def test_masked_vrt_respects_tile_threads(tmp_path, threads, expected):
    source = raster(tmp_path/'source.tif', np.ones((5, 7), dtype='uint8'))
    path = utils._create_masked_vrt('test', source, out_dir=str(tmp_path), tile_threads=threads)
    text = open(path).read()
    assert f'<Option name="NUM_THREADS">{expected}</Option>' in text
    assert 'ALL_CPUS' not in text


def test_masked_vrt_uses_balanced_budget(tmp_path):
    class Budget:
        def tile_threads_for_jobs(self, value):
            assert value == 2
            return 5
    source = raster(tmp_path/'source.tif', np.ones((5, 7), dtype='uint8'))
    with tile_thread_budget(Budget()):
        path = utils._create_masked_vrt('test', source, out_dir=str(tmp_path), tile_threads=2)
    assert '<Option name="NUM_THREADS">5</Option>' in open(path).read()


def test_large_seed_grid_under_memory_limit(tmp_path):
    if not sys.platform.startswith('linux'):
        pytest.skip('Requires Linux address-space accounting')
    code = r'''
import os, resource, sys
from pathlib import Path
import numpy as np
from osgeo import gdal
from spectralmatch.pif.mask_windows import seed_mask
gdal.SetCacheMax(16 * 1024**2)
base = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')
resource.setrlimit(resource.RLIMIT_AS, (base + 128*1024**2, base + 128*1024**2))
folder = Path(sys.argv[1])
seed_mask(np.array([[5,5], [29995,29995]]), 30000, 30000, (0,1,0,30000,0,-1), '', 30,
          folder/'large.json', folder/'large.tif')
ds = gdal.Open(str(folder/'large.tif'))
assert ds.GetRasterBand(1).ReadAsArray(0,0,10,10).all()
assert not ds.GetRasterBand(1).ReadAsArray(10000,10000,10,10).any()
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_disabled_vrt_tile_threads_stay_serial_under_budget(tmp_path):
    class Budget:
        def tile_threads_for_jobs(self, value):
            return None if value is None else 8
    source = raster(tmp_path/'source.tif', np.ones((5, 7), dtype='uint8'))
    with tile_thread_budget(Budget()):
        path = utils._create_masked_vrt('test', source, out_dir=str(tmp_path), tile_threads=None)
    assert '<Option name="NUM_THREADS">1</Option>' in open(path).read()
