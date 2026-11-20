import heapq
import itertools
import numpy as np
import pytest
from spectralmatch.overlap_sampling import spread_centers, _occupied_cell, _nearest_valid_center
from spectralmatch.valid_mask_index import ValidMaskIndex


def legacy_centers(mask, pixel_size):
    serial = itertools.count()
    count, bounds = _occupied_cell(mask, (0, 0, mask.shape[1], mask.shape[0]))
    heap = [(-count, next(serial), bounds)] if count else []
    seen = set()
    while heap:
        _, _, bounds = heapq.heappop(heap)
        point = _nearest_valid_center(mask, bounds, pixel_size)
        if point not in seen:
            seen.add(point)
            yield point
        x0, y0, x1, y1 = bounds
        if x1-x0 == 1 and y1-y0 == 1:
            continue
        if x1-x0 > 1 and (y1-y0 == 1 or (x1-x0)*pixel_size[0] >= (y1-y0)*pixel_size[1]):
            mid = (x0+x1)//2
            children = ((x0,y0,mid,y1),(mid,y0,x1,y1))
        else:
            mid = (y0+y1)//2
            children = ((x0,y0,x1,mid),(x0,mid,x1,y1))
        for child in children:
            count, occupied = _occupied_cell(mask, child)
            if count:
                heapq.heappush(heap, (-count, next(serial), occupied))


@pytest.mark.parametrize('shape', [(1, 149), (151, 1), (19, 31), (173, 227)])
@pytest.mark.parametrize('density', [0, .01, .35, 1])
@pytest.mark.parametrize('pixel_size', [(1, 1), (.3, 1.7)])
def test_exact_sequence_matches_original(shape, density, pixel_size):
    mask = np.random.default_rng(29).random(shape) < density
    old = list(itertools.islice(legacy_centers(mask, pixel_size), 250))
    new = list(itertools.islice(spread_centers(mask, pixel_size), 250))
    assert new == old


def test_index_queries_and_nearest_preserve_holes_and_edge_tiles():
    rng = np.random.default_rng(51)
    mask = rng.random((243, 319)) < .02
    mask[40:200, 50:250] = False
    index = ValidMaskIndex.build(mask)
    for _ in range(100):
        x0, x1 = sorted(rng.choice(mask.shape[1]+1, 2, replace=False))
        y0, y1 = sorted(rng.choice(mask.shape[0]+1, 2, replace=False))
        bounds = tuple(map(int, (x0, y0, x1, y1)))
        count, occupied = _occupied_cell(mask, bounds)
        assert index.count(bounds) == count
        actual_count, actual_bounds = index.occupied(bounds)
        assert actual_count == count
        if count:
            assert actual_bounds == occupied
            assert index.nearest(occupied, (.3, 1.7)) == _nearest_valid_center(mask, occupied, (.3, 1.7))


def test_full_region_queries_do_not_rescan_full_mask():
    class Reads:
        def __init__(self):
            self.data = np.ones((4096,4096), dtype=bool)
            self.shape = self.data.shape
            self.pixels = 0
        def __getitem__(self, window):
            data = self.data[window]
            self.pixels += data.size
            return data
    mask = Reads()
    index = ValidMaskIndex.build(mask)
    mask.pixels = 0
    assert index.occupied((0,0,4096,4096)) == (4096**2, (0,0,4096,4096))
    assert mask.pixels == 0
    assert index.nearest((0,0,4096,4096), (1,1)) == (2047,2047)
    assert mask.pixels <= 4 * index.block_size**2


def test_index_built_from_raster_windows_matches_direct_build():
    mask = np.random.default_rng(3).random((731, 1327)) < .2
    incremental = ValidMaskIndex(mask)
    for y in range(0, mask.shape[0], 512):
        for x in range(0, mask.shape[1], 1024):
            incremental.add_window(x, y, mask[y:y+512, x:x+1024])
    incremental.finish()
    np.testing.assert_array_equal(incremental.prefix, ValidMaskIndex.build(mask).prefix)
    assert incremental.prefix[-1,-1] == np.count_nonzero(mask)
