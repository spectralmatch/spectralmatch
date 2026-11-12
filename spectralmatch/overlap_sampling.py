"""Area-based tie quotas and deterministic subdivision of shared valid pixels."""
import heapq
import itertools
import math
import re

import numpy as np


class InsufficientTieSamples(ValueError):
    """A readable overlap could not supply the required accepted matched pairs."""


def parse_sample_quota(value, name):
    """Return an absolute count or (count, square side length) density."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    number = r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
    match = re.fullmatch(rf'({number})/({number})crs', value.strip()) if isinstance(value, str) else None
    if match:
        count, side = map(float, match.groups())
        if all(math.isfinite(v) and v > 0 for v in (count, side)):
            density = count / side / side
            if math.isfinite(density) and density > 0:
                return count, side
    raise ValueError(f'{name} must be a positive integer or a density such as "1/30crs" (one pair per 30 x 30 CRS-unit square).')


def validate_sampling(method, maximum, minimum, attempts):
    if method not in {'grid', 'overlap_spread'}:
        raise ValueError("tie_sampling_method must be 'grid' or 'overlap_spread'.")
    high = parse_sample_quota(maximum, 'tie_max_samples')
    low = parse_sample_quota(minimum, 'tie_min_samples')
    if isinstance(high, int) and isinstance(low, int) and high < low:
        raise ValueError('tie_max_samples must be >= tie_min_samples.')
    if isinstance(high, tuple) and isinstance(low, tuple) and high[0] / high[1] / high[1] < low[0] / low[1] / low[1]:
        raise ValueError('tie_max_samples density must be >= tie_min_samples density.')
    if attempts is not None and (isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1):
        raise ValueError('tie_max_attempts must be a positive integer or None.')


def resolve_sample_limits(area, maximum, minimum, required=1):
    """Round densities upward; preserve the caller's geometric minimum (PIF: 3)."""
    if not math.isfinite(area) or area < 0:
        raise ValueError('Valid overlap area must be finite and nonnegative.')
    values = []
    for value, name in ((maximum, 'tie_max_samples'), (minimum, 'tie_min_samples')):
        parsed = parse_sample_quota(value, name)
        if isinstance(parsed, tuple):
            resolved = area * (parsed[0] / parsed[1] / parsed[1])
            if not math.isfinite(resolved):
                raise ValueError(f'{name} resolves to a nonfinite count.')
            values.append(max(required, math.ceil(resolved)))
        else:
            values.append(parsed)
    goal, minimum = values
    minimum = max(required, minimum)
    if goal < minimum:
        raise ValueError(f'tie_max_samples resolves to {goal}, below the required minimum of {minimum} matched pairs.')
    return goal, minimum


def _row_blocks(bounds):
    x0, y0, x1, y1 = bounds
    step = max(1, min(256, 1048576 // max(1, x1 - x0)))
    for y in range(y0, y1, step):
        yield y, min(y + step, y1)


def _occupied_cell(mask, bounds):
    x0, y0, x1, y1 = bounds
    count = 0
    left, top, right, bottom = x1, y1, x0, y0
    for y, end in _row_blocks(bounds):
        block = mask[y:end, x0:x1]
        rows = np.flatnonzero(np.any(block, axis=1))
        if not rows.size:
            continue
        columns = np.flatnonzero(np.any(block, axis=0))
        count += int(np.count_nonzero(block))
        left, right = min(left, x0 + int(columns[0])), max(right, x0 + int(columns[-1]) + 1)
        top, bottom = min(top, y + int(rows[0])), max(bottom, y + int(rows[-1]) + 1)
    return count, (left, top, right, bottom)


def _nearest_valid_center(mask, bounds, pixel_size):
    x0, y0, x1, y1 = bounds
    cx, cy = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
    dx = ((np.arange(x0, x1) - cx) * pixel_size[0]) ** 2
    best, point = float('inf'), None
    for y, end in _row_blocks(bounds):
        distances = dx[None, :] + ((np.arange(y, end) - cy) * pixel_size[1])[:, None] ** 2
        distances[~mask[y:end, x0:x1]] = np.inf
        index = int(np.argmin(distances))
        distance = float(distances.flat[index])
        if distance < best:
            row, col = np.unravel_index(index, distances.shape)
            best, point = distance, (y + int(row), x0 + int(col))
    return point


def spread_centers(valid_mask, pixel_size=(1.0, 1.0)):
    """Center first, then bisect the region containing the largest valid area.

    Each split follows the longest CRS dimension. Snap each region's center
    into its valid pixels; never yield a pixel twice. Equal areas use insertion
    order and equal distances use row/column order. Disk-backed boolean masks
    are supported, and temporary array memory is bounded by row blocks.
    """
    if valid_mask.ndim != 2 or valid_mask.dtype != np.bool_:
        raise ValueError('valid_mask must be a two-dimensional boolean array.')
    if len(pixel_size) != 2 or any(not math.isfinite(v) or v <= 0 for v in pixel_size):
        raise ValueError('pixel_size must contain two positive finite lengths.')
    serial = itertools.count()
    count, bounds = _occupied_cell(valid_mask, (0, 0, valid_mask.shape[1], valid_mask.shape[0]))
    heap = [(-count, next(serial), bounds)] if count else []
    seen = set()
    while heap:
        _, _, bounds = heapq.heappop(heap)
        point = _nearest_valid_center(valid_mask, bounds, pixel_size)
        if point not in seen:
            seen.add(point)
            yield point
        x0, y0, x1, y1 = bounds
        if x1 - x0 == 1 and y1 - y0 == 1:
            continue
        if x1 - x0 > 1 and (y1 - y0 == 1 or (x1 - x0) * pixel_size[0] >= (y1 - y0) * pixel_size[1]):
            middle = (x0 + x1) // 2
            children = ((x0, y0, middle, y1), (middle, y0, x1, y1))
        else:
            middle = (y0 + y1) // 2
            children = ((x0, y0, x1, middle), (x0, middle, x1, y1))
        for child in children:
            count, occupied = _occupied_cell(valid_mask, child)
            if count:
                heapq.heappush(heap, (-count, next(serial), occupied))
