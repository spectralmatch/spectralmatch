"""Match windows placed by largest-valid-area subdivision."""
import math
import os

import numpy as np
from osgeo import gdal

from ..overlap_sampling import InsufficientTieSamples, resolve_sample_limits, spread_centers
from ..utils_logging import _print_line


def _shared_mask(reference, sensed, info_i, info_j, tmpdir):
    from .joint_coregistration import _build_overlap_valid_mask
    width = min(reference.RasterXSize, sensed.RasterXSize)
    height = min(reference.RasterYSize, sensed.RasterYSize)
    mask = np.memmap(os.path.join(tmpdir, 'shared_valid.bin'), mode='w+', dtype=np.bool_, shape=(height, width))
    count = 0
    for y in range(0, height, 512):
        for x in range(0, width, 1024):
            window = (x, y, min(1024, width - x), min(512, height - y))
            valid = _build_overlap_valid_mask(reference, sensed, window, info_i.matching_bands, info_j.matching_bands)
            mask[y:y + window[3], x:x + window[2]] = valid
            count += int(np.count_nonzero(valid))
    transform = reference.GetGeoTransform()
    pixel_area = abs(transform[1] * transform[5] - transform[2] * transform[4])
    return mask, count * pixel_area


def _valid_endpoints(points, mask, transform, info_i, info_j):
    from .joint_coregistration import _pixels_to_map
    keep = np.ones(len(points), dtype=bool)
    inverse = gdal.InvGeoTransform(transform)
    for info, pixels in ((info_i, points[:, :2]), (info_j, points[:, 2:])):
        xy = _pixels_to_map(info.transform, pixels)
        columns = inverse[0] + xy[:, 0] * inverse[1] + xy[:, 1] * inverse[2]
        rows = inverse[3] + xy[:, 0] * inverse[4] + xy[:, 1] * inverse[5]
        finite = np.isfinite(columns) & np.isfinite(rows)
        inside = finite & (columns >= 0) & (rows >= 0) & (columns < mask.shape[1]) & (rows < mask.shape[0])
        indices = np.flatnonzero(inside)
        inside[indices] &= mask[rows[indices].astype(int), columns[indices].astype(int)]
        keep &= inside
    return points[keep]


def _accepted(candidates, centers, info_i, info_j, maximum_displacement, threshold, per_window):
    from .joint_coregistration import _filter_map_ties, _select_window_ties
    if not candidates:
        return np.empty((0, 4), dtype=float)
    points, windows = np.vstack(candidates), np.vstack(centers)
    keep = _filter_map_ties(points, info_i, info_j, maximum_displacement, threshold)
    selected = _select_window_ties(points[keep], windows[keep], info_i, per_window)
    # Preserve spatial subdivision order when the last batch exceeds the goal.
    first_seen = {}
    for index, point in enumerate(points):
        first_seen.setdefault(tuple(np.round(point, 6)), index)
    order = sorted(range(len(selected)), key=lambda i: first_seen[tuple(np.round(selected[i], 6))])
    return selected[order]


def extract_spread_ties(info_i, info_j, feature_method, maximum_displacement, threshold,
                        search_radius, max_features, per_window, maximum, minimum,
                        max_attempts, required, tmpdir, *, debug_logs=False, source_ties=None):
    from .joint_coregistration import (
        _build_pair_overlap_vrts, _extract_grid_pixel_matches,
        _overlap_to_original_pixels, _drop_duplicate_ties,
    )
    ref_path, sensed_path = _build_pair_overlap_vrts(info_i, info_j, tmpdir)
    reference, sensed = gdal.Open(ref_path), gdal.Open(sensed_path)
    mask, area = _shared_mask(reference, sensed, info_i, info_j, tmpdir)
    try:
        if area == 0:
            raise InsufficientTieSamples(f'Overlap {info_i.name} <-> {info_j.name} has no shared valid pixels.')
        goal, required = resolve_sample_limits(area, maximum, minimum, required)
        transform = reference.GetGeoTransform()
        if source_ties is not None:
            selected = _drop_duplicate_ties(_valid_endpoints(np.asarray(source_ties, dtype=float).reshape(-1, 4), mask, transform, info_i, info_j))
            attempts, reason = 0, 'loaded tie cache'
        else:
            # A finite retry budget keeps featureless overlaps from searching
            # billions of pixel centers. It is independent of accepted quotas.
            budget = max_attempts if max_attempts is not None else max(100, 10 * goal)
            candidates, centers = [], []
            selected = np.empty((0, 4), dtype=float)
            attempts, reason = 0, 'valid search centers exhausted'
            batch = max(1, min(1024, goal // 4))
            sizes = (math.hypot(transform[1], transform[4]), math.hypot(transform[2], transform[5]))
            for row, col in spread_centers(mask, sizes):
                x, y = gdal.ApplyGeoTransform(transform, col + .5, row + .5)
                points, windows = _extract_grid_pixel_matches(
                    reference, sensed, info_i, info_j, 500, search_radius,
                    feature_method, max_features, centers=[(x, y)],
                )
                attempts += 1
                if len(points):
                    original = _overlap_to_original_pixels(points, transform, info_i, info_j)
                    original = _valid_endpoints(original, mask, transform, info_i, info_j)
                    if len(original):
                        candidates.append(original)
                        centers.append(np.tile((x, y), (len(original), 1)))
                if attempts % batch == 0 or attempts == budget:
                    selected = _accepted(candidates, centers, info_i, info_j, maximum_displacement, threshold, per_window)
                    if len(selected) >= goal:
                        reason = 'target reached'
                        break
                if attempts >= budget:
                    reason = f'attempt limit {budget} reached'
                    break
            else:
                selected = _accepted(candidates, centers, info_i, info_j, maximum_displacement, threshold, per_window)
        found = len(selected)
        if debug_logs or found < goal:
            _print_line(f'overlap_spread {info_i.name} <-> {info_j.name}: {min(found, goal)}/{goal} accepted pairs, '
                        f'minimum {required}, {attempts} searches, {area:g} square CRS units; {reason}.')
        if found < required:
            raise InsufficientTieSamples(f'Not enough overlap_spread matched pairs between {info_i.name} and {info_j.name}: '
                                         f'{found} found, {required} required (target {goal}; {reason}).')
        return selected[:goal]
    finally:
        # Close before the temporary directory is removed (also on Windows).
        mask._mmap.close()
        reference = sensed = None
