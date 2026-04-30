import os
import tempfile
from typing import Literal

import numpy as np
from osgeo import gdal

from ..types_and_validation import Universal
from ..utils import _set_gdal_cache, _set_gdal_workers
from ..utils_multiprocessing import _resolve_parallel_config

gdal.UseExceptions()


def geometric_correction(
    reference_overlap_vrt: str,
    sensed_overlap_vrt: str,
    output_raster_path: str,
    *,
    valid_mask_path: str | None = None,
    feature_method: Literal["orb"] = "orb",
    cache: Universal.Cache = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    debug_logs: Universal.DebugLogs = False,
) -> tuple[str, list[tuple[int, int, int, int]]]:
    """
    Geometrically correct an overlap raster using conjugate points and return the
    corrected raster path plus the matched point pairs used to estimate the warp.
    """
    if not isinstance(reference_overlap_vrt, str) or not isinstance(sensed_overlap_vrt, str):
        raise ValueError("reference_overlap_vrt and sensed_overlap_vrt must be strings.")
    if not isinstance(output_raster_path, str):
        raise ValueError("output_raster_path must be a string.")
    if feature_method != "orb":
        raise ValueError("Only feature_method='orb' is currently supported.")

    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)
    tile_thread_on, tile_thread_workers = _resolve_parallel_config(tile_threads)

    ref_ds = gdal.Open(reference_overlap_vrt, gdal.GA_ReadOnly)
    sensed_ds = gdal.Open(sensed_overlap_vrt, gdal.GA_ReadOnly)
    if ref_ds is None or sensed_ds is None:
        raise RuntimeError("Could not open overlap rasters for geometric correction.")

    valid_mask = (
        _read_mask(valid_mask_path)
        if valid_mask_path is not None
        else _build_overlap_valid_mask(ref_ds, sensed_ds)
    )
    point_pairs = _extract_conjugate_point_pairs(
        reference_overlap_vrt,
        sensed_overlap_vrt,
        valid_mask,
        feature_method,
    )
    point_pairs = _filter_point_pairs(point_pairs, debug_logs)
    if len(point_pairs) < 3:
        raise ValueError("At least 3 conjugate point pairs are required for geometric correction.")

    output_dir = os.path.dirname(output_raster_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spectralmatch_gcps_") as tmpdir:
        gcp_vrt = os.path.join(tmpdir, "sensed_gcps.vrt")
        gcp_ds = gdal.Translate(
            gcp_vrt,
            sensed_overlap_vrt,
            options=gdal.TranslateOptions(format="VRT"),
        )
        if gcp_ds is None:
            raise RuntimeError("Failed to create temporary VRT for geometric correction.")

        ref_gt = ref_ds.GetGeoTransform()
        projection = ref_ds.GetProjectionRef()
        gcps = []
        for ref_row, ref_col, sensed_row, sensed_col in point_pairs:
            map_x, map_y = _pixel_to_map_coords(ref_gt, ref_row, ref_col)
            gcps.append(gdal.GCP(map_x, map_y, 0.0, float(sensed_col), float(sensed_row)))

        gcp_ds.SetGCPs(gcps, projection)
        gcp_ds = None

        min_x, min_y, max_x, max_y = _get_bounds_from_gt(
            ref_ds.GetGeoTransform(),
            ref_ds.RasterXSize,
            ref_ds.RasterYSize,
        )
        format_name = _infer_gdal_format(output_raster_path)
        warp_options = ["SKIP_NOSOURCE=YES", "UNIFIED_SRC_NODATA=YES"]
        if tile_thread_on:
            warp_options.append(f"NUM_THREADS={tile_thread_workers}")

        corrected_ds = gdal.Warp(
            output_raster_path,
            gcp_vrt,
            options=gdal.WarpOptions(
                format=format_name,
                tps=True,
                dstSRS=projection or None,
                outputBounds=(min_x, min_y, max_x, max_y),
                width=ref_ds.RasterXSize,
                height=ref_ds.RasterYSize,
                resampleAlg=gdal.GRIORA_Bilinear,
                dstAlpha=True,
                multithread=tile_thread_on,
                warpOptions=warp_options,
                creationOptions=_get_creation_options(format_name),
            ),
        )
        if corrected_ds is None:
            raise RuntimeError("Failed to warp overlap raster for geometric correction.")
        corrected_ds = None

    ref_ds = None
    sensed_ds = None
    return output_raster_path, point_pairs


def _extract_conjugate_point_pairs(
    ref_vrt: str,
    sensed_vrt: str,
    valid_mask: np.ndarray,
    feature_method: str,
) -> list[tuple[int, int, int, int]]:
    if feature_method != "orb":
        raise ValueError("Only feature_method='orb' is currently supported.")

    import cv2

    ref_gray = _read_uint8_gray(ref_vrt, valid_mask)
    sensed_gray = _read_uint8_gray(sensed_vrt, valid_mask)
    detector = cv2.ORB_create(nfeatures=5000)
    ref_keypoints, ref_descriptors = detector.detectAndCompute(ref_gray, None)
    sensed_keypoints, sensed_descriptors = detector.detectAndCompute(sensed_gray, None)
    if ref_descriptors is None or sensed_descriptors is None:
        raise ValueError("Could not compute conjugate point descriptors for geometric correction.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(ref_descriptors, sensed_descriptors)
    if not matches:
        raise ValueError("No conjugate point matches found for geometric correction.")

    point_pairs = []
    matches = sorted(matches, key=lambda match: match.distance)
    keep_count = max(1, int(np.ceil(len(matches) * 0.5)))
    for match in matches[:keep_count]:
        x_ref, y_ref = ref_keypoints[match.queryIdx].pt
        x_sensed, y_sensed = sensed_keypoints[match.trainIdx].pt
        if abs(x_ref - x_sensed) > 20 or abs(y_ref - y_sensed) > 20:
            continue
        ref_row = int(round(y_ref))
        ref_col = int(round(x_ref))
        sensed_row = int(round(y_sensed))
        sensed_col = int(round(x_sensed))
        if (
            0 <= ref_row < valid_mask.shape[0]
            and 0 <= ref_col < valid_mask.shape[1]
            and 0 <= sensed_row < valid_mask.shape[0]
            and 0 <= sensed_col < valid_mask.shape[1]
            and valid_mask[ref_row, ref_col]
            and valid_mask[sensed_row, sensed_col]
        ):
            point_pairs.append((ref_row, ref_col, sensed_row, sensed_col))
    if not point_pairs:
        raise ValueError("No valid conjugate point pairs remained after filtering.")
    return point_pairs


def _filter_point_pairs(
    point_pairs: list[tuple[int, int, int, int]],
    debug_logs: bool,
) -> list[tuple[int, int, int, int]]:
    point_pairs = _drop_duplicate_point_pairs(point_pairs)
    point_pairs = _keep_affine_inlier_pairs(point_pairs)
    if debug_logs:
        print(f"Geometric correction conjugate points kept: {len(point_pairs)}")
    return point_pairs


def _drop_duplicate_point_pairs(
    point_pairs: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    unique_pairs = []
    seen_reference = set()
    seen_sensed = set()
    seen_full = set()
    for pair in point_pairs:
        ref_key = (pair[0], pair[1])
        sensed_key = (pair[2], pair[3])
        if pair in seen_full or ref_key in seen_reference or sensed_key in seen_sensed:
            continue
        seen_full.add(pair)
        seen_reference.add(ref_key)
        seen_sensed.add(sensed_key)
        unique_pairs.append(pair)
    return unique_pairs


def _keep_affine_inlier_pairs(
    point_pairs: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if len(point_pairs) < 3:
        return point_pairs

    import cv2

    sensed_points = np.asarray(
        [[pair[3], pair[2]] for pair in point_pairs],
        dtype=np.float32,
    )
    reference_points = np.asarray(
        [[pair[1], pair[0]] for pair in point_pairs],
        dtype=np.float32,
    )
    _, inlier_mask = cv2.estimateAffine2D(
        sensed_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if inlier_mask is None:
        return point_pairs

    inlier_mask = inlier_mask.ravel().astype(bool)
    inlier_pairs = [pair for pair, keep in zip(point_pairs, inlier_mask) if keep]
    if len(inlier_pairs) < 3:
        return point_pairs
    return inlier_pairs


def _read_mask(mask_path: str) -> np.ndarray:
    ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open mask raster: {mask_path}")
    mask = ds.GetRasterBand(1).ReadAsArray().astype(bool)
    ds = None
    return mask


def _build_overlap_valid_mask(ref_ds, sensed_ds) -> np.ndarray:
    width = min(ref_ds.RasterXSize, sensed_ds.RasterXSize)
    height = min(ref_ds.RasterYSize, sensed_ds.RasterYSize)
    ref_mask = ref_ds.GetRasterBand(1).GetMaskBand().ReadAsArray(0, 0, width, height) > 0
    sensed_mask = sensed_ds.GetRasterBand(1).GetMaskBand().ReadAsArray(0, 0, width, height) > 0
    return ref_mask & sensed_mask


def _read_uint8_gray(raster_path: str, valid_mask: np.ndarray) -> np.ndarray:
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open raster for feature detection: {raster_path}")
    rows, cols = valid_mask.shape
    bands = [
        ds.GetRasterBand(band_index).ReadAsArray(0, 0, cols, rows).astype(np.float32)
        for band_index in range(1, min(3, ds.RasterCount) + 1)
    ]
    ds = None
    gray = np.mean(bands, axis=0)
    values = gray[valid_mask & np.isfinite(gray)]
    if values.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    lo, hi = np.nanpercentile(values, [2, 98])
    if hi <= lo:
        return np.zeros(gray.shape, dtype=np.uint8)
    scaled = np.clip((gray - lo) / (hi - lo), 0, 1) * 255
    scaled[~valid_mask] = 0
    return scaled.astype(np.uint8)


def _pixel_to_map_coords(
    gt: tuple[float, float, float, float, float, float],
    row: int,
    col: int,
) -> tuple[float, float]:
    x = gt[0] + (col + 0.5) * gt[1] + (row + 0.5) * gt[2]
    y = gt[3] + (col + 0.5) * gt[4] + (row + 0.5) * gt[5]
    return x, y


def _get_bounds_from_gt(
    gt: tuple[float, float, float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    xs = [
        gt[0],
        gt[0] + width * gt[1],
        gt[0] + height * gt[2],
        gt[0] + width * gt[1] + height * gt[2],
    ]
    ys = [
        gt[3],
        gt[3] + width * gt[4],
        gt[3] + height * gt[5],
        gt[3] + width * gt[4] + height * gt[5],
    ]
    return min(xs), min(ys), max(xs), max(ys)


def _infer_gdal_format(output_path: str) -> str:
    if output_path.lower().endswith(".vrt"):
        return "VRT"
    return "GTiff"


def _get_creation_options(format_name: str) -> list[str]:
    if format_name == "GTiff":
        return ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
    return []
