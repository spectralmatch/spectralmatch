"""Joint global and locally varying coregistration of overlapping rasters."""

from __future__ import annotations

import json
import math
import os
import tempfile
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Literal
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.spatial import cKDTree



import numpy as np
from osgeo import gdal

from ..handlers import (
    _check_raster_requirements,
    _existing_outputs_are_reusable,
    _resolve_paths,
    _resolve_reusable_output_paths,
)
from ..match.global_regression import _find_overlaps
from ..types_and_validation import JointCoregistration, Universal
from ..utils import (
    _get_gdal_bounds,
    _resolve_gdal_dtype,
    _resolve_window_size,
    _set_gdal_cache,
    _set_gdal_workers,
    compute_resolution,
    compute_overviews,
)
from ..utils_multiprocessing import _get_executor, _resolve_parallel_config

gdal.UseExceptions()


@dataclass(frozen=True)
class _ImageInfo:
    name: str
    path: str
    transform: tuple[float, float, float, float, float, float]
    projection: str
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    pixel_size: float
    center: tuple[float, float]
    coordinate_scale: float


@dataclass
class _Mesh:
    xs: np.ndarray
    ys: np.ndarray
    displacement: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.ys)

    @property
    def columns(self) -> int:
        return len(self.xs)


class _SparseEquations:
    def __init__(self, columns: int):
        self.columns = columns
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.values: list[float] = []
        self.targets: list[float] = []
        self.weights: list[float] = []
        self.robust_rows: list[bool] = []

    def add(self, coefficients: dict[int, float], target: float, weight: float, robust: bool = False):
        row = len(self.targets)
        for column, value in coefficients.items():
            if value:
                self.rows.append(row)
                self.cols.append(column)
                self.values.append(float(value))
        self.targets.append(float(target))
        self.weights.append(float(weight))
        self.robust_rows.append(robust)

    def arrays(self):

        matrix = sparse.coo_matrix(
            (self.values, (self.rows, self.cols)),
            shape=(len(self.targets), self.columns),
        ).tocsr()
        return (
            matrix,
            np.asarray(self.targets, dtype=float),
            np.asarray(self.weights, dtype=float),
            np.asarray(self.robust_rows, dtype=bool),
        )


def joint_coregistration(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    *,
    global_model: Literal["none", "translation", "similarity", "affine"] = "translation",
    global_image_position_preservation_weights: dict[str, float] | None = None,
    global_tie_point_alignment_strength: float = 1.0,
    local_model: Literal["none", "bilinear", "piecewise_affine"] = "piecewise_affine",
    local_image_position_preservation_weights: dict[str, float] | None = None,
    local_tie_point_alignment_strength: float = 1.0,
    local_grid_spacing: float = 500.0,
    local_smoothness_weight: float = 1.0,
    local_bending_weight: float = 1.0,
    local_anchor_falloff_distance: float = 500.0,
    feature_method: Literal["orb"] = "orb",
    maximum_tie_point_displacement: float | None = None,
    ransac_reprojection_threshold: float | None = None,
    robust_loss: Literal["none", "huber", "soft_l1", "cauchy"] = "huber",
    robust_loss_scale: float | None = None,
    save_adjustments: str | None = None,
    load_adjustments: str | None = None,
    resampling_method: Literal["nearest", "bilinear", "cubic", "lanczos"] = "bilinear",
    tap: bool = False,
    resolution: Universal.Resolution = None,
    output_dtype: Universal.CustomOutputDtype = None,
    custom_nodata_value: Universal.CustomNodataValue = None,
    window_size: Universal.WindowSize = None,
    save_as_cog: Universal.SaveAsCog = False,
    build_overviews: bool = False,
    window_scales: tuple[int, ...] | None = (2, 4, 8, 16, 32),
    cache: Universal.Cache = None,
    image_threads: Universal.Threads = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    concurrent_processing_backend: Universal.ConcurrentProcessingBackend = "process_pool",
    dask_scheduler: Universal.DaskScheduler = None,
    debug_logs: Universal.DebugLogs = False,
    resume_from_outputs: Literal["no", "yes", "validate"] = "no",
) -> list[str]:
    """Coregister overlapping rasters with a joint global transform and local deformation mesh.

    Tie points are detected pairwise, while both correction stages are solved jointly
    across the complete overlap network. Image-position weights are keyed by
    extension-free basenames; larger values preserve an image's existing position
    more strongly. Spatial distances use the common input CRS units.

    Args:
        input_images: Input folder, glob, or list of raster paths.
        output_images: Output folder, template, or list of raster paths.
        global_model: Image-wide correction model.
        global_image_position_preservation_weights: Per-basename resistance to global movement.
        global_tie_point_alignment_strength: Global tie-point strength from 0 to 1.
        local_model: Residual deformation interpolation model.
        local_image_position_preservation_weights: Per-basename resistance to local movement.
        local_tie_point_alignment_strength: Local tie-point strength from 0 to 1.
        local_grid_spacing: Local mesh spacing in input CRS units.
        local_smoothness_weight: Neighboring-node displacement penalty.
        local_bending_weight: Local displacement curvature penalty.
        local_anchor_falloff_distance: Tie-point influence falloff in input CRS units.
        feature_method: Feature matching method.
        maximum_tie_point_displacement: Optional match-displacement limit in input CRS units.
        ransac_reprojection_threshold: Optional RANSAC threshold in input CRS units.
        robust_loss: Loss used to downweight residual match errors.
        robust_loss_scale: Robust-loss scale in input CRS units; defaults to the RANSAC threshold.
        save_adjustments: JSON path at which to save raw per-pair pixel tie points.
        load_adjustments: JSON path from which to partially reuse pixel tie points.
        resampling_method: Resampling used for the final composed warp.
        tap: Snap rewritten output extents to the target-resolution grid.
        resolution: Shared pixel size strategy, positive CRS-unit pixel size, or None to preserve native resolution.
        output_dtype: Output GDAL data type, or None to retain each input type.
        custom_nodata_value: Optional output NoData override.
        window_size: Output tile size.
        save_as_cog: Save Cloud-Optimized GeoTIFF outputs.
        build_overviews: Build output overviews.
        window_scales: Overview decimation factors, default (2, 4, 8, 16, 32); None or an empty tuple disables overview creation.
        cache: GDAL cache size in gigabytes.
        image_threads: Parallel workers for overlap matching and output images.
        io_threads: GDAL I/O workers.
        tile_threads: GDAL warp/tile workers.
        concurrent_processing_backend: Use a local process pool or an existing Dask cluster.
        dask_scheduler: Existing Dask scheduler as ("file", path) or ("address", address).
        debug_logs: Print processing details.
        resume_from_outputs: Reuse no, existing, or validated existing outputs.

    Returns:
        Paths to coregistered output rasters, in input order.
    """
    print("Start joint coregistration")
    Universal._validate(
        input_images=input_images,
        window_scales=window_scales,
        output_images=output_images,
        save_as_cog=save_as_cog,
        debug_logs=debug_logs,
        window_size=window_size,
        custom_nodata_value=custom_nodata_value,
        output_dtype=output_dtype,
        cache=cache,
        image_threads=image_threads,
        io_threads=io_threads,
        tile_threads=tile_threads,
        concurrent_processing_backend=concurrent_processing_backend,
        dask_scheduler=dask_scheduler,
    )
    JointCoregistration._validate(
        global_model=global_model,
        global_image_position_preservation_weights=global_image_position_preservation_weights,
        global_tie_point_alignment_strength=global_tie_point_alignment_strength,
        local_model=local_model,
        local_image_position_preservation_weights=local_image_position_preservation_weights,
        local_tie_point_alignment_strength=local_tie_point_alignment_strength,
        local_grid_spacing=local_grid_spacing,
        local_smoothness_weight=local_smoothness_weight,
        local_bending_weight=local_bending_weight,
        local_anchor_falloff_distance=local_anchor_falloff_distance,
        feature_method=feature_method,
        maximum_tie_point_displacement=maximum_tie_point_displacement,
        ransac_reprojection_threshold=ransac_reprojection_threshold,
        robust_loss=robust_loss,
        robust_loss_scale=robust_loss_scale,
        resampling_method=resampling_method,
        tap=tap,
        resolution=resolution,
        build_overviews=build_overviews,
        save_adjustments=save_adjustments,
        load_adjustments=load_adjustments,
        resume_from_outputs=resume_from_outputs,
    )

    input_paths = _resolve_paths("search", input_images, kwargs={"default_file_pattern": "*.tif"})
    output_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={"paths_or_bases": input_paths, "default_file_pattern": "$_Coregistered.tif"},
    )
    if len(input_paths) < 2:
        raise ValueError("joint_coregistration requires at least two input images.")
    if len(output_paths) != len(input_paths):
        raise ValueError("output_images must resolve to one path per input image.")
    if len({os.path.abspath(path) for path in output_paths}) != len(output_paths):
        raise ValueError("output_images must resolve to unique paths.")
    if {os.path.abspath(path) for path in input_paths} & {
        os.path.abspath(path) for path in output_paths
    }:
        raise ValueError("joint_coregistration does not support in-place output paths.")

    names = _resolve_paths("name", input_paths)
    if len(set(names)) != len(names):
        raise ValueError("Input image basenames must be unique.")
    _validate_weight_names(global_image_position_preservation_weights, names)
    _validate_weight_names(local_image_position_preservation_weights, names)
    _check_raster_requirements(input_paths, debug_logs, check_geotransform=True, check_crs=True)
    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)
    target_resolution = compute_resolution(input_paths, resolution)

    infos = _use_common_coordinate_frame(
        {name: _read_image_info(name, path) for name, path in zip(names, input_paths)}
    )
    overlaps = _find_overlaps({name: info.bounds for name, info in infos.items()})
    reusable = _resolve_reusable_output_paths(
        output_paths,
        resume_mode=resume_from_outputs,
        debug_logs=debug_logs,
        step_name="joint_coregistration",
    )
    if len(reusable) == len(output_paths):
        return output_paths

    if debug_logs:
        print(f"Input images: {input_paths}")
        print(f"Output images: {output_paths}")
        print(f"Overlapping image pairs: {len(overlaps)}")
        print(f"Output resolution: {target_resolution or 'native'}")

    alignment_enabled = (
        global_model != "none" and global_tie_point_alignment_strength > 0
    ) or (local_model != "none" and local_tie_point_alignment_strength > 0)
    if alignment_enabled or save_adjustments:
        loaded = _load_tie_points(load_adjustments) if load_adjustments else {}
        tie_points, raw_tie_points, thresholds = _collect_tie_points(
            overlaps,
            infos,
            loaded,
            feature_method,
            maximum_tie_point_displacement,
            ransac_reprojection_threshold,
            image_threads,
            debug_logs,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
        )
    else:
        tie_points, raw_tie_points, thresholds = {}, {}, []
    if save_adjustments:
        _save_tie_points(save_adjustments, raw_tie_points)

    loss_scale = robust_loss_scale or (float(np.median(thresholds)) if thresholds else 1.0)
    global_weights = _resolve_image_weights(global_image_position_preservation_weights, names)
    local_weights = _resolve_image_weights(local_image_position_preservation_weights, names)
    global_parameters = _solve_global_alignment(
        infos,
        tie_points,
        global_model,
        global_weights,
        global_tie_point_alignment_strength,
        robust_loss,
        loss_scale,
        debug_logs,
    )
    meshes = _solve_local_alignment(
        infos,
        tie_points,
        global_model,
        global_parameters,
        local_model,
        local_weights,
        local_tie_point_alignment_strength,
        local_grid_spacing,
        local_smoothness_weight,
        local_bending_weight,
        local_anchor_falloff_distance,
        robust_loss,
        loss_scale,
        debug_logs,
    )

    image_threads_on, image_thread_workers = _resolve_parallel_config(
        image_threads, concurrent_processing_backend, dask_scheduler
    )
    tile_thread_on, tile_thread_workers = _resolve_parallel_config(tile_threads)
    output_dtype_names = {
        name: _resolve_gdal_dtype(output_dtype, infos[name].path, debug_logs)
        for name in names
    }
    args = [
        (
            infos[name],
            output_path,
            global_model,
            global_parameters[name],
            local_model,
            meshes.get(name),
            resampling_method,
            target_resolution,
            tap,
            output_dtype_names[name],
            custom_nodata_value,
            window_size,
            save_as_cog,
            tile_thread_on,
            tile_thread_workers,
            debug_logs,
            resume_from_outputs,
        )
        for name, output_path in zip(names, output_paths)
        if output_path not in reusable
    ]
    if debug_logs:
        print("Apply joint coregistration and saving results for:")
    if image_threads_on:
        with _get_executor(
            "thread",
            image_thread_workers,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
        ) as executor:
            futures = [executor.submit(_apply_alignment_process_image, *arg) for arg in args]
            for future in as_completed(futures):
                future.result()
    else:
        for arg in args:
            _apply_alignment_process_image(*arg)

    if build_overviews and window_scales:
        compute_overviews(
            input_images_paths=output_paths,
            window_scales=window_scales,
            cache=cache,
            image_threads=image_threads,
            io_threads=io_threads,
            tile_threads=tile_threads,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
            debug_logs=debug_logs,
        )
    return output_paths


def _validate_weight_names(weights: dict[str, float] | None, names: list[str]) -> None:
    unknown = sorted(set(weights or {}) - set(names))
    if unknown:
        raise ValueError(f"Image-position weight basenames not found in input_images: {unknown}")


def _resolve_image_weights(weights: dict[str, float] | None, names: list[str]) -> dict[str, float]:
    resolved = {name: float((weights or {}).get(name, 1.0)) for name in names}
    mean = float(np.mean(list(resolved.values())))
    return {name: value / mean for name, value in resolved.items()}


def _read_image_info(name: str, path: str) -> _ImageInfo:
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not open raster: {path}")
    transform = tuple(dataset.GetGeoTransform())
    width, height = dataset.RasterXSize, dataset.RasterYSize
    projection = dataset.GetProjectionRef() or ""
    dataset = None
    if not projection:
        raise ValueError(f"Input image has no coordinate reference system: {path}")
    bounds = _get_gdal_bounds(path)
    center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    pixel_size = max(math.hypot(transform[1], transform[4]), math.hypot(transform[2], transform[5]))
    coordinate_scale = max(bounds[2] - bounds[0], bounds[3] - bounds[1], pixel_size)
    return _ImageInfo(name, path, transform, projection, width, height, bounds, pixel_size, center, coordinate_scale)


def _use_common_coordinate_frame(infos: dict[str, _ImageInfo]) -> dict[str, _ImageInfo]:
    min_x = min(info.bounds[0] for info in infos.values())
    min_y = min(info.bounds[1] for info in infos.values())
    max_x = max(info.bounds[2] for info in infos.values())
    max_y = max(info.bounds[3] for info in infos.values())
    center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    scale = max(max_x - min_x, max_y - min_y, max(info.pixel_size for info in infos.values()))
    return {
        name: _ImageInfo(
            info.name,
            info.path,
            info.transform,
            info.projection,
            info.width,
            info.height,
            info.bounds,
            info.pixel_size,
            center,
            scale,
        )
        for name, info in infos.items()
    }


def _collect_tie_points(
    overlaps,
    infos,
    loaded,
    feature_method,
    maximum_displacement,
    ransac_threshold,
    image_threads,
    debug_logs,
    concurrent_processing_backend="process_pool",
    dask_scheduler=None,
):
    current_pairs = {_canonical_pair(*pair) for pair in overlaps}
    raw_tie_points = {pair: points for pair, points in loaded.items() if pair in current_pairs}
    tie_points = {}
    thresholds = []
    for pair, points in list(raw_tie_points.items()):
        threshold = _resolved_ransac_threshold(infos[pair[0]], infos[pair[1]], ransac_threshold)
        filtered = _filter_map_tie_points(
            points, infos[pair[0]], infos[pair[1]], maximum_displacement, threshold
        )
        if len(filtered):
            tie_points[pair] = filtered
            thresholds.append(threshold)
        else:
            raw_tie_points.pop(pair)

    missing = [pair for pair in overlaps if _canonical_pair(*pair) not in tie_points]
    if debug_logs:
        print(f"Loaded tie-point pairs: {len(tie_points)}")
        print(f"Tie-point pairs to calculate: {len(missing)}")
    args = [
        (
            infos[name_i],
            infos[name_j],
            feature_method,
            maximum_displacement,
            ransac_threshold,
            debug_logs,
        )
        for name_i, name_j in missing
    ]
    image_threads_on, image_thread_workers = _resolve_parallel_config(
        image_threads, concurrent_processing_backend, dask_scheduler
    )
    if image_threads_on:
        with _get_executor(
            "thread",
            image_thread_workers,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
        ) as executor:
            futures = [executor.submit(_extract_pair_tie_points, *arg) for arg in args]
            results = [future.result() for future in as_completed(futures)]
    else:
        results = [_extract_pair_tie_points(*arg) for arg in args]
    for pair, raw_points, points, threshold in results:
        raw_tie_points[pair] = raw_points
        tie_points[pair] = points
        thresholds.append(threshold)
    return tie_points, raw_tie_points, thresholds


def _canonical_pair(name_i: str, name_j: str) -> tuple[str, str]:
    return tuple(sorted((name_i, name_j)))


def _resolved_ransac_threshold(info_i, info_j, threshold):
    return float(threshold if threshold is not None else 3 * max(info_i.pixel_size, info_j.pixel_size))


def _extract_pair_tie_points(info_i, info_j, feature_method, maximum_displacement, ransac_threshold, debug_logs):
    import cv2

    pair = _canonical_pair(info_i.name, info_j.name)
    if pair != (info_i.name, info_j.name):
        info_i, info_j = info_j, info_i
    threshold = _resolved_ransac_threshold(info_i, info_j, ransac_threshold)
    if debug_logs:
        print(f"    Tie points: {info_i.name} <-> {info_j.name}")
    try:
        with tempfile.TemporaryDirectory(prefix="spectralmatch_tie_points_") as tmpdir:
            ref_vrt, sensed_vrt = _build_pair_overlap_vrts(info_i, info_j, tmpdir)
            ref_ds = gdal.Open(ref_vrt, gdal.GA_ReadOnly)
            sensed_ds = gdal.Open(sensed_vrt, gdal.GA_ReadOnly)
            valid_mask = _build_overlap_valid_mask(ref_ds, sensed_ds)
            overlap_transform = tuple(ref_ds.GetGeoTransform())
            ref_ds = sensed_ds = None
            overlap_points = _extract_overlap_pixel_matches(ref_vrt, sensed_vrt, valid_mask, feature_method)
            raw_points = _overlap_to_original_pixels(overlap_points, overlap_transform, info_i, info_j)
            points = _filter_map_tie_points(raw_points, info_i, info_j, maximum_displacement, threshold)
    except (ValueError, RuntimeError, cv2.error) as error:
        if debug_logs:
            print(f"    No usable tie points for {info_i.name} <-> {info_j.name}: {error}")
        raw_points = np.empty((0, 4), dtype=float)
        points = np.empty((0, 4), dtype=float)
    if debug_logs:
        print(f"    Tie points kept: {len(points)}")
    return pair, raw_points, points, threshold


def _build_pair_overlap_vrts(info_i, info_j, tmpdir):
    x_min = max(info_i.bounds[0], info_j.bounds[0])
    y_min = max(info_i.bounds[1], info_j.bounds[1])
    x_max = min(info_i.bounds[2], info_j.bounds[2])
    y_max = min(info_i.bounds[3], info_j.bounds[3])
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Images do not overlap.")
    resolution = max(info_i.pixel_size, info_j.pixel_size)
    paths = (os.path.join(tmpdir, "reference.vrt"), os.path.join(tmpdir, "sensed.vrt"))
    for source, destination in zip((info_i.path, info_j.path), paths):
        dataset = gdal.Warp(
            destination,
            source,
            options=gdal.WarpOptions(
                format="VRT",
                dstSRS=info_i.projection or None,
                outputBounds=(x_min, y_min, x_max, y_max),
                xRes=resolution,
                yRes=resolution,
                resampleAlg=gdal.GRIORA_Bilinear,
                dstAlpha=True,
                warpOptions=["SKIP_NOSOURCE=YES", "UNIFIED_SRC_NODATA=YES"],
            ),
        )
        if dataset is None:
            raise RuntimeError("Failed to create overlap raster.")
        dataset = None
    return paths


def _extract_overlap_pixel_matches(ref_vrt, sensed_vrt, valid_mask, feature_method):
    if feature_method != "orb":
        raise ValueError("Only feature_method='orb' is currently supported.")
    import cv2

    ref_gray = _read_uint8_gray(ref_vrt, valid_mask)
    sensed_gray = _read_uint8_gray(sensed_vrt, valid_mask)
    detector = cv2.ORB_create(nfeatures=5000)
    mask = valid_mask.astype(np.uint8) * 255
    ref_keypoints, ref_descriptors = detector.detectAndCompute(ref_gray, mask)
    sensed_keypoints, sensed_descriptors = detector.detectAndCompute(sensed_gray, mask)
    if ref_descriptors is None or sensed_descriptors is None:
        raise ValueError("Could not compute tie-point descriptors.")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(ref_descriptors, sensed_descriptors)
    if not matches:
        raise ValueError("No tie-point matches found.")
    matches = sorted(matches, key=lambda match: match.distance)
    matches = matches[: max(1, int(math.ceil(len(matches) * 0.5)))]
    points = [
        (*ref_keypoints[match.queryIdx].pt, *sensed_keypoints[match.trainIdx].pt)
        for match in matches
    ]
    return np.asarray(points, dtype=float)


def _overlap_to_original_pixels(points, overlap_transform, info_i, info_j):
    if not len(points):
        return np.empty((0, 4), dtype=float)
    inv_i, inv_j = gdal.InvGeoTransform(info_i.transform), gdal.InvGeoTransform(info_j.transform)
    converted = []
    for ref_col, ref_row, sensed_col, sensed_row in points:
        ref_x, ref_y = _pixel_to_map_coords(overlap_transform, ref_row, ref_col)
        sensed_x, sensed_y = _pixel_to_map_coords(overlap_transform, sensed_row, sensed_col)
        col_i, row_i = gdal.ApplyGeoTransform(inv_i, ref_x, ref_y)
        col_j, row_j = gdal.ApplyGeoTransform(inv_j, sensed_x, sensed_y)
        converted.append((col_i - 0.5, row_i - 0.5, col_j - 0.5, row_j - 0.5))
    return np.asarray(converted, dtype=float)


def _filter_map_tie_points(points, info_i, info_j, maximum_displacement, ransac_threshold):
    points = np.asarray(points, dtype=float).reshape((-1, 4))
    if not len(points):
        return points
    valid = np.isfinite(points).all(axis=1)
    valid &= (points[:, 0] >= 0) & (points[:, 0] < info_i.width)
    valid &= (points[:, 1] >= 0) & (points[:, 1] < info_i.height)
    valid &= (points[:, 2] >= 0) & (points[:, 2] < info_j.width)
    valid &= (points[:, 3] >= 0) & (points[:, 3] < info_j.height)
    points = points[valid]
    if not len(points):
        return points
    map_i = _pixels_to_map(info_i.transform, points[:, :2])
    map_j = _pixels_to_map(info_j.transform, points[:, 2:])
    if maximum_displacement is not None:
        keep = np.linalg.norm(map_i - map_j, axis=1) <= maximum_displacement
        points, map_i, map_j = points[keep], map_i[keep], map_j[keep]
    if len(points) < 3:
        return _drop_duplicate_tie_points(points)

    import cv2

    origin = np.mean(np.vstack((map_i, map_j)), axis=0)
    _, inliers = cv2.estimateAffine2D(
        (map_j - origin).astype(np.float32),
        (map_i - origin).astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold),
        maxIters=2000,
        confidence=0.99,
    )
    if inliers is None or np.count_nonzero(inliers) < 3:
        return np.empty((0, 4), dtype=float)
    points = points[inliers.ravel().astype(bool)]
    return _drop_duplicate_tie_points(points)


def _drop_duplicate_tie_points(points):
    if not len(points):
        return points
    _, indices = np.unique(np.round(points, 6), axis=0, return_index=True)
    return points[np.sort(indices)]


def _pixels_to_map(transform, pixels):
    columns, rows = pixels[:, 0] + 0.5, pixels[:, 1] + 0.5
    return np.column_stack(
        (
            transform[0] + columns * transform[1] + rows * transform[2],
            transform[3] + columns * transform[4] + rows * transform[5],
        )
    )


def _load_tie_points(path: str) -> dict[tuple[str, str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as file:
        model = json.load(file)
    if not isinstance(model, dict) or not isinstance(model.get("tie_points"), list):
        raise ValueError("Tie-point JSON must contain a 'tie_points' list.")
    loaded = {}
    for record in model["tie_points"]:
        if not isinstance(record, dict):
            raise ValueError("Each tie-point pair must be an object.")
        name_i, name_j = record.get("image_1"), record.get("image_2")
        points = record.get("points")
        if not isinstance(name_i, str) or not isinstance(name_j, str) or name_i == name_j:
            raise ValueError("Each tie-point pair must name two different images.")
        if not isinstance(points, list):
            raise ValueError("Each tie-point pair must contain a points list.")
        parsed = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Each tie point must contain two pixel coordinates.")
            pixel_i, pixel_j = point
            if not _valid_pixel_coordinate(pixel_i) or not _valid_pixel_coordinate(pixel_j):
                raise ValueError("Tie-point pixels must be finite [column, row] number pairs.")
            parsed.append((*map(float, pixel_i), *map(float, pixel_j)))
        pair = _canonical_pair(name_i, name_j)
        if pair in loaded:
            raise ValueError(f"Duplicate tie-point pair in JSON: {pair}")
        array = np.asarray(parsed, dtype=float).reshape((-1, 4))
        loaded[pair] = array if pair == (name_i, name_j) else array[:, [2, 3, 0, 1]]
    return loaded


def _valid_pixel_coordinate(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
        and np.isfinite(value).all()
    )


def _save_tie_points(path: str, tie_points: dict[tuple[str, str], np.ndarray]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    model = {
        "tie_points": [
            {
                "image_1": name_i,
                "image_2": name_j,
                "points": [
                    [
                        [float(point[0]), float(point[1])],
                        [float(point[2]), float(point[3])],
                    ]
                    for point in points
                ],
            }
            for (name_i, name_j), points in sorted(tie_points.items())
        ]
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)


def _global_parameter_count(model: str) -> int:
    return {"none": 0, "translation": 2, "similarity": 4, "affine": 6}[model]


def _global_basis(info: _ImageInfo, x: float, y: float, model: str):
    xn = (x - info.center[0]) / info.coordinate_scale
    yn = (y - info.center[1]) / info.coordinate_scale
    if model == "translation":
        return {0: 1.0}, {1: 1.0}
    if model == "similarity":
        return {0: 1.0, 2: xn, 3: -yn}, {1: 1.0, 2: yn, 3: xn}
    if model == "affine":
        return {0: 1.0, 2: xn, 3: yn}, {1: 1.0, 4: xn, 5: yn}
    return {}, {}


def _solve_global_alignment(
    infos,
    tie_points,
    model,
    image_weights,
    strength,
    robust_loss,
    robust_scale,
    debug_logs,
):
    parameter_count = _global_parameter_count(model)
    zeros = {name: np.zeros(parameter_count, dtype=float) for name in infos}
    if parameter_count == 0 or strength == 0 or not any(len(points) for points in tie_points.values()):
        return zeros
    names = list(infos)
    offsets = {name: index * parameter_count for index, name in enumerate(names)}
    equations = _SparseEquations(len(names) * parameter_count)
    for (name_i, name_j), points in tie_points.items():
        if not len(points):
            continue
        map_i = _pixels_to_map(infos[name_i].transform, points[:, :2])
        map_j = _pixels_to_map(infos[name_j].transform, points[:, 2:])
        pair_weight = 1.0 / len(points)
        for point_i, point_j in zip(map_i, map_j):
            basis_ix, basis_iy = _global_basis(infos[name_i], *point_i, model)
            basis_jx, basis_jy = _global_basis(infos[name_j], *point_j, model)
            equations.add(
                _difference_coefficients(
                    basis_ix,
                    offsets[name_i],
                    basis_jx,
                    offsets[name_j],
                    1 / math.sqrt(image_weights[name_i]),
                    1 / math.sqrt(image_weights[name_j]),
                ),
                point_j[0] - point_i[0],
                pair_weight,
                True,
            )
            equations.add(
                _difference_coefficients(
                    basis_iy,
                    offsets[name_i],
                    basis_jy,
                    offsets[name_j],
                    1 / math.sqrt(image_weights[name_i]),
                    1 / math.sqrt(image_weights[name_j]),
                ),
                point_j[1] - point_i[1],
                pair_weight,
                True,
            )
    solution = _solve_sparse_equations(equations, robust_loss, robust_scale)
    result = {
        name: (
            solution[offsets[name] : offsets[name] + parameter_count]
            * strength
            / math.sqrt(image_weights[name])
        )
        for name in names
    }
    if debug_logs:
        print(f"Solved joint global {model} model from {sum(map(len, tie_points.values()))} tie points")
        for name in names:
            print(f"    {name}: {np.array2string(result[name], precision=4)}")
    return result


def _difference_coefficients(basis_i, offset_i, basis_j, offset_j, scale_i=1.0, scale_j=1.0):
    coefficients = {offset_i + key: value * scale_i for key, value in basis_i.items()}
    for key, value in basis_j.items():
        column = offset_j + key
        coefficients[column] = coefficients.get(column, 0.0) - value * scale_j
    return coefficients


def _solve_sparse_equations(equations: _SparseEquations, robust_loss: str, robust_scale: float):

    matrix, target, base_weights, robust_rows = equations.arrays()
    if not len(target) or matrix.shape[1] == 0:
        return np.zeros(matrix.shape[1], dtype=float)
    solution = np.zeros(matrix.shape[1], dtype=float)
    robust_weights = np.ones(len(target), dtype=float)
    for iteration in range(10):
        weights = np.sqrt(np.maximum(base_weights * robust_weights, 0))
        weighted_matrix = sparse.diags(weights) @ matrix
        updated = lsmr(
            weighted_matrix,
            target * weights,
            atol=1e-9,
            btol=1e-9,
            maxiter=max(1000, matrix.shape[1] * 2),
        )[0]
        if iteration and np.linalg.norm(updated - solution) <= 1e-8 * (1 + np.linalg.norm(solution)):
            solution = updated
            break
        solution = updated
        if robust_loss != "none":
            residual = matrix @ solution - target
            indices = np.flatnonzero(robust_rows)
            if len(indices) % 2:
                raise RuntimeError("Tie-point residual rows must occur in x/y pairs.")
            pairs = indices.reshape((-1, 2))
            magnitudes = np.linalg.norm(residual[pairs], axis=1)
            pair_weights = _robust_weights(magnitudes, robust_loss, robust_scale)
            robust_weights[pairs] = pair_weights[:, None]
    return solution


def _robust_weights(residual, loss, scale):
    magnitude = np.abs(residual) / max(float(scale), np.finfo(float).eps)
    if loss == "huber":
        return np.where(magnitude <= 1, 1.0, 1.0 / np.maximum(magnitude, 1))
    if loss == "soft_l1":
        return 1.0 / np.sqrt(1.0 + magnitude**2)
    if loss == "cauchy":
        return 1.0 / (1.0 + magnitude**2)
    return np.ones_like(magnitude)


def _evaluate_global(info, parameters, model, coordinates):
    coordinates = np.asarray(coordinates, dtype=float).reshape((-1, 2))
    if model == "none" or not len(parameters):
        return coordinates.copy()
    corrections = np.empty_like(coordinates)
    for index, (x, y) in enumerate(coordinates):
        basis_x, basis_y = _global_basis(info, x, y, model)
        corrections[index] = (
            sum(parameters[key] * value for key, value in basis_x.items()),
            sum(parameters[key] * value for key, value in basis_y.items()),
        )
    return coordinates + corrections


def _build_mesh(info: _ImageInfo, spacing: float) -> _Mesh:
    min_x, min_y, max_x, max_y = info.bounds
    columns = max(2, int(math.ceil((max_x - min_x) / spacing)) + 1)
    rows = max(2, int(math.ceil((max_y - min_y) / spacing)) + 1)
    return _Mesh(
        np.linspace(min_x, max_x, columns),
        np.linspace(min_y, max_y, rows),
        np.zeros((rows, columns, 2), dtype=float),
    )


def _solve_local_alignment(
    infos,
    tie_points,
    global_model,
    global_parameters,
    local_model,
    image_weights,
    strength,
    grid_spacing,
    smoothness_weight,
    bending_weight,
    falloff_distance,
    robust_loss,
    robust_scale,
    debug_logs,
):
    if local_model == "none":
        return {}
    meshes = {name: _build_mesh(info, grid_spacing) for name, info in infos.items()}
    if strength == 0 or not any(len(points) for points in tie_points.values()):
        return meshes
    offsets, column_count = {}, 0
    for name, mesh in meshes.items():
        offsets[name] = column_count
        column_count += mesh.rows * mesh.columns * 2
    equations = _SparseEquations(column_count)
    support = {name: [] for name in infos}

    for (name_i, name_j), points in tie_points.items():
        if not len(points):
            continue
        map_i = _pixels_to_map(infos[name_i].transform, points[:, :2])
        map_j = _pixels_to_map(infos[name_j].transform, points[:, 2:])
        corrected_i = _evaluate_global(infos[name_i], global_parameters[name_i], global_model, map_i)
        corrected_j = _evaluate_global(infos[name_j], global_parameters[name_j], global_model, map_j)
        pair_weight = 1.0 / len(points)
        support[name_i].extend(map_i)
        support[name_j].extend(map_j)
        for point_i, point_j, global_i, global_j in zip(map_i, map_j, corrected_i, corrected_j):
            basis_i = _mesh_basis(meshes[name_i], *point_i, local_model)
            basis_j = _mesh_basis(meshes[name_j], *point_j, local_model)
            for axis in (0, 1):
                coefficients = {
                    offsets[name_i] + node * 2 + axis: value
                    for node, value in basis_i
                }
                for node, value in basis_j:
                    column = offsets[name_j] + node * 2 + axis
                    coefficients[column] = coefficients.get(column, 0.0) - value
                equations.add(
                    coefficients,
                    global_j[axis] - global_i[axis],
                    pair_weight,
                    True,
                )

    for name, mesh in meshes.items():
        offset = offsets[name]
        node_count = mesh.rows * mesh.columns
        support_points = np.asarray(support[name], dtype=float).reshape((-1, 2))
        tree = cKDTree(support_points) if len(support_points) else None
        anchor_weights = _mesh_anchor_weights(mesh, tree, falloff_distance)
        for row in range(mesh.rows):
            for column in range(mesh.columns):
                node = row * mesh.columns + column
                for axis in (0, 1):
                    equations.add(
                        {offset + node * 2 + axis: 1.0},
                        0.0,
                        image_weights[name] * anchor_weights[row, column] / node_count,
                    )
        _add_mesh_regularization(
            equations,
            mesh,
            offset,
            smoothness_weight,
            bending_weight,
        )

    solution = _solve_sparse_equations(equations, robust_loss, robust_scale) * strength
    for name, mesh in meshes.items():
        count = mesh.rows * mesh.columns * 2
        mesh.displacement = solution[offsets[name] : offsets[name] + count].reshape(
            (mesh.rows, mesh.columns, 2)
        )
    if debug_logs:
        node_total = sum(mesh.rows * mesh.columns for mesh in meshes.values())
        print(f"Solved joint local {local_model} model with {node_total} displacement nodes")
    return meshes


def _mesh_basis(mesh: _Mesh, x: float, y: float, model: str):
    column = min(max(np.searchsorted(mesh.xs, x) - 1, 0), mesh.columns - 2)
    row = min(max(np.searchsorted(mesh.ys, y) - 1, 0), mesh.rows - 2)
    fx = np.clip((x - mesh.xs[column]) / (mesh.xs[column + 1] - mesh.xs[column]), 0, 1)
    fy = np.clip((y - mesh.ys[row]) / (mesh.ys[row + 1] - mesh.ys[row]), 0, 1)
    node_00 = row * mesh.columns + column
    node_10 = node_00 + 1
    node_01 = node_00 + mesh.columns
    node_11 = node_01 + 1
    if model == "piecewise_affine":
        if fx + fy <= 1:
            return [(node_00, 1 - fx - fy), (node_10, fx), (node_01, fy)]
        return [(node_11, fx + fy - 1), (node_01, 1 - fx), (node_10, 1 - fy)]
    return [
        (node_00, (1 - fx) * (1 - fy)),
        (node_10, fx * (1 - fy)),
        (node_01, (1 - fx) * fy),
        (node_11, fx * fy),
    ]


def _mesh_anchor_weights(mesh: _Mesh, tree, falloff_distance: float):
    xx, yy = np.meshgrid(mesh.xs, mesh.ys)
    nodes = np.column_stack((xx.ravel(), yy.ravel()))
    if tree is None:
        distances = np.full(len(nodes), np.inf)
    else:
        distances = tree.query(nodes, workers=1)[0]
    falloff = 1 - np.exp(-np.square(distances / falloff_distance))
    return np.maximum(falloff, 1e-6).reshape((mesh.rows, mesh.columns))


def _add_mesh_regularization(equations, mesh, offset, smoothness_weight, bending_weight):
    horizontal_edges = mesh.rows * (mesh.columns - 1)
    vertical_edges = (mesh.rows - 1) * mesh.columns
    edge_count = max(horizontal_edges + vertical_edges, 1)
    if smoothness_weight:
        for row in range(mesh.rows):
            for column in range(mesh.columns - 1):
                _add_node_difference(equations, offset, row * mesh.columns + column, row * mesh.columns + column + 1, smoothness_weight / edge_count)
        for row in range(mesh.rows - 1):
            for column in range(mesh.columns):
                _add_node_difference(equations, offset, row * mesh.columns + column, (row + 1) * mesh.columns + column, smoothness_weight / edge_count)
    bend_count = max(mesh.rows * max(mesh.columns - 2, 0) + max(mesh.rows - 2, 0) * mesh.columns, 1)
    if bending_weight:
        for row in range(mesh.rows):
            for column in range(1, mesh.columns - 1):
                nodes = (row * mesh.columns + column - 1, row * mesh.columns + column, row * mesh.columns + column + 1)
                _add_node_bend(equations, offset, nodes, bending_weight / bend_count)
        for row in range(1, mesh.rows - 1):
            for column in range(mesh.columns):
                nodes = ((row - 1) * mesh.columns + column, row * mesh.columns + column, (row + 1) * mesh.columns + column)
                _add_node_bend(equations, offset, nodes, bending_weight / bend_count)


def _add_node_difference(equations, offset, node_i, node_j, weight):
    for axis in (0, 1):
        equations.add(
            {offset + node_i * 2 + axis: 1.0, offset + node_j * 2 + axis: -1.0},
            0.0,
            weight,
        )


def _add_node_bend(equations, offset, nodes, weight):
    for axis in (0, 1):
        equations.add(
            {
                offset + nodes[0] * 2 + axis: 1.0,
                offset + nodes[1] * 2 + axis: -2.0,
                offset + nodes[2] * 2 + axis: 1.0,
            },
            0.0,
            weight,
        )


def _evaluate_mesh(mesh: _Mesh | None, coordinates, model: str):
    coordinates = np.asarray(coordinates, dtype=float).reshape((-1, 2))
    if mesh is None:
        return np.zeros_like(coordinates)
    values = np.empty_like(coordinates)
    flat = mesh.displacement.reshape((-1, 2))
    for index, (x, y) in enumerate(coordinates):
        values[index] = sum(
            (weight * flat[node] for node, weight in _mesh_basis(mesh, x, y, model)),
            start=np.zeros(2),
        )
    return values


def _apply_alignment_process_image(
    info,
    output_path,
    global_model,
    global_parameters,
    local_model,
    mesh,
    resampling_method,
    target_resolution,
    tap,
    output_dtype,
    nodata_value,
    window_size,
    save_as_cog,
    tile_thread_on,
    tile_thread_workers,
    debug_logs,
    resume_from_outputs,
):
    if debug_logs:
        print(f"    {info.name}")
    if _existing_outputs_are_reusable(
        [output_path],
        resume_mode=resume_from_outputs,
        debug_logs=debug_logs,
        step_name="joint_coregistration",
    ):
        return
    resolved_window = _resolve_window_size(window_size, info.path, debug_logs) or 256
    resolved_window = max(16, int(math.ceil(resolved_window / 16)) * 16)
    driver, creation_options = _output_creation_options(
        save_as_cog, resolved_window, tile_thread_on, tile_thread_workers
    )
    local_is_zero = mesh is None or np.allclose(mesh.displacement, 0, atol=1e-10)
    _validate_corrected_geotransform(info, global_parameters, global_model)
    with tempfile.TemporaryDirectory(prefix="spectralmatch_joint_coregistration_") as tmpdir:
        if local_is_zero:
            output = _write_global_affine_output(
                info,
                output_path,
                global_model,
                global_parameters,
                output_dtype,
                nodata_value,
                driver,
                creation_options,
                tmpdir,
                resampling_method,
                target_resolution,
                tap,
                tile_thread_on,
                tile_thread_workers,
            )
        else:
            output = _write_local_warp_output(
                info,
                output_path,
                global_model,
                global_parameters,
                local_model,
                mesh,
                resampling_method,
                output_dtype,
                nodata_value,
                driver,
                creation_options,
                tile_thread_on,
                tile_thread_workers,
                tmpdir,
                target_resolution,
                tap,
            )
    if output is None:
        raise RuntimeError(f"Failed to write coregistered image: {output_path}")
    if debug_logs:
        print(f"Wrote: {output_path}")


def _output_creation_options(save_as_cog, window_size, tile_thread_on, tile_thread_workers):
    if save_as_cog:
        return "COG", [
            "COMPRESS=ZSTD",
            "LEVEL=9",
            f"BLOCKSIZE={window_size}",
            "OVERVIEWS=AUTO",
            "RESAMPLING=NEAREST",
        ]
    options = [
        "TILED=YES",
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "ZLEVEL=6",
        "BIGTIFF=IF_SAFER",
        f"BLOCKXSIZE={window_size}",
        f"BLOCKYSIZE={window_size}",
    ]
    if tile_thread_on:
        options.append(f"NUM_THREADS={tile_thread_workers}")
    return "GTiff", options


def _transform_resolution(transform):
    return (
        math.hypot(transform[1], transform[4]),
        math.hypot(transform[2], transform[5]),
    )


def _write_global_affine_output(
    info,
    output_path,
    global_model,
    global_parameters,
    output_dtype,
    nodata_value,
    driver,
    creation_options,
    tmpdir,
    resampling_method="bilinear",
    target_resolution=None,
    tap=False,
    tile_thread_on=False,
    tile_thread_workers=None,
):
    vrt_path = os.path.join(tmpdir, "global_alignment.vrt")
    dataset = gdal.Translate(vrt_path, info.path, options=gdal.TranslateOptions(format="VRT"))
    if dataset is None:
        raise RuntimeError("Failed to create global-alignment VRT.")
    corrected_transform = _corrected_geotransform(info, global_parameters, global_model)
    dataset.SetGeoTransform(corrected_transform)
    dataset = None
    _unlink_output(output_path)
    if target_resolution is None and not tap:
        output = gdal.Translate(
            output_path,
            vrt_path,
            options=gdal.TranslateOptions(
                format=driver,
                outputType=gdal.GetDataTypeByName(output_dtype),
                noData=nodata_value,
                creationOptions=creation_options,
            ),
        )
    else:
        x_resolution, y_resolution = target_resolution or _transform_resolution(
            corrected_transform
        )
        warp_options = ["SKIP_NOSOURCE=YES", "UNIFIED_SRC_NODATA=YES"]
        if tile_thread_on:
            warp_options.append(f"NUM_THREADS={tile_thread_workers}")
        output = gdal.Warp(
            output_path,
            vrt_path,
            options=gdal.WarpOptions(
                format=driver,
                dstSRS=info.projection or None,
                xRes=x_resolution,
                yRes=y_resolution,
                targetAlignedPixels=tap,
                resampleAlg=_gdal_resampling(resampling_method),
                outputType=gdal.GetDataTypeByName(output_dtype),
                srcNodata=nodata_value,
                dstNodata=nodata_value,
                multithread=tile_thread_on,
                warpOptions=warp_options,
                creationOptions=creation_options,
            ),
        )
    if output is None:
        return None
    output = None
    return output_path


def _corrected_geotransform(info, parameters, model):
    transform = info.transform
    source = np.asarray(
        [
            gdal.ApplyGeoTransform(transform, 0, 0),
            gdal.ApplyGeoTransform(transform, 1, 0),
            gdal.ApplyGeoTransform(transform, 0, 1),
        ],
        dtype=float,
    )
    corrected = _evaluate_global(info, parameters, model, source)
    return (
        corrected[0, 0],
        corrected[1, 0] - corrected[0, 0],
        corrected[2, 0] - corrected[0, 0],
        corrected[0, 1],
        corrected[1, 1] - corrected[0, 1],
        corrected[2, 1] - corrected[0, 1],
    )


def _validate_corrected_geotransform(info, parameters, model):
    corrected = _corrected_geotransform(info, parameters, model)
    source_determinant = info.transform[1] * info.transform[5] - info.transform[2] * info.transform[4]
    corrected_determinant = corrected[1] * corrected[5] - corrected[2] * corrected[4]
    if (
        not np.isfinite(corrected_determinant)
        or abs(corrected_determinant) <= abs(source_determinant) * 1e-8
        or source_determinant * corrected_determinant <= 0
    ):
        raise ValueError(f"Global alignment for {info.name} creates an invalid image transform.")


def _write_local_warp_output(
    info,
    output_path,
    global_model,
    global_parameters,
    local_model,
    mesh,
    resampling_method,
    output_dtype,
    nodata_value,
    driver,
    creation_options,
    tile_thread_on,
    tile_thread_workers,
    tmpdir,
    target_resolution=None,
    tap=False,
):
    x_path = os.path.join(tmpdir, "geolocation_x.tif")
    y_path = os.path.join(tmpdir, "geolocation_y.tif")
    vrt_path = os.path.join(tmpdir, "geolocated_source.vrt")
    columns, rows, corrected = _geolocation_grid(
        info, global_parameters, global_model, mesh, local_model
    )
    x_values = corrected[:, 0].reshape((len(rows), len(columns)))
    y_values = corrected[:, 1].reshape((len(rows), len(columns)))
    _write_coordinate_raster(x_path, x_values, info.projection)
    _write_coordinate_raster(y_path, y_values, info.projection)

    source = gdal.Translate(vrt_path, info.path, options=gdal.TranslateOptions(format="VRT"))
    if source is None:
        raise RuntimeError("Failed to create geolocation source VRT.")
    source.SetMetadata(
        {
            "SRS": info.projection,
            "X_DATASET": x_path,
            "X_BAND": "1",
            "Y_DATASET": y_path,
            "Y_BAND": "1",
            "PIXEL_OFFSET": "0",
            "LINE_OFFSET": "0",
            "PIXEL_STEP": str(columns[1] - columns[0] if len(columns) > 1 else 1),
            "LINE_STEP": str(rows[1] - rows[0] if len(rows) > 1 else 1),
            "GEOREFERENCING_CONVENTION": "TOP_LEFT_CORNER",
        },
        "GEOLOCATION",
    )
    source = None

    native_x_resolution, native_y_resolution = _transform_resolution(info.transform)
    x_resolution, y_resolution = target_resolution or (
        native_x_resolution,
        native_y_resolution,
    )
    bounds = (
        float(np.min(x_values) - native_x_resolution / 2),
        float(np.min(y_values) - native_y_resolution / 2),
        float(np.max(x_values) + native_x_resolution / 2),
        float(np.max(y_values) + native_y_resolution / 2),
    )
    warp_options = ["SKIP_NOSOURCE=YES", "UNIFIED_SRC_NODATA=YES"]
    if tile_thread_on:
        warp_options.append(f"NUM_THREADS={tile_thread_workers}")
    _unlink_output(output_path)
    output = gdal.Warp(
        output_path,
        vrt_path,
        options=gdal.WarpOptions(
            format=driver,
            geoloc=True,
            transformerOptions=["SRC_METHOD=GEOLOC_ARRAY"],
            dstSRS=info.projection or None,
            outputBounds=bounds,
            xRes=x_resolution,
            yRes=y_resolution,
            targetAlignedPixels=tap,
            resampleAlg=_gdal_resampling(resampling_method),
            errorThreshold=0.0,
            outputType=gdal.GetDataTypeByName(output_dtype),
            srcNodata=nodata_value,
            dstNodata=nodata_value,
            multithread=tile_thread_on,
            warpOptions=warp_options,
            creationOptions=creation_options,
        ),
    )
    if output is None:
        return None
    output = None
    return output_path


def _geolocation_grid(info, global_parameters, global_model, mesh, local_model):
    local_spacing = min(
        np.min(np.diff(mesh.xs)) if len(mesh.xs) > 1 else info.pixel_size,
        np.min(np.diff(mesh.ys)) if len(mesh.ys) > 1 else info.pixel_size,
    )
    pixel_step = max(1.0, local_spacing / max(2 * info.pixel_size, np.finfo(float).eps))
    column_count = max(2, int(math.ceil(max(info.width - 1, 1) / pixel_step)) + 1)
    row_count = max(2, int(math.ceil(max(info.height - 1, 1) / pixel_step)) + 1)
    columns = np.linspace(0, max(info.width - 1, 0), column_count)
    rows = np.linspace(0, max(info.height - 1, 0), row_count)
    cc, rr = np.meshgrid(columns, rows)
    pixels = np.column_stack((cc.ravel(), rr.ravel()))
    source = _pixels_to_map(info.transform, pixels)
    globally_corrected = _evaluate_global(info, global_parameters, global_model, source)
    local_displacement = _evaluate_mesh(mesh, source, local_model)
    corrected = globally_corrected + local_displacement
    if _grid_has_foldover(globally_corrected, corrected, len(rows), len(columns)):
        raise ValueError(f"Local alignment for {info.name} creates a folded displacement grid.")
    return columns, rows, corrected


def _grid_has_foldover(reference, corrected, rows, columns):
    if rows < 2 or columns < 2:
        return False
    ref = reference.reshape((rows, columns, 2))
    out = corrected.reshape((rows, columns, 2))
    def triangle_crosses(grid):
        first_h = grid[:-1, 1:] - grid[:-1, :-1]
        first_v = grid[1:, :-1] - grid[:-1, :-1]
        second_h = grid[1:, :-1] - grid[1:, 1:]
        second_v = grid[:-1, 1:] - grid[1:, 1:]
        first = first_h[..., 0] * first_v[..., 1] - first_h[..., 1] * first_v[..., 0]
        second = second_h[..., 0] * second_v[..., 1] - second_h[..., 1] * second_v[..., 0]
        return first, second

    reference_crosses = triangle_crosses(ref)
    output_crosses = triangle_crosses(out)
    return any(
        np.any(reference_cross * output_cross <= 0)
        for reference_cross, output_cross in zip(reference_crosses, output_crosses)
    )


def _write_coordinate_raster(path, values, projection):
    rows, columns = values.shape
    dataset = gdal.GetDriverByName("GTiff").Create(
        path,
        columns,
        rows,
        1,
        gdal.GDT_Float64,
        options=["TILED=YES", "COMPRESS=DEFLATE"],
    )
    if dataset is None:
        raise RuntimeError("Failed to create temporary geolocation raster.")
    if projection:
        dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(values)
    dataset = None


def _gdal_resampling(method):
    return {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "lanczos": gdal.GRA_Lanczos,
    }[method]


def _unlink_output(path):
    if os.path.exists(path):
        gdal.Unlink(path)


def _coregister_overlap(
    reference_overlap_vrt: str,
    sensed_overlap_vrt: str,
    output_raster_path: str,
    *,
    valid_mask_path: str | None = None,
    tie_point_pairs: list[tuple[float, float, float, float]] | None = None,
    feature_method: Literal["orb"] = "orb",
    cache: Universal.Cache = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    debug_logs: Universal.DebugLogs = False,
) -> tuple[str, list[tuple[float, float, float, float]]]:
    """Coregister an overlap raster using conjugate points.

    Returns the corrected raster path and the matched ``(reference_row,
    reference_column, sensed_row, sensed_column)`` point pairs used by the
    legacy PIF workflow.
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
    reference = gdal.Open(reference_overlap_vrt, gdal.GA_ReadOnly)
    sensed = gdal.Open(sensed_overlap_vrt, gdal.GA_ReadOnly)
    if reference is None or sensed is None:
        raise RuntimeError("Could not open overlap rasters for coregistration.")
    valid_mask = _read_mask(valid_mask_path) if valid_mask_path else _build_overlap_valid_mask(reference, sensed)
    supplied_tie_points = tie_point_pairs is not None
    if not supplied_tie_points:
        point_pairs = _extract_conjugate_point_pairs(
            reference_overlap_vrt, sensed_overlap_vrt, valid_mask, feature_method
        )
    else:
        supplied = np.asarray(tie_point_pairs, dtype=float)
        if supplied.ndim != 2 or supplied.shape[1] != 4 or not np.isfinite(supplied).all():
            raise ValueError("tie_point_pairs must contain finite row/column quadruples.")
        point_pairs = [tuple(point) for point in supplied]
    point_pairs = _filter_point_pairs(point_pairs, debug_logs)
    if len(point_pairs) < 3:
        raise ValueError("At least 3 conjugate point pairs are required for overlap coregistration.")
    output_dir = os.path.dirname(output_raster_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spectralmatch_gcps_") as tmpdir:
        gcp_vrt = os.path.join(tmpdir, "sensed_gcps.vrt")
        gcp_dataset = gdal.Translate(gcp_vrt, sensed_overlap_vrt, options=gdal.TranslateOptions(format="VRT"))
        if gcp_dataset is None:
            raise RuntimeError("Failed to create temporary VRT for overlap coregistration.")
        reference_transform = reference.GetGeoTransform()
        projection = reference.GetProjectionRef()
        gcps = [
            gdal.GCP(
                *_pixel_to_map_coords(reference_transform, ref_row, ref_col),
                0.0,
                float(sensed_col),
                float(sensed_row),
            )
            for ref_row, ref_col, sensed_row, sensed_col in point_pairs
        ]
        gcp_dataset.SetGCPs(gcps, projection)
        gcp_dataset = None
        bounds = _get_bounds_from_gt(reference_transform, reference.RasterXSize, reference.RasterYSize)
        format_name = _infer_gdal_format(output_raster_path)
        warp_options = ["SKIP_NOSOURCE=YES", "UNIFIED_SRC_NODATA=YES"]
        if tile_thread_on:
            warp_options.append(f"NUM_THREADS={tile_thread_workers}")
        corrected = gdal.Warp(
            output_raster_path,
            gcp_vrt,
            options=gdal.WarpOptions(
                format=format_name,
                tps=True,
                dstSRS=projection or None,
                outputBounds=bounds,
                width=reference.RasterXSize,
                height=reference.RasterYSize,
                resampleAlg=gdal.GRIORA_Bilinear,
                dstAlpha=True,
                multithread=tile_thread_on,
                warpOptions=warp_options,
                creationOptions=_get_creation_options(format_name),
            ),
        )
        if corrected is None:
            raise RuntimeError("Failed to warp overlap raster for coregistration.")
        corrected = None
    reference = sensed = None
    return output_raster_path, point_pairs


def _extract_conjugate_point_pairs(ref_vrt, sensed_vrt, valid_mask, feature_method):
    points = _extract_overlap_pixel_matches(ref_vrt, sensed_vrt, valid_mask, feature_method)
    pairs = []
    for ref_col, ref_row, sensed_col, sensed_row in points:
        if abs(ref_col - sensed_col) > 20 or abs(ref_row - sensed_row) > 20:
            continue
        pair = tuple(map(lambda value: int(round(value)), (ref_row, ref_col, sensed_row, sensed_col)))
        if (
            0 <= pair[0] < valid_mask.shape[0]
            and 0 <= pair[1] < valid_mask.shape[1]
            and 0 <= pair[2] < valid_mask.shape[0]
            and 0 <= pair[3] < valid_mask.shape[1]
            and valid_mask[pair[0], pair[1]]
            and valid_mask[pair[2], pair[3]]
        ):
            pairs.append(pair)
    if not pairs:
        raise ValueError("No valid conjugate point pairs remained after filtering.")
    return pairs


def _filter_point_pairs(point_pairs, debug_logs):
    point_pairs = _drop_duplicate_point_pairs(point_pairs)
    point_pairs = _keep_affine_inlier_pairs(point_pairs)
    if debug_logs:
        print(f"Overlap coregistration conjugate points kept: {len(point_pairs)}")
    return point_pairs


def _drop_duplicate_point_pairs(point_pairs):
    unique, reference_seen, sensed_seen = [], set(), set()
    for pair in point_pairs:
        reference_key, sensed_key = pair[:2], pair[2:]
        if reference_key in reference_seen or sensed_key in sensed_seen:
            continue
        reference_seen.add(reference_key)
        sensed_seen.add(sensed_key)
        unique.append(pair)
    return unique


def _keep_affine_inlier_pairs(point_pairs):
    if len(point_pairs) < 3:
        return point_pairs
    import cv2

    sensed = np.asarray([[pair[3], pair[2]] for pair in point_pairs], dtype=np.float32)
    reference = np.asarray([[pair[1], pair[0]] for pair in point_pairs], dtype=np.float32)
    _, inliers = cv2.estimateAffine2D(
        sensed,
        reference,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if inliers is None:
        return point_pairs
    filtered = [pair for pair, keep in zip(point_pairs, inliers.ravel().astype(bool)) if keep]
    return filtered if len(filtered) >= 3 else point_pairs


def _read_mask(mask_path):
    dataset = gdal.Open(mask_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not open mask raster: {mask_path}")
    mask = dataset.GetRasterBand(1).ReadAsArray().astype(bool)
    dataset = None
    return mask


def _build_overlap_valid_mask(reference, sensed):
    width = min(reference.RasterXSize, sensed.RasterXSize)
    height = min(reference.RasterYSize, sensed.RasterYSize)
    ref_mask = reference.GetRasterBand(1).GetMaskBand().ReadAsArray(0, 0, width, height) > 0
    sensed_mask = sensed.GetRasterBand(1).GetMaskBand().ReadAsArray(0, 0, width, height) > 0
    return ref_mask & sensed_mask


def _read_uint8_gray(raster_path, valid_mask):
    dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not open raster for feature detection: {raster_path}")
    rows, columns = valid_mask.shape
    band_indices = [
        index
        for index in range(1, dataset.RasterCount + 1)
        if dataset.GetRasterBand(index).GetColorInterpretation() != gdal.GCI_AlphaBand
    ][:3]
    bands = [
        dataset.GetRasterBand(index).ReadAsArray(0, 0, columns, rows).astype(np.float32)
        for index in band_indices
    ]
    dataset = None
    if not bands:
        return np.zeros(valid_mask.shape, dtype=np.uint8)
    gray = np.mean(bands, axis=0)
    values = gray[valid_mask & np.isfinite(gray)]
    if not values.size:
        return np.zeros(gray.shape, dtype=np.uint8)
    low, high = np.nanpercentile(values, [2, 98])
    if high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    scaled = np.clip((gray - low) / (high - low), 0, 1) * 255
    scaled[~valid_mask] = 0
    return scaled.astype(np.uint8)


def _pixel_to_map_coords(transform, row, column):
    return (
        transform[0] + (column + 0.5) * transform[1] + (row + 0.5) * transform[2],
        transform[3] + (column + 0.5) * transform[4] + (row + 0.5) * transform[5],
    )


def _get_bounds_from_gt(transform, width, height):
    corners = [
        gdal.ApplyGeoTransform(transform, column, row)
        for column, row in ((0, 0), (width, 0), (width, height), (0, height))
    ]
    return (
        min(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[0] for point in corners),
        max(point[1] for point in corners),
    )


def _infer_gdal_format(output_path):
    return "VRT" if output_path.lower().endswith(".vrt") else "GTiff"


def _get_creation_options(format_name):
    return ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"] if format_name == "GTiff" else []
