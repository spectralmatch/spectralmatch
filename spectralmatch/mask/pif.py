import math
import os
import tempfile
from typing import Literal
from html import escape

import numpy as np
from osgeo import gdal

from ..handlers import _check_raster_requirements, _resolve_nodata_value, _resolve_paths
from ..types_and_validation import Universal
from ..utils import _get_gdal_bounds


def generate_pifs(
    input_images: Universal.SearchFolderOrListFiles,
    *,
    input_image_names: list[str] | None = None,
    included_names: list[str] | None = None,
    overlapping_pairs: tuple[tuple[str, str], ...] | None = None,
    calculation_dtype: Universal.CalculationDtype = "float32",
    custom_nodata_value: Universal.CustomNodataValue = None,
    red_band_index: int | None = None,
    nir_band_index: int | None = None,
    vegetation_threshold: float = 0.2,
    inz_threshold: float = 0.25,
    region_radius: int = 5,
    max_samples: int = 100000,
    min_samples: int = 32,
    custom_mean_factor: float = 1.0,
    custom_std_factor: float = 1.0,
    feature_method: Literal["orb"] = "orb",
    debug_logs: Universal.DebugLogs = False,
) -> np.ndarray:
    """
    Generate conjugate-point-based PIF correction parameters.

    This follows the main PIF ideas from Kim and Han (2021): use conjugate
    points as seeds, remove vegetation-sensitive areas, identify stable pixels
    with an integrated normalized Z-score image, grow PIFs around seed points,
    and fit per-band linear radiometric corrections.

    Returns:
        np.ndarray: Shape ``(num_bands, 2 * num_images, 1)``. For each image
        and band, entries are ``scale`` then ``offset`` such that
        ``corrected = scale * image + offset``.
    """
    input_image_paths = _resolve_paths(
        "search",
        input_images,
        kwargs={"default_file_pattern": "*.tif"},
    )
    if not input_image_paths:
        raise ValueError("No input images found for PIF generation.")

    _check_raster_requirements(
        input_image_paths,
        debug_logs,
        check_geotransform=True,
        check_crs=True,
        check_bands=True,
        check_nodata=True,
    )

    if input_image_names is None:
        input_image_names = _resolve_paths("name", input_image_paths)
    if included_names is None:
        included_names = list(input_image_names)

    nodata_value = _resolve_nodata_value(input_image_paths[0], custom_nodata_value)
    first_ds = gdal.Open(input_image_paths[0], gdal.GA_ReadOnly)
    num_bands = first_ds.RasterCount
    first_ds = None

    image_path_pairs = dict(zip(input_image_names, input_image_paths))
    if overlapping_pairs is None:
        bounds = {
            name: _get_gdal_bounds(path)
            for name, path in image_path_pairs.items()
        }
        overlapping_pairs = _find_overlaps(bounds)

    all_overlap_stats = {}
    all_whole_stats = {}
    for name_i, name_j in overlapping_pairs:
        if name_i not in image_path_pairs or name_j not in image_path_pairs:
            continue
        if name_i not in included_names and name_j not in included_names:
            continue
        if debug_logs:
            print(f"Generating conjugate PIF stats: {name_i} <-> {name_j}")

        pair_stats, whole_updates = _calculate_pair_pif_stats(
            reference_path=image_path_pairs[name_i],
            sensed_path=image_path_pairs[name_j],
            reference_name=name_i,
            sensed_name=name_j,
            num_bands=num_bands,
            nodata_value=nodata_value,
            calculation_dtype=calculation_dtype,
            red_band_index=red_band_index,
            nir_band_index=nir_band_index,
            vegetation_threshold=vegetation_threshold,
            inz_threshold=inz_threshold,
            region_radius=region_radius,
            max_samples=max_samples,
            min_samples=min_samples,
            feature_method=feature_method,
            debug_logs=debug_logs,
        )
        for outer, inner in pair_stats.items():
            all_overlap_stats.setdefault(outer, {}).update(inner)
        _merge_whole_stat_updates(all_whole_stats, whole_updates)

    return _solve_pif_global_model(
        num_bands=num_bands,
        all_image_names=input_image_names,
        included_names=included_names,
        all_overlap_stats=all_overlap_stats,
        all_whole_stats=all_whole_stats,
        custom_mean_factor=custom_mean_factor,
        custom_std_factor=custom_std_factor,
        overlapping_pairs=overlapping_pairs,
        debug_logs=debug_logs,
    )


def _find_overlaps(
    image_bounds_dict: dict[str, tuple[float, float, float, float]],
) -> tuple[tuple[str, str], ...]:
    overlaps = []
    keys = sorted(image_bounds_dict)
    for idx, name_i in enumerate(keys):
        minx_i, miny_i, maxx_i, maxy_i = image_bounds_dict[name_i]
        for name_j in keys[idx + 1:]:
            minx_j, miny_j, maxx_j, maxy_j = image_bounds_dict[name_j]
            if minx_i < maxx_j and maxx_i > minx_j and miny_i < maxy_j and maxy_i > miny_j:
                overlaps.append((name_i, name_j))
    return tuple(overlaps)


def _merge_whole_stat_updates(
    all_whole_stats: dict,
    updates: dict[str, dict[int, dict[str, float | int]]],
) -> None:
    for name, band_values in updates.items():
        all_whole_stats.setdefault(name, {})
        for band_index, stats in band_values.items():
            all_whole_stats[name].setdefault(band_index, [])
            all_whole_stats[name][band_index].append(stats)


def _finalize_whole_stats(all_whole_stats: dict) -> dict:
    finalized = {}
    for name, band_values in all_whole_stats.items():
        finalized[name] = {}
        for band_index, stats_groups in band_values.items():
            total_size = sum(stats["size"] for stats in stats_groups)
            if total_size <= 0:
                finalized[name][band_index] = {"mean": 0.0, "std": 0.0, "size": 0}
                continue
            mean = sum(stats["mean"] * stats["size"] for stats in stats_groups) / total_size
            variance = sum(
                stats["size"] * (stats["std"] ** 2 + (stats["mean"] - mean) ** 2)
                for stats in stats_groups
            ) / total_size
            finalized[name][band_index] = {
                "mean": float(mean),
                "std": float(math.sqrt(max(variance, 0.0))),
                "size": int(total_size),
            }
    return finalized


def _solve_pif_global_model(
    *,
    num_bands: int,
    all_image_names: list[str],
    included_names: list[str],
    all_overlap_stats: dict,
    all_whole_stats: dict,
    custom_mean_factor: float,
    custom_std_factor: float,
    overlapping_pairs: tuple[tuple[str, str], ...],
    debug_logs: bool,
) -> np.ndarray:
    all_whole_stats = _finalize_whole_stats(all_whole_stats)
    num_total = len(all_image_names)
    all_params = np.zeros((num_bands, 2 * num_total, 1), dtype=float)
    image_names_with_id = list(enumerate(all_image_names))

    valid_pairs = []
    for name_i, name_j in overlapping_pairs:
        stats = all_overlap_stats.get(name_i, {}).get(name_j)
        if stats and any(band_stats["size"] > 0 for band_stats in stats.values()):
            valid_pairs.append((name_i, name_j))

    if not valid_pairs:
        raise ValueError("No valid conjugate PIF overlap pairs were found.")

    for band_index in range(num_bands):
        if debug_logs:
            print(f"\nProcessing conjugate PIF band {band_index}:")

        A, y, total_overlap = [], [], 0.0
        for i, name_i in image_names_with_id:
            for j, name_j in image_names_with_id[i + 1:]:
                if (name_i, name_j) not in valid_pairs and (name_j, name_i) not in valid_pairs:
                    continue
                if name_i not in included_names and name_j not in included_names:
                    continue

                stat_i = all_overlap_stats.get(name_i, {}).get(name_j, {}).get(band_index)
                stat_j = all_overlap_stats.get(name_j, {}).get(name_i, {}).get(band_index)
                if not stat_i or not stat_j or stat_i["size"] <= 0:
                    continue

                row_m = [0] * (2 * num_total)
                row_s = [0] * (2 * num_total)
                row_m[2 * i: 2 * i + 2] = [stat_i["mean"], 1]
                row_m[2 * j: 2 * j + 2] = [-stat_j["mean"], -1]
                row_s[2 * i], row_s[2 * j] = stat_i["std"], -stat_j["std"]
                A.extend(
                    [
                        [v * custom_mean_factor for v in row_m],
                        [v * custom_std_factor for v in row_s],
                    ]
                )
                y.extend([0, 0])
                total_overlap += 1.0

        anchor_weight = 1.0 if total_overlap == 0 else total_overlap / (2.0 * num_total)
        for name in included_names:
            if name not in all_whole_stats or band_index not in all_whole_stats[name]:
                continue
            image_index = all_image_names.index(name)
            mean = all_whole_stats[name][band_index]["mean"]
            std = all_whole_stats[name][band_index]["std"]
            row_m = [0] * (2 * num_total)
            row_s = [0] * (2 * num_total)
            row_m[2 * image_index: 2 * image_index + 2] = [
                mean * anchor_weight,
                anchor_weight,
            ]
            row_s[2 * image_index] = std * anchor_weight
            A.extend([row_m, row_s])
            y.extend([mean * anchor_weight, std * anchor_weight])

        if not A:
            raise ValueError(f"No conjugate PIF constraints found for band {band_index + 1}.")

        A_arr = np.asarray(A)
        y_arr = np.asarray(y)
        all_params[band_index, :, 0] = np.linalg.lstsq(A_arr, y_arr, rcond=None)[0]

    return all_params


def _calculate_pair_pif_stats(
    *,
    reference_path: str,
    sensed_path: str,
    reference_name: str,
    sensed_name: str,
    num_bands: int,
    nodata_value,
    calculation_dtype: str,
    red_band_index: int | None,
    nir_band_index: int | None,
    vegetation_threshold: float,
    inz_threshold: float,
    region_radius: int,
    max_samples: int,
    min_samples: int,
    feature_method: str,
    debug_logs: bool,
) -> tuple[dict, dict[str, dict[int, dict[str, float | int]]]]:
    with tempfile.TemporaryDirectory(prefix="spectralmatch_pif_") as tmpdir:
        overlap = _build_overlap_vrts(
            reference_path,
            sensed_path,
            tmpdir,
        )
        if overlap is None:
            raise ValueError(f"No overlap between {reference_path} and {sensed_path}.")
        ref_vrt, sensed_vrt, width, height, gt, projection = overlap

        valid_mask_path = _build_valid_mask_raster(
            ref_vrt,
            sensed_vrt,
            width,
            height,
            gt,
            projection,
            tmpdir,
        )
        stable_mask_path = _build_inz_stable_mask_raster(
            ref_vrt=ref_vrt,
            sensed_vrt=sensed_vrt,
            valid_mask_path=valid_mask_path,
            width=width,
            height=height,
            gt=gt,
            projection=projection,
            num_bands=num_bands,
            red_band_index=red_band_index,
            nir_band_index=nir_band_index,
            vegetation_threshold=vegetation_threshold,
            inz_threshold=inz_threshold,
            tmpdir=tmpdir,
        )
        seed_points = _extract_conjugate_seed_points(
            ref_vrt,
            sensed_vrt,
            valid_mask_path,
            feature_method,
            debug_logs,
        )
        seed_mask_path = _build_seed_mask_raster(
            seed_points,
            width,
            height,
            gt,
            projection,
            region_radius,
            tmpdir,
        )
        pif_mask_path = _combine_masks_raster(
            stable_mask_path,
            seed_mask_path,
            width,
            height,
            gt,
            projection,
            tmpdir,
        )
        pif_count = _count_mask_pixels(pif_mask_path)
        if max_samples and pif_count > max_samples:
            pif_mask_path = _sample_mask_raster(
                pif_mask_path,
                max_samples,
                width,
                height,
                gt,
                projection,
                tmpdir,
            )
            pif_count = _count_mask_pixels(pif_mask_path)
        if debug_logs:
            print(f"Conjugate PIF pixels found: {pif_count} for {reference_name} <-> {sensed_name}")
        if pif_count < min_samples:
            raise ValueError(
                f"Not enough conjugate PIF samples between {reference_path} and "
                f"{sensed_path}: {pif_count} found, {min_samples} required."
            )

        pair_stats = {reference_name: {sensed_name: {}}, sensed_name: {reference_name: {}}}
        whole_updates = {reference_name: {}, sensed_name: {}}
        for band_index in range(num_bands):
            ref_stats = _masked_band_stats(ref_vrt, band_index + 1, pif_mask_path)
            sensed_stats = _masked_band_stats(sensed_vrt, band_index + 1, pif_mask_path)
            if ref_stats["size"] < min_samples:
                raise ValueError(
                    f"Band {band_index + 1} has {ref_stats['size']} conjugate PIF "
                    f"samples between {reference_name} and {sensed_name}; "
                    f"{min_samples} required."
                )
            pair_stats[reference_name][sensed_name][band_index] = ref_stats
            pair_stats[sensed_name][reference_name][band_index] = sensed_stats
            whole_updates[reference_name][band_index] = ref_stats
            whole_updates[sensed_name][band_index] = sensed_stats

    return pair_stats, whole_updates


def _build_overlap_vrts(
    reference_path: str,
    sensed_path: str,
    tmpdir: str,
) -> tuple[str, str, int, int, tuple, str] | None:
    ref_bounds = _get_gdal_bounds(reference_path)
    sensed_bounds = _get_gdal_bounds(sensed_path)
    x_min = max(ref_bounds[0], sensed_bounds[0])
    y_min = max(ref_bounds[1], sensed_bounds[1])
    x_max = min(ref_bounds[2], sensed_bounds[2])
    y_max = min(ref_bounds[3], sensed_bounds[3])
    if x_min >= x_max or y_min >= y_max:
        return None

    ref_ds = gdal.Open(reference_path, gdal.GA_ReadOnly)
    projection = ref_ds.GetProjectionRef()
    ref_gt = ref_ds.GetGeoTransform()
    x_res = abs(ref_gt[1])
    y_res = abs(ref_gt[5])
    ref_ds = None

    ref_vrt = os.path.join(tmpdir, "reference_overlap.vrt")
    sensed_vrt = os.path.join(tmpdir, "sensed_overlap.vrt")
    ref_crop = gdal.Translate(
        ref_vrt,
        reference_path,
        options=gdal.TranslateOptions(
            format="VRT",
            projWin=[x_min, y_max, x_max, y_min],
        ),
    )
    if ref_crop is None:
        raise RuntimeError("Failed to crop reference image for PIF extraction.")
    ref_crop = None

    sensed_warp = gdal.Warp(
        sensed_vrt,
        sensed_path,
        options=gdal.WarpOptions(
            format="VRT",
            dstSRS=projection or None,
            outputBounds=(x_min, y_min, x_max, y_max),
            xRes=x_res,
            yRes=y_res,
            resampleAlg=gdal.GRIORA_Bilinear,
            dstAlpha=True,
            warpOptions=["SKIP_NOSOURCE=YES", "UNIFIED_SRC_NODATA=YES"],
        ),
    )
    if sensed_warp is None:
        raise RuntimeError("Failed to warp sensed image for PIF extraction.")
    sensed_warp = None

    ref_overlap = gdal.Open(ref_vrt, gdal.GA_ReadOnly)
    sensed_overlap = gdal.Open(sensed_vrt, gdal.GA_ReadOnly)
    width = min(ref_overlap.RasterXSize, sensed_overlap.RasterXSize)
    height = min(ref_overlap.RasterYSize, sensed_overlap.RasterYSize)
    gt = ref_overlap.GetGeoTransform()
    ref_overlap = None
    sensed_overlap = None
    return ref_vrt, sensed_vrt, width, height, gt, projection


def _write_expression_vrt(
    output_path: str,
    sources: dict[str, tuple[str, str | int]],
    expression: str,
    width: int,
    height: int,
    gt,
    projection: str,
    data_type: str = "Float32",
    nodata_value: float | int | None = None,
) -> None:
    source_xml = []
    for name, (source_path, source_band) in sources.items():
        source_xml.append(
            f"""    <SimpleSource name="{escape(name)}">
      <SourceFilename relativeToVRT="0">{escape(source_path)}</SourceFilename>
      <SourceBand>{source_band}</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>
      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>
    </SimpleSource>"""
        )
    vrt_xml = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <SRS>{escape(projection or "")}</SRS>
  <GeoTransform>{", ".join(str(v) for v in gt)}</GeoTransform>
  <VRTRasterBand dataType="{data_type}" band="1" subClass="VRTDerivedRasterBand">
    {f"<NoDataValue>{nodata_value}</NoDataValue>" if nodata_value is not None else ""}
    <PixelFunctionType>expression</PixelFunctionType>
    <PixelFunctionArguments dialect="muparser" expression="{escape(expression)}"/>
{os.linesep.join(source_xml)}
  </VRTRasterBand>
</VRTDataset>
"""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(vrt_xml)


def _translate_vrt_to_raster(
    vrt_path: str,
    output_path: str,
    output_type,
    nodata_value: float | int | None = None,
) -> None:
    ds = gdal.Translate(
        output_path,
        vrt_path,
        options=gdal.TranslateOptions(
            format="GTiff",
            outputType=output_type,
            noData=nodata_value,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
        ),
    )
    if ds is None:
        raise RuntimeError(f"Failed to materialize raster: {output_path}")
    ds = None


def _build_valid_mask_raster(
    ref_vrt: str,
    sensed_vrt: str,
    width: int,
    height: int,
    gt,
    projection: str,
    tmpdir: str,
) -> str:
    valid_vrt = os.path.join(tmpdir, "valid_mask.vrt")
    valid_tif = os.path.join(tmpdir, "valid_mask.tif")
    _write_expression_vrt(
        valid_vrt,
        {"rm": (ref_vrt, "mask,1"), "sm": (sensed_vrt, "mask,1")},
        "(rm > 0 && sm > 0) ? 1 : 0",
        width,
        height,
        gt,
        projection,
        data_type="Byte",
        nodata_value=0,
    )
    _translate_vrt_to_raster(valid_vrt, valid_tif, gdal.GDT_Byte, 0)
    return valid_tif


def _build_inz_stable_mask_raster(
    *,
    ref_vrt: str,
    sensed_vrt: str,
    valid_mask_path: str,
    width: int,
    height: int,
    gt,
    projection: str,
    num_bands: int,
    red_band_index: int | None,
    nir_band_index: int | None,
    vegetation_threshold: float,
    inz_threshold: float,
    tmpdir: str,
) -> str:
    sources = {"m": (valid_mask_path, 1)}
    z_terms = []
    for band_index in range(1, num_bands + 1):
        sources[f"r{band_index}"] = (ref_vrt, band_index)
        sources[f"s{band_index}"] = (sensed_vrt, band_index)
        diff_vrt = os.path.join(tmpdir, f"diff_b{band_index}.vrt")
        _write_expression_vrt(
            diff_vrt,
            {
                "r": (ref_vrt, band_index),
                "s": (sensed_vrt, band_index),
                "m": (valid_mask_path, 1),
            },
            "m == 0 ? -999999999 : (s - r)",
            width,
            height,
            gt,
            projection,
            data_type="Float32",
            nodata_value=-999999999,
        )
        diff_ds = gdal.Open(diff_vrt, gdal.GA_ReadOnly)
        _, _, mean, std = diff_ds.GetRasterBand(1).GetStatistics(0, 1)
        diff_ds = None
        if std == 0:
            std = 1.0
        z_terms.append(f"abs(((s{band_index} - r{band_index}) - {mean}) / {std})")

    inz_expr = f"(({ ' + '.join(z_terms) }) / {num_bands})"
    vegetation_expr = "1"
    if red_band_index is not None and nir_band_index is not None:
        if (
            red_band_index < 1
            or nir_band_index < 1
            or red_band_index > num_bands
            or nir_band_index > num_bands
        ):
            raise ValueError("red_band_index and nir_band_index must be valid 1-based band indices.")
        ref_ndvi = (
            f"((r{nir_band_index} + r{red_band_index}) != 0 ? "
            f"((r{nir_band_index} - r{red_band_index}) / "
            f"(r{nir_band_index} + r{red_band_index})) : 0)"
        )
        sensed_ndvi = (
            f"((s{nir_band_index} + s{red_band_index}) != 0 ? "
            f"((s{nir_band_index} - s{red_band_index}) / "
            f"(s{nir_band_index} + s{red_band_index})) : 0)"
        )
        vegetation_expr = (
            f"(({ref_ndvi}) <= {vegetation_threshold} && "
            f"({sensed_ndvi}) <= {vegetation_threshold})"
        )

    stable_vrt = os.path.join(tmpdir, "stable_mask.vrt")
    stable_tif = os.path.join(tmpdir, "stable_mask.tif")
    _write_expression_vrt(
        stable_vrt,
        sources,
        f"(m > 0 && ({vegetation_expr}) && ({inz_expr}) <= {inz_threshold}) ? 1 : 0",
        width,
        height,
        gt,
        projection,
        data_type="Byte",
        nodata_value=0,
    )
    _translate_vrt_to_raster(stable_vrt, stable_tif, gdal.GDT_Byte, 0)
    return stable_tif


def _extract_conjugate_seed_points(
    ref_vrt: str,
    sensed_vrt: str,
    valid_mask_path: str,
    feature_method: str,
    debug_logs: bool,
) -> np.ndarray:
    if feature_method != "orb":
        raise ValueError("Only feature_method='orb' is currently supported.")
    import cv2

    valid_mask = _read_mask(valid_mask_path)
    ref_gray = _read_uint8_gray(ref_vrt, valid_mask)
    sensed_gray = _read_uint8_gray(sensed_vrt, valid_mask)
    detector = cv2.ORB_create(nfeatures=5000)
    ref_keypoints, ref_descriptors = detector.detectAndCompute(ref_gray, None)
    sensed_keypoints, sensed_descriptors = detector.detectAndCompute(sensed_gray, None)
    if ref_descriptors is None or sensed_descriptors is None:
        raise ValueError("Could not compute conjugate point descriptors for PIF extraction.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(ref_descriptors, sensed_descriptors)
    if not matches:
        raise ValueError("No conjugate point matches found for PIF extraction.")
    matches = sorted(matches, key=lambda match: match.distance)
    keep_count = max(1, int(math.ceil(len(matches) * 0.5)))
    points = []
    for match in matches[:keep_count]:
        x_ref, y_ref = ref_keypoints[match.queryIdx].pt
        x_sensed, y_sensed = sensed_keypoints[match.trainIdx].pt
        if abs(x_ref - x_sensed) > 20 or abs(y_ref - y_sensed) > 20:
            continue
        row = int(round((y_ref + y_sensed) / 2))
        col = int(round((x_ref + x_sensed) / 2))
        if 0 <= row < valid_mask.shape[0] and 0 <= col < valid_mask.shape[1] and valid_mask[row, col]:
            points.append((row, col))
    return np.asarray(points, dtype=int)


def _read_mask(mask_path: str) -> np.ndarray:
    ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open mask raster: {mask_path}")
    mask = ds.GetRasterBand(1).ReadAsArray().astype(bool)
    ds = None
    return mask


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


def _build_seed_mask_raster(
    seed_points: np.ndarray,
    width: int,
    height: int,
    gt,
    projection: str,
    radius: int,
    tmpdir: str,
) -> str:
    if seed_points.size == 0:
        raise ValueError("No conjugate seed points available for PIF extraction.")
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    valid_rows = (seed_points[:, 0] >= 0) & (seed_points[:, 0] < height)
    valid_cols = (seed_points[:, 1] >= 0) & (seed_points[:, 1] < width)
    valid_points = seed_points[valid_rows & valid_cols]
    if valid_points.size == 0:
        raise ValueError("Conjugate seed points fall outside the overlap raster.")
    mask[valid_points[:, 0], valid_points[:, 1]] = 1

    radius = max(0, int(radius))
    if radius > 0:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    output_path = os.path.join(tmpdir, "seed_mask.tif")
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        output_path,
        width,
        height,
        1,
        gdal.GDT_Byte,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    if ds is None:
        raise RuntimeError(f"Failed to create seed mask raster: {output_path}")
    ds.SetGeoTransform(gt)
    if projection:
        ds.SetProjection(projection)
    ds.GetRasterBand(1).WriteArray(mask)
    ds = None
    return output_path


def _combine_masks_raster(
    stable_mask_path: str,
    seed_mask_path: str,
    width: int,
    height: int,
    gt,
    projection: str,
    tmpdir: str,
) -> str:
    pif_vrt = os.path.join(tmpdir, "pif_mask.vrt")
    pif_tif = os.path.join(tmpdir, "pif_mask.tif")
    _write_expression_vrt(
        pif_vrt,
        {
            "stable": (stable_mask_path, 1),
            "seed": (seed_mask_path, 1),
        },
        "(stable > 0 && seed > 0) ? 1 : 0",
        width,
        height,
        gt,
        projection,
        data_type="Byte",
    )
    _translate_vrt_to_raster(pif_vrt, pif_tif, gdal.GDT_Byte)
    return pif_tif


def _count_mask_pixels(mask_path: str) -> int:
    mask = _read_mask(mask_path)
    return int(np.count_nonzero(mask))


def _sample_mask_raster(
    mask_path: str,
    max_samples: int,
    width: int,
    height: int,
    gt,
    projection: str,
    tmpdir: str,
) -> str:
    mask = _read_mask(mask_path)
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size <= max_samples:
        return mask_path

    sampled = np.zeros(mask.size, dtype=np.uint8)
    keep_positions = np.linspace(0, valid_indices.size - 1, max_samples, dtype=int)
    sampled[valid_indices[keep_positions]] = 1
    sampled = sampled.reshape(mask.shape)

    output_path = os.path.join(tmpdir, "sampled_pif_mask.tif")
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        output_path,
        width,
        height,
        1,
        gdal.GDT_Byte,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    if ds is None:
        raise RuntimeError(f"Failed to create sampled PIF mask: {output_path}")
    ds.SetGeoTransform(gt)
    if projection:
        ds.SetProjection(projection)
    ds.GetRasterBand(1).WriteArray(sampled)
    ds = None
    return output_path


def _masked_band_stats(
    raster_path: str,
    band_index: int,
    mask_path: str,
) -> dict[str, float | int]:
    raster_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    mask_ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
    if raster_ds is None:
        raise RuntimeError(f"Could not open raster for masked statistics: {raster_path}")
    if mask_ds is None:
        raise RuntimeError(f"Could not open mask for masked statistics: {mask_path}")

    width = min(raster_ds.RasterXSize, mask_ds.RasterXSize)
    height = min(raster_ds.RasterYSize, mask_ds.RasterYSize)
    stats_vrt = os.path.join(
        os.path.dirname(mask_path),
        f"stats_{os.path.basename(raster_path)}_b{band_index}.vrt",
    )
    projection = raster_ds.GetProjectionRef()
    gt = raster_ds.GetGeoTransform()
    raster_ds = None
    mask_ds = None

    _write_expression_vrt(
        stats_vrt,
        {"b": (raster_path, band_index), "m": (mask_path, 1)},
        "m > 0 ? b : -999999999",
        width,
        height,
        gt,
        projection,
        data_type="Float32",
        nodata_value=-999999999,
    )
    stats_ds = gdal.Open(stats_vrt, gdal.GA_ReadOnly)
    if stats_ds is None:
        raise RuntimeError(f"Could not open masked statistics VRT: {stats_vrt}")
    band = stats_ds.GetRasterBand(1)
    _, _, mean, std = band.GetStatistics(0, 1)
    stats_ds = None
    return {
        "mean": float(mean),
        "std": float(std),
        "size": _count_mask_pixels(mask_path),
    }
