import os
import numpy as np
import sys
import json
import tempfile
import re
import math

from osgeo import gdal
gdal.UseExceptions()
from typing import List, Dict
from numpy import ndarray
from scipy.optimize import least_squares

from ..utils import  _resolve_window_size, _get_valid_count


def _solve_global_model(
    num_bands: int,
    num_total: int,
    all_image_names: list[str],
    included_names: list[str],
    input_image_names: list[str],
    all_overlap_stats: dict,
    all_whole_stats: dict,
    custom_mean_factor: float,
    custom_std_factor: float,
    overlapping_pairs: tuple[tuple[str, str], ...],
    debug_logs: bool = False,
    apply_size_weighting: bool = False,
) -> np.ndarray:
    """
    Computes global radiometric normalization parameters (scale and offset) for each image and band using least squares regression.

    Args:
        num_bands: Number of image bands.
        num_total: Total number of images (including loaded).
        all_image_names: Ordered list of all image names.
        included_names: Subset of images used to constrain the model.
        input_image_names: Names of input images to apply normalization to.
        all_overlap_stats: Pairwise overlap statistics per band.
        all_whole_stats: Whole-image stats (mean, std) per band.
        custom_mean_factor: Weight for mean constraints.
        custom_std_factor: Weight for std constraints.
        overlapping_pairs: Pairs of overlapping images.
        debug_logs: If True, prints debug information.
        apply_size_weighting (bool): Whether to use the overlap size to weight its influence.

    Returns:
        np.ndarray: Adjustment parameters of shape (bands, 2 * num_images, 1).
    """

    # prune zero-size overlaps
    valid_pairs = []

    for i, j in overlapping_pairs:
        stats = all_overlap_stats.get(i, {}).get(j)
        if not stats or not any(b["size"] > 0 for b in stats.values()):
            all_overlap_stats.get(i, {}).pop(j, None)
            all_overlap_stats.get(j, {}).pop(i, None)
            continue
        valid_pairs.append((i, j))

    overlapping_pairs = tuple(valid_pairs)

    # Calculate
    all_params = np.zeros((num_bands, 2 * num_total, 1), dtype=float)
    image_names_with_id = [(i, name) for i, name in enumerate(all_image_names)]
    for b in range(num_bands):
        if debug_logs:
            print(f"\nProcessing band {b}:")

        A, y, tot_overlap = [], [], 0
        for i, name_i in image_names_with_id:
            for j, name_j in image_names_with_id[i + 1 :]:
                stat = all_overlap_stats.get(name_i, {}).get(name_j)
                if stat is None:
                    continue

                # This condition ensures that only overlaps involving at least one included image contribute constraints, allowing external images to be calibrated against the model without influencing it.
                if name_i not in included_names and name_j not in included_names:
                    continue

                s = stat[b]["size"]
                m1, v1 = stat[b]["mean"], stat[b]["std"]
                m2, v2 = (
                    all_overlap_stats[name_j][name_i][b]["mean"],
                    all_overlap_stats[name_j][name_i][b]["std"],
                )

                weight = s if apply_size_weighting else 1.0

                row_m = [0] * (2 * num_total)
                row_s = [0] * (2 * num_total)
                row_m[2 * i : 2 * i + 2] = [m1, 1]
                row_m[2 * j : 2 * j + 2] = [-m2, -1]
                row_s[2 * i], row_s[2 * j] = v1, -v2

                A.extend(
                    [
                        [v * weight * custom_mean_factor for v in row_m],
                        [v * weight * custom_std_factor for v in row_s],
                    ]
                )
                y.extend([0, 0])
                tot_overlap += weight

        pjj = 1.0 if tot_overlap == 0 else tot_overlap / (2.0 * num_total)

        for name in included_names:
            mj = all_whole_stats[name][b]["mean"]
            vj = all_whole_stats[name][b]["std"]
            j_idx = all_image_names.index(name)
            row_m = [0] * (2 * num_total)
            row_s = [0] * (2 * num_total)
            row_m[2 * j_idx : 2 * j_idx + 2] = [mj * pjj, 1 * pjj]
            row_s[2 * j_idx] = vj * pjj
            A.extend([row_m, row_s])
            y.extend([mj * pjj, vj * pjj])

        for name in input_image_names:
            if name in included_names:
                continue
            row = [0] * (2 * num_total)
            A.append(row.copy())
            y.append(0)
            A.append(row.copy())
            y.append(0)

        A_arr = np.asarray(A)
        y_arr = np.asarray(y)
        res = least_squares(lambda p: A_arr @ p - y_arr, [1, 0] * num_total)
        all_params[b, :, 0] = res.x

        if debug_logs:
            _print_constraint_system(
                constraint_matrix=A_arr,
                adjustment_params=res.x,
                observed_values_vector=y_arr,
                overlap_pairs=overlapping_pairs,
                image_names_with_id=image_names_with_id,
            )
    return all_params


def _finalize_pif_whole_stats(all_whole_stats: dict) -> dict:
    finalized = {}
    for name, band_values in all_whole_stats.items():
        finalized[name] = {}
        for band_index, stats_groups in band_values.items():
            total_size = sum(stats["size"] for stats in stats_groups)
            if total_size <= 0:
                finalized[name][band_index] = {"mean": 0.0, "std": 0.0, "size": 0}
                continue
            mean = (
                sum(stats["mean"] * stats["size"] for stats in stats_groups)
                / total_size
            )
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
    all_whole_stats = _finalize_pif_whole_stats(all_whole_stats)
    num_total = len(all_image_names)
    all_params = np.zeros((num_bands, 2 * num_total, 1), dtype=float)
    image_names_with_id = list(enumerate(all_image_names))

    valid_pairs = []
    for name_i, name_j in overlapping_pairs:
        stats = all_overlap_stats.get(name_i, {}).get(name_j)
        if stats and any(band_stats["size"] > 0 for band_stats in stats.values()):
            valid_pairs.append((name_i, name_j))

    if not valid_pairs:
        raise ValueError("No valid flood_from_match_points PIF overlap pairs were found.")

    for band_index in range(num_bands):
        if debug_logs:
            print(f"\nProcessing flood_from_match_points PIF band {band_index}:")

        A, y, total_overlap = [], [], 0.0
        for i, name_i in image_names_with_id:
            for j, name_j in image_names_with_id[i + 1 :]:
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
                row_m[2 * i : 2 * i + 2] = [stat_i["mean"], 1]
                row_m[2 * j : 2 * j + 2] = [-stat_j["mean"], -1]
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
            row_m[2 * image_index : 2 * image_index + 2] = [
                mean * anchor_weight,
                anchor_weight,
            ]
            row_s[2 * image_index] = std * anchor_weight
            A.extend([row_m, row_s])
            y.extend([mean * anchor_weight, std * anchor_weight])

        if not A:
            raise ValueError(
                f"No flood_from_match_points PIF constraints found for band {band_index + 1}."
            )

        A_arr = np.asarray(A)
        y_arr = np.asarray(y)
        all_params[band_index, :, 0] = np.linalg.lstsq(A_arr, y_arr, rcond=None)[0]

        if debug_logs:
            _print_constraint_system(
                constraint_matrix=A_arr,
                adjustment_params=all_params[band_index, :, 0],
                observed_values_vector=y_arr,
                overlap_pairs=tuple(valid_pairs),
                image_names_with_id=image_names_with_id,
            )

    return all_params


def _apply_adjustments_process_image(
    tile_thread_on: bool,
    tile_thread_workers: int,
    image_name: str,
    input_image_path: str,
    output_image_path: str,
    scale: np.ndarray,
    offset: np.ndarray,
    num_bands: int,
    nodata_val: int | float | None,
    window_size,
    output_dtype: str | None,
    calculation_dtype: str | None,
    save_as_cog,
    debug_logs: bool,
):
    """
    Applies per-band linear radiometric adjustments to an image using GDAL VRT metadata and materializes the result as a GeoTIFF or COG. Each band is transformed according to: y = a * x + b, where `a` = scale and `b` = offset.

    Args:
        tile_thread_on (bool): Enable multithreaded GDAL translation for tile-level work.
        tile_thread_workers (int): Number of worker threads if `tile_thread_on=True`.
        image_name (str): Basename (no extension) of the input image; used for temporary files.
        input_image_path (str): Path to the input image to adjust.
        output_image_path (str): Path where the adjusted raster will be written.
        scale (np.ndarray): 1D array of per-band scale coefficients (length = num_bands).
        offset (np.ndarray): 1D array of per-band offset coefficients (length = num_bands).
        num_bands (int): Number of data bands to adjust (alpha not included).
        nodata_val (int | float | None): NoData value to assign to output bands, if provided.
        window_size: Window size used for tiling (currently not applied at pixel level).
        output_dtype (str | None): Desired GDAL output data type (e.g., "UInt16"). If None, preserves source dtype.
        calculation_dtype (str | None): Desired GDAL calculation dtype.
        save_as_cog (bool): If True, writes output as Cloud-Optimized GeoTIFF (COG); otherwise, writes a standard tiled GeoTIFF.
        debug_logs (bool, optional): If True, print detailed logging about the process. Defaults to False.

    Returns:
        None
    """
    if debug_logs: print(f"    {image_name}")
    # VRT wrapper of the input
    with tempfile.TemporaryDirectory(prefix="spectralmatch_adjust_") as tmpdir:
        vrt_path = os.path.join(tmpdir, f"{image_name}_linear.vrt")

        window_size = _resolve_window_size(window_size, input_image_path, debug_logs)

        ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
        w, h = ds.RasterXSize, ds.RasterYSize
        srs = ds.GetProjectionRef() or ""
        gt = ds.GetGeoTransform()
        ds = None

        bands_xml = []
        for i in range(1, num_bands + 1):
            if nodata_val is None: expr = f"{float(scale[i - 1])}*B1 + {float(offset[i - 1])}"
            else: expr = f"(B1=={nodata_val}) ? {nodata_val} : ({float(scale[i-1])}*B1 + {float(offset[i-1])})"

            bands_xml.append(
                f'  <VRTRasterBand dataType="{calculation_dtype}" band="{i}" subClass="VRTDerivedRasterBand">\n' +
                (f"    <NoDataValue>{nodata_val}</NoDataValue>\n" if nodata_val is not None else "") +
                "    <PixelFunctionType>expression</PixelFunctionType>\n"
                f'    <PixelFunctionArguments dialect="muparser" expression="{expr}"/>\n' +
                "    <SimpleSource>\n"
                f'      <SourceFilename relativeToVRT="0">{input_image_path}</SourceFilename>\n' +
                f'      <SourceBand>{i}</SourceBand>\n' +
                f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>\n' +
                f'      <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>\n' +
                "    </SimpleSource>\n"
                "  </VRTRasterBand>"
            )

        vrt_xml = (
                f'<VRTDataset rasterXSize="{w}" rasterYSize="{h}">\n' +
                f'  <SRS>{srs}</SRS>\n' +
                f'  <GeoTransform>{", ".join(str(v) for v in gt)}</GeoTransform>\n' +
                "\n".join(bands_xml) + "\n</VRTDataset>\n"
        )

        with open(vrt_path, "w", encoding="utf-8") as f:
            f.write(vrt_xml)

        # Materialize to output with UNscale=True (applies band Scale/Offset in C++)
        driver_name = "COG" if (bool(save_as_cog)) else "GTiff"

        co = []
        if driver_name == "COG":
            co = [
                "COMPRESS=ZSTD",
                "LEVEL=9",
                f"BLOCKSIZE={window_size}",
                "OVERVIEWS=AUTO",
                "RESAMPLING=NEAREST"
            ]
        else:
            # Tiled GeoTIFF
            co = [
                "TILED=YES",
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "ZLEVEL=6",
                "BIGTIFF=IF_SAFER",
                f"BLOCKXSIZE={window_size}",
                f"BLOCKYSIZE={window_size}",
            ]
            if tile_thread_on: co.append(f"NUM_THREADS={tile_thread_workers}")

        try:
            if os.path.exists(output_image_path):
                gdal.Unlink(output_image_path)
        except Exception:
            pass

        out_ds = gdal.Translate(
            output_image_path,
            vrt_path,
            options=gdal.TranslateOptions(
                format=driver_name,
                creationOptions=co,
                bandList=list(range(1, num_bands + 1)),
                noData=nodata_val if nodata_val is not None else None,
                outputType=gdal.GetDataTypeByName(output_dtype)
            )
        )
        if out_ds is None:
            raise RuntimeError("gdal.Translate failed to write adjusted image")
        out_ds = None

        if debug_logs: print(f"Wrote: {output_image_path}")


def _save_adjustments(
    save_path: str,
    input_image_names: List[str],
    all_params: np.ndarray,
    all_whole_stats: dict,
    all_overlap_stats: dict,
    num_bands: int,
    calculation_dtype: str,
) -> None:
    """
    Saves adjustment parameters, whole-image stats, and overlap stats in a nested JSON format.

    Args:
        save_path (str): Output JSON path.
        input_image_names (List[str]): List of input image names.
        all_params (np.ndarray): Adjustment parameters, shape (bands, 2 * num_images, 1).
        all_whole_stats (dict): Per-image stats (keyed by image name).
        all_overlap_stats (dict): Per-pair overlap stats (keyed by image name).
        num_bands (int): Number of bands.
        calculation_dtype (str): Precision for saving values (e.g., "float32").
    """

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cast = lambda x: float(np.dtype(calculation_dtype).type(x))

    full_model = {}
    for i, name in enumerate(input_image_names):
        full_model[name] = {
            "adjustments": {
                f"band_{b}": {
                    "scale": cast(all_params[b, 2 * i, 0]),
                    "offset": cast(all_params[b, 2 * i + 1, 0]),
                }
                for b in range(num_bands)
            },
            "whole_stats": {
                f"band_{b}": {
                    "mean": cast(all_whole_stats[name][b]["mean"]),
                    "std": cast(all_whole_stats[name][b]["std"]),
                    "size": int(all_whole_stats[name][b]["size"]),
                }
                for b in range(num_bands)
            },
            "overlap_stats": {},
        }

    for name_i, j_stats in all_overlap_stats.items():
        for name_j, band_stats in j_stats.items():
            if name_j not in full_model[name_i]["overlap_stats"]:
                full_model[name_i]["overlap_stats"][name_j] = {}
            for b, stats in band_stats.items():
                full_model[name_i]["overlap_stats"][name_j][f"band_{b}"] = {
                    "mean": cast(stats["mean"]),
                    "std": cast(stats["std"]),
                    "size": int(stats["size"]),
                }

    with open(save_path, "w") as f:
        json.dump(full_model, f, indent=2)


def _validate_adjustment_model_structure(model: dict) -> None:
    """
    Validates the structure of a loaded adjustment model dictionary.

    Ensures that:
    - Each top-level key is an image name mapping to a dictionary.
    - Each image has 'adjustments' and 'whole_stats' with per-band keys like 'band_0'.
    - Each band entry in 'adjustments' contains 'scale' and 'offset'.
    - Each band entry in 'whole_stats' contains 'mean', 'std', and 'size'.
    - If present, 'overlap_stats' maps to other image names with valid per-band statistics.

    The expected model structure is a dictionary with this format:

    {
        "image_name_1": {
            "adjustments": {
                "band_0": {"scale": float, "offset": float},
                "band_1": {"scale": float, "offset": float},
                ...
            },
            "whole_stats": {
                "band_0": {"mean": float, "std": float, "size": int},
                "band_1": {"mean": float, "std": float, "size": int},
                ...
            },
            "overlap_stats": {
                "image_name_2": {
                    "band_0": {"mean": float, "std": float, "size": int},
                    "band_1": {"mean": float, "std": float, "size": int},
                    ...
                },
                ...
            }
        },
        ...
    }

    - Keys are image basenames (without extension).
    - Band keys are of the form "band_0", "band_1", etc.
    - All numerical values are stored as floats (except 'size', which is an int).

    Args:
        model (dict): Parsed JSON adjustment model.

    Raises:
        ValueError: If any structural issues or missing keys are detected.
    """
    for image_name, image_data in model.items():
        if not isinstance(image_data, dict):
            raise ValueError(f"'{image_name}' must map to a dictionary.")

        adjustments = image_data.get("adjustments")
        if not isinstance(adjustments, dict):
            raise ValueError(f"'{image_name}' is missing 'adjustments' dictionary.")

        for band_key, band_vals in adjustments.items():
            if not band_key.startswith("band_"):
                raise ValueError(
                    f"Invalid band key '{band_key}' in adjustments for '{image_name}'."
                )
            if not {"scale", "offset"} <= band_vals.keys():
                raise ValueError(
                    f"Missing 'scale' or 'offset' in adjustments[{band_key}] for '{image_name}'."
                )

        whole_stats = image_data.get("whole_stats")
        if not isinstance(whole_stats, dict):
            raise ValueError(f"'{image_name}' is missing 'whole_stats' dictionary.")

        for band_key, stat_vals in whole_stats.items():
            if not band_key.startswith("band_"):
                raise ValueError(
                    f"Invalid band key '{band_key}' in whole_stats for '{image_name}'."
                )
            if not {"mean", "std", "size"} <= stat_vals.keys():
                raise ValueError(
                    f"Missing 'mean', 'std', or 'size' in whole_stats[{band_key}] for '{image_name}'."
                )

        overlap_stats = image_data.get("overlap_stats", {})
        if not isinstance(overlap_stats, dict):
            raise ValueError(
                f"'overlap_stats' for '{image_name}' must be a dictionary if present."
            )

        for other_image, bands in overlap_stats.items():
            if not isinstance(bands, dict):
                raise ValueError(
                    f"'overlap_stats[{other_image}]' for '{image_name}' must be a dictionary."
                )
            for band_key, stat_vals in bands.items():
                if not band_key.startswith("band_"):
                    raise ValueError(
                        f"Invalid band key '{band_key}' in overlap_stats[{other_image}] for '{image_name}'."
                    )
                if not {"mean", "std", "size"} <= stat_vals.keys():
                    raise ValueError(
                        f"Missing 'mean', 'std', or 'size' in overlap_stats[{other_image}][{band_key}] for '{image_name}'."
                    )
    print("Loaded adjustments structure passed validation")


def _print_constraint_system(
    constraint_matrix: np.ndarray,
    adjustment_params: np.ndarray,
    observed_values_vector: np.ndarray,
    overlap_pairs: tuple,
    image_names_with_id: list[tuple[int, str]],
) -> None:
    """
    Prints the constraint matrix system with labeled rows and columns for debugging regression inputs.

    Args:
        constraint_matrix (ndarray): Coefficient matrix used in the regression system.
        adjustment_params (ndarray): Solved adjustment parameters (regression output).
        observed_values_vector (ndarray): Target values in the regression system.
        overlap_pairs (tuple): Pairs of overlapping image indices used in constraints.
        image_names_with_id (list of tuple): List of (ID, name) pairs corresponding to each image's position in the system.

    Returns:
        None
    """
    np.set_printoptions(
        suppress=True,
        precision=3,
        linewidth=300,
        formatter={"float_kind": lambda x: f"{x: .3f}"},
    )

    print("constraint_matrix with labels:")

    name_to_id = {n: i for i, n in image_names_with_id}

    # Build row labels
    row_labels = []
    for i, j in overlap_pairs:
        row_labels.append(f"Overlap({name_to_id[i]}-{name_to_id[j]}) Mean Diff")
        row_labels.append(f"Overlap({name_to_id[i]}-{name_to_id[j]}) Std Diff")

    for i, name in image_names_with_id:
        row_labels.append(f"[{i}] Mean Cnstr")
        row_labels.append(f"[{i}] Std Cnstr")

    # Build column labels
    col_labels = []
    for i, name in image_names_with_id:
        col_labels.append(f"a{i}")
        col_labels.append(f"b{i}")

    # Print column headers
    header = f"{'':<30}"
    for lbl in col_labels:
        header += f"{lbl:>18}"
    print(header)

    # Print matrix rows
    for row_label, row in zip(row_labels, constraint_matrix):
        line = f"{row_label:<30}"
        for val in row:
            line += f"{val:18.3f}"
        print(line)

    print("\nadjustment_params:")
    np.savetxt(sys.stdout, adjustment_params, fmt="%18.3f")

    print("\nobserved_values_vector:")
    np.savetxt(sys.stdout, observed_values_vector, fmt="%18.3f")


def _find_overlaps(
    image_bounds_dict: dict[str, tuple[float, float, float, float]],
) -> tuple[tuple[str, str], ...]:
    """
    Finds all pairs of image names with overlapping spatial bounds.

    Args:
        image_bounds_dict: Map of image name -> (minx, miny, maxx, maxy).

    Returns:
        Tuple of (name_i, name_j) pairs with overlapping extents.
    """
    overlaps: list[tuple[str, str]] = []
    keys = sorted(image_bounds_dict)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            minx1, miny1, maxx1, maxy1 = image_bounds_dict[k1]
            minx2, miny2, maxx2, maxy2 = image_bounds_dict[k2]

            if (minx1 < maxx2 and maxx1 > minx2 and
                miny1 < maxy2 and maxy1 > miny2):
                overlaps.append((k1, k2))

    return tuple(overlaps)


def _overlap_stats_process_image(
    tile_thread_on: bool,
    tile_thread_workers: int,
    num_bands: int,
    input_image_path_i: str,
    input_image_path_j: str,
    name_i: str,
    name_j: str,
    bound_i: tuple[float, float, float, float],
    bound_j: tuple[float, float, float, float],
    estimate_stats: bool,
    debug_logs: bool = False,
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    """
    Computes per-band overlap statistics between two images using alpha masks, without VRT recursion.

    Args:
        tile_thread_on (bool): Enable multithreaded GDAL warp/translate operations for per-tile work.
        tile_thread_workers (int): Number of worker threads when `tile_thread_on=True`.
        num_bands (int): Number of data bands to analyze.
        input_image_path_i (str): Path to image I.
        input_image_path_j (str): Path to image J.
        name_i (str): Basename (no extension) for image I; used as a key in outputs.
        name_j (str): Basename (no extension) for image J; used as a key in outputs.
        bound_i: (minx, miny, maxx, maxy) bounds for image I (dataset CRS).
        bound_j: (minx, miny, maxx, maxy) bounds for image J (dataset CRS).
        estimate_stats (bool): If True, use GDAL’s approximate statistics; if False, use exact.
        debug_logs (bool, optional): If True, print progress and intermediate details. Defaults to False.

    Returns:
        Dict[str, Dict[str, Dict[int, Dict[str, float]]]]: Nested mapping of overlap stats:
            {
              name_i: {
                name_j: {
                  band_index: {"mean": float, "std": float, "size": int},  # band_index is 0-based
                  ...
                }
              },
              name_j: {
                name_i: { ... }  # symmetric entries
              }
            }
    """
    # Open I to get metadata
    ds_i_full = gdal.Open(input_image_path_i, gdal.GA_ReadOnly)
    proj_i = ds_i_full.GetProjectionRef()
    gt_i = ds_i_full.GetGeoTransform()
    px_w_i, px_h_i = gt_i[1], abs(gt_i[5])
    ds_i_full = None

    # Overlap bbox
    minx_i, miny_i, maxx_i, maxy_i = bound_i
    minx_j, miny_j, maxx_j, maxy_j = bound_j
    x_min = max(minx_i, minx_j)
    x_max = min(maxx_i, maxx_j)
    y_min = max(miny_i, miny_j)
    y_max = min(maxy_i, maxy_j)
    if (x_min >= x_max) or (y_min >= y_max):
        return {name_i: {name_j: {}}, name_j: {name_i: {}}}

    with tempfile.TemporaryDirectory(prefix="spectralmatch_adjust_") as tmpdir:
        # J Base: Warp J to I grid over bbox
        j_base = os.path.join(tmpdir, f"{name_j}_on_{name_i}.vrt")
        j_ds = gdal.Warp(
            j_base,
            input_image_path_j,
            options=gdal.WarpOptions(
                format="VRT",
                dstSRS=proj_i,
                dstAlpha=True,
                outputBounds=(x_min, y_min, x_max, y_max),
                xRes=px_w_i, yRes=px_h_i,
                resampleAlg=gdal.GRIORA_NearestNeighbour,
                multithread=tile_thread_on,
                warpOptions=(["SKIP_NOSOURCE=YES", "UNIFIED_SRC_NODATA=YES"] + ([f"NUM_THREADS={tile_thread_workers}"] if tile_thread_on else [])),
            ),
        )
        if j_ds is None:
            raise RuntimeError("Warp failed for J when calculating overlap stats")
        j_ds = None

        # I base
        i_base = os.path.join(tmpdir, f"{name_i}_overlap.vrt")
        i_ds = gdal.Translate(
            i_base,
            input_image_path_i,
            options=gdal.TranslateOptions(format="VRT", projWin=[x_min, y_max, x_max, y_min]),
        )
        if i_ds is None:
            raise RuntimeError("Translate failed for I overlap")
        i_ds = None

        # Overlap size
        ds = gdal.Open(i_base)
        RX, RY = ds.RasterXSize, ds.RasterYSize
        ds = None

        # Combined mask VRT: mask = min(I.mask, J.mask)
        mask_vrt = os.path.join(tmpdir, "combined_mask.vrt")
        mask_xml = f"""<VRTDataset rasterXSize="{RX}" rasterYSize="{RY}">
      <SRS>{proj_i}</SRS>
      <VRTRasterBand dataType="Byte" subClass="VRTDerivedRasterBand" band="1">
        <ColorInterp>Alpha</ColorInterp>
        <PixelFunctionType>min</PixelFunctionType>
        <SimpleSource>
          <SourceFilename relativeToVRT="1">{os.path.basename(i_base)}</SourceFilename>
          <SourceBand>mask,1</SourceBand>
        </SimpleSource>
        <SimpleSource>
          <SourceFilename relativeToVRT="1">{os.path.basename(j_base)}</SourceFilename>
          <SourceBand>mask,1</SourceBand>
        </SimpleSource>
      </VRTRasterBand>
    </VRTDataset>
    """
        with open(mask_vrt, "w", encoding="utf-8") as f:
            f.write(mask_xml)

        # Make *stats* VRT copies and attach combined mask
        def _make_stats_vrt(base_vrt: str, out_vrt: str):
            with open(base_vrt, "r", encoding="utf-8") as f:
                xml = f.read()

            # Remove any existing NoDataValue tags
            xml = re.sub(r"<NoDataValue>.*?</NoDataValue>", "", xml)

            # Create mask band block referencing the combined mask
            mask_block = (
                "<MaskBand>\n"
                '  <VRTRasterBand dataType="Byte" subClass="VRTSourcedRasterBand">\n'
                "    <SimpleSource>\n"
                f'      <SourceFilename relativeToVRT="1">{os.path.basename(mask_vrt)}</SourceFilename>\n'
                "      <SourceBand>1</SourceBand>\n"
                f'      <SourceProperties RasterXSize="{RX}" RasterYSize="{RY}" DataType="Byte" BlockXSize="256" BlockYSize="256"/>\n'
                f'      <SrcRect xOff="0" yOff="0" xSize="{RX}" ySize="{RY}"/>\n'
                f'      <DstRect xOff="0" yOff="0" xSize="{RX}" ySize="{RY}"/>\n'
                "    </SimpleSource>\n"
                "  </VRTRasterBand>\n"
                "</MaskBand>\n"
            )

            # Replace existing MaskBand or add it after VRTDataset opening tag
            if "<MaskBand>" in xml:
                xml = re.sub(r"<MaskBand>.*?</MaskBand>", mask_block, xml, flags=re.S)
            else:
                xml = re.sub(r"(<VRTDataset[^>]*>)", r"\1\n" + mask_block, xml, count=1)

            with open(out_vrt, "w", encoding="utf-8") as f:
                f.write(xml)

        i_stats = os.path.join(tmpdir, f"{name_i}_stats.vrt")
        j_stats = os.path.join(tmpdir, f"{name_j}_stats.vrt")
        _make_stats_vrt(i_base, i_stats)
        _make_stats_vrt(j_base, j_stats)

        # Open masked VRTs and compute statistics
        i_ds = gdal.Open(i_stats, gdal.GA_ReadOnly)
        j_ds = gdal.Open(j_stats, gdal.GA_ReadOnly)

        stats = {name_i: {name_j: {}}, name_j: {name_i: {}}}

        # Get valid pixel count from the first band's mask
        size_i = _get_valid_count(i_ds.GetRasterBand(1), estimate_stats)

        # No valid overlap pixels
        if size_i == 0:
            i_ds = None
            j_ds = None
            mean_i, std_i, mean_j, std_j = 0, 0, 0, 0

        # size_j = _get_valid_count(j_ds.GetRasterBand(1), estimate_stats)
        # if size_i != size_j: raise ValueError(f"Raster sizes differ: {size_i} vs {size_j}") # They should not differ but just in case

        # Compute statistics for each band
        for b in range(1, num_bands + 1):
            if size_i != 0:
                _, _, mean_i, std_i = i_ds.GetRasterBand(b).GetStatistics(1 if estimate_stats else 0, 1)
                _, _, mean_j, std_j = j_ds.GetRasterBand(b).GetStatistics(1 if estimate_stats else 0, 1)

            stats[name_i][name_j][b - 1] = {"mean": float(mean_i), "std": float(std_i), "size": size_i}
            stats[name_j][name_i][b - 1] = {"mean": float(mean_j), "std": float(std_j), "size": size_i}

        i_ds = None
        j_ds = None
        return stats


def _whole_stats_process_image(
    tile_thread_on: bool,
    tile_thread_worker: int,
    input_image_path: str,
    num_bands: int,
    image_name: str,
    estimate_stats: bool,
    debug_logs: bool,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Computes whole-image statistics (mean, standard deviation, valid pixel count) for each band of a masked raster.

    Args:
        tile_thread_on (bool): Enable multithreaded GDAL operations for tile-level work.
        tile_thread_worker (int): Number of worker threads if `tile_thread_on=True`.
        input_image_path (str): Path to an input raster (VRT with alpha/nodata applied).
        num_bands (int): Number of data bands to compute stats for (alpha excluded).
        image_name (str): Basename (no extension) of the image; used as key in output.
        estimate_stats (bool): If True, use GDAL’s approximate statistics; if False, compute exact statistics.
        debug_logs (bool): If True, print detailed per-band statistics.

    Returns:
        Dict[str, Dict[int, Dict[str, float]]]: Mapping of image name to per-band stats:
            {
              image_name: {
                band_index: {
                  "mean": float,  # band mean
                  "std": float,   # band standard deviation
                  "size": int     # count of valid pixels (shared across bands)
                },
                ...
              }
            }
    """
    stats = {image_name: {}}

    ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)

    # Assumed valid & shared across bands
    valid_count = _get_valid_count(ds.GetRasterBand(1), estimate_stats)

    # Per-band stats
    for b in range(1, num_bands + 1):
        rb = ds.GetRasterBand(b)
        _min, _max, mean, std = rb.ComputeStatistics(approx_ok=estimate_stats)
        stats[image_name][b - 1] = {
            "mean": float(mean),
            "std": float(std),
            "size": valid_count,
        }
    ds = None
    return stats
