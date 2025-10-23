import rasterio
import fiona
import os
import numpy as np
import sys
import json
from skimage.transform import resize

from concurrent.futures import as_completed
from typing import List, Tuple, Literal
from numpy import ndarray
from scipy.optimize import least_squares
from rasterio.windows import Window
from rasterio.transform import rowcol
from rasterio.features import geometry_mask
from rasterio.coords import BoundingBox
from tqdm import tqdm

from ..handlers import _resolve_paths, _get_nodata_value, _check_raster_requirements
from ..utils_multiprocessing import (
    _resolve_parallel_config,
    _resolve_windows,
    _get_executor,
    WorkerContext,
)
from ..types_and_validation import Universal, Match


def global_regression(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    *,
    calculation_dtype: Universal.CalculationDtype = "float32",
    output_dtype: Universal.CustomOutputDtype = None,
    vector_mask: Universal.VectorMask = None,
    debug_logs: Universal.DebugLogs = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
    image_parallel_workers: Universal.ImageParallelWorkers = None,
    window_parallel_workers: Universal.WindowParallelWorkers = None,
    window_size: Universal.WindowSize = None,
    save_as_cog: Universal.SaveAsCog = False,
    specify_model_images: Match.SpecifyModelImages = None,
    custom_mean_factor: float = 1.0,
    custom_std_factor: float = 1.0,
    save_adjustments: str | None = None,
    load_adjustments: str | None = None,
    min_overlap_size: int | None = None,
) -> list:
    """
    Performs global radiometric normalization across overlapping images using least squares regression.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Global.tif), ["/input/one.tif", "/input/two.tif"].
        calculation_dtype (str, optional): Data type used for internal calculations. Defaults to "float32".
        output_dtype (str | None, optional): Data type for output rasters. Defaults to input image dtype.
        vector_mask (Tuple[Literal["include", "exclude"], str, Optional[str]] | None): Mask to limit stats calculation to specific areas in the format of a tuple with two or three items: literal "include" or "exclude" the mask area, str path to the vector file, optional str of field name in vector file that *includes* (can be substring) input image name to filter geometry by. Loaded stats won't have this applied to them. The matching solution is still applied to these areas in the output. Defaults to None for no mask.
        debug_logs (bool, optional): If True, prints debug information and constraint matrices. Defaults to False.
        custom_nodata_value (float | int | None, optional): Overrides detected NoData value. Defaults to None.
        image_parallel_workers (Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None = None): Parallelization strategy at the image level. Provide a tuple like ("process", "cpu") to use multiprocessing with all available cores. Threads are supported too. Set to None to disable.
        window_parallel_workers (Tuple[Literal["process"], Literal["cpu"] | int] | None = None): Parallelization strategy at the window level within each image. Same format as image_parallel_workers. Threads are not supported. Set to None to disable.
        window_size (int | Tuple[int, int] | Literal["internal"] | None): Tile size for reading and writing: int for square tiles, tuple for (width, height), "internal" to use raster's native tiling, or None for full image. "internal" enables efficient streaming from COGs.
        save_as_cog (bool): If True, saves output as a Cloud-Optimized GeoTIFF using proper band and block order.
        specify_model_images (Tuple[Literal["exclude", "include"], List[str]] | None ): First item in tuples sets weather to 'include' or 'exclude' the listed images from model building statistics. Second item is the list of image names (without their extension) to apply criteria to. For example, if this param is only set to 'include' one image, all other images will be matched to that one image. Defaults to no exclusion.
        custom_mean_factor (float, optional): Weight for mean constraints in regression. Defaults to 1.0.
        custom_std_factor (float, optional): Weight for standard deviation constraints in regression. Defaults to 1.0.
        save_adjustments (str | None, optional): The output path of a .json file to save adjustments parameters. Defaults to not saving.
        load_adjustments (str | None, optional): If set, loads saved whole and overlapping statistics only for images that exist in the .json file. Other images will still have their statistics calculated. Defaults to None.
        min_overlap_size: Minimum overlap size for each overlap. Defaults to None for no minimum overlap size.

    Returns:
        List[str]: Paths to the globally adjusted output raster images.
    """

    print("Start global regression")

    # Validate params
    Universal.validate(
        input_images=input_images,
        output_images=output_images,
        save_as_cog=save_as_cog,
        debug_logs=debug_logs,
        vector_mask=vector_mask,
        window_size=window_size,
        custom_nodata_value=custom_nodata_value,
        image_parallel_workers=image_parallel_workers,
        window_parallel_workers=window_parallel_workers,
        calculation_dtype=calculation_dtype,
        output_dtype=output_dtype,
    )

    Match.validate_match(
        specify_model_images=specify_model_images,
    )

    Match.validate_global_regression(
        custom_mean_factor=custom_mean_factor,
        custom_std_factor=custom_std_factor,
        save_adjustments=save_adjustments,
        load_adjustments=load_adjustments,
    )

    # Input and output paths
    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_Global.tif",
        },
    )

    if debug_logs:
        print(f"Input images: {input_image_paths}")
    if debug_logs:
        print(f"Output images: {output_image_paths}")

    input_image_names = [
        os.path.splitext(os.path.basename(p))[0] for p in input_image_paths
    ]
    input_image_path_pairs = dict(zip(input_image_names, input_image_paths))
    output_image_path_pairs = dict(zip(input_image_names, output_image_paths))

    # Check raster requirements
    _check_raster_requirements(
        input_image_paths,
        debug_logs,
        check_geotransform=True,
        check_crs=True,
        check_bands=True,
        check_nodata=True,
    )

    nodata_val = _get_nodata_value(
        list(input_image_path_pairs.values()), custom_nodata_value
    )

    # Determine multiprocessing and worker count
    image_parallel, image_backend, image_max_workers = _resolve_parallel_config(
        image_parallel_workers
    )
    window_parallel, window_backend, window_max_workers = _resolve_parallel_config(
        window_parallel_workers
    )

    # Find loaded and input files if load adjustments
    loaded_model = {}
    if load_adjustments:
        with open(load_adjustments, "r") as f:
            loaded_model = json.load(f)
        _validate_adjustment_model_structure(loaded_model)
        loaded_names = set(loaded_model.keys())
        input_names = set(input_image_names)
    else:
        loaded_names = set([])
        input_names = set(input_image_names)

    matched = input_names & loaded_names
    only_loaded = loaded_names - input_names
    only_input = input_names - loaded_names
    if debug_logs:
        print(
            f"Total images: input images: {len(input_names)}, loaded images {len(loaded_names)}: "
        )
        print(
            f"    Matched adjustments (to override) ({len(matched)}):", sorted(matched)
        )
        print(
            f"    Only in loaded adjustments (to add) ({len(only_loaded)}):",
            sorted(only_loaded),
        )
        print(
            f"    Only in input (to calculate) ({len(only_input)}):", sorted(only_input)
        )

    # Find images to include in model
    included_names = list(matched | only_loaded | only_input)
    if specify_model_images:
        mode, names = specify_model_images
        name_set = set(names)
        if mode == "include":
            included_names = [n for n in input_image_names if n in name_set]
        elif mode == "exclude":
            included_names = [n for n in input_image_names if n not in name_set]
        excluded_names = [n for n in input_image_names if n not in included_names]
    if debug_logs:
        print("Images to influence the model:")
        print(
            f"    Included in model ({len(included_names)}): {sorted(included_names)}"
        )
        if specify_model_images:
            print(
                f"    Excluded from model ({len(excluded_names)}): {sorted(excluded_names)}"
            )
        else:
            print(f"    Excluded from model (0): []")

    if debug_logs:
        print("Calculating statistics")
    with rasterio.open(list(input_image_path_pairs.values())[0]) as src:
        num_bands = src.count

    # Get images bounds
    all_bounds = {}
    for name, path in input_image_path_pairs.items():
        with rasterio.open(path) as ds:
            all_bounds[name] = ds.bounds

    # Overlap stats
    overlapping_pairs = _find_overlaps(all_bounds)
    all_overlap_stats = {}

    # Load overlap stats
    if load_adjustments:
        for name_i, model_entry in loaded_model.items():
            if name_i not in input_image_path_pairs:
                continue

            for name_j, bands in model_entry.get("overlap_stats", {}).items():
                if name_j not in input_image_path_pairs:
                    continue

                all_overlap_stats.setdefault(name_i, {})[name_j] = {
                    int(k.split("_")[1]): {
                        "mean": bands[k]["mean"],
                        "std": bands[k]["std"],
                        "size": bands[k]["size"],
                    }
                    for k in bands
                }

    # Calculate overlap stats
    parallel_args = [
        (
            window_parallel,
            window_max_workers,
            window_backend,
            num_bands,
            input_image_path_pairs[name_i],
            input_image_path_pairs[name_j],
            name_i,
            name_j,
            all_bounds[name_i],
            all_bounds[name_j],
            nodata_val,
            nodata_val,
            vector_mask,
            window_size,
            debug_logs,
        )
        for name_i, name_j in overlapping_pairs
        if name_i not in loaded_model
        or name_j not in loaded_model.get(name_i, {}).get("overlap_stats", {})
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [
                executor.submit(_overlap_stats_process_image, *args)
                for args in parallel_args
            ]
            for future in as_completed(futures):
                stats = future.result()
                for outer, inner in stats.items():
                    all_overlap_stats.setdefault(outer, {}).update(inner)
    else:
        for args in parallel_args:
            stats = _overlap_stats_process_image(*args)
            for outer, inner in stats.items():
                all_overlap_stats.setdefault(outer, {}).update(inner)

    # Load whole stats
    all_whole_stats = {
        name: {
            int(k.split("_")[1]): {
                "mean": loaded_model[name]["whole_stats"][k]["mean"],
                "std": loaded_model[name]["whole_stats"][k]["std"],
                "size": loaded_model[name]["whole_stats"][k]["size"],
            }
            for k in loaded_model[name]["whole_stats"]
        }
        for name in input_image_path_pairs
        if name in loaded_model
    }

    # Calculate whole stats
    parallel_args = [
        (
            window_parallel,
            window_max_workers,
            window_backend,
            image_path,
            nodata_val,
            num_bands,
            image_name,
            vector_mask,
            window_size,
            debug_logs,
        )
        for image_name, image_path in input_image_path_pairs.items()
        if image_name not in loaded_model
    ]

    # Compute whole stats
    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [
                executor.submit(_whole_stats_process_image, *args)
                for args in parallel_args
            ]
            for future in as_completed(futures):
                result = future.result()
                all_whole_stats.update(result)
    else:
        for args in parallel_args:
            result = _whole_stats_process_image(*args)
            all_whole_stats.update(result)

    # Get image names
    all_image_names = list(dict.fromkeys(input_image_names + list(loaded_model.keys())))
    num_total = len(all_image_names)

    # Print model sources
    if debug_logs:
        print(
            f"\nCreating model for {len(all_image_names)} total images from {len(included_names)} included:"
        )
        print(f"    {'ID':<4}\t{'Source':<6}\t{'Inclusion':<8}\tName")
        for i, name in enumerate(all_image_names):
            source = "load" if name in (matched | only_loaded) else "calc"
            included = "incl" if name in included_names else "excl"
            print(f"    {i:<4}\t{source:<6}\t{included:<8}\t{name}")

    # Build model
    all_params = _solve_global_model(
        num_bands,
        num_total,
        all_image_names,
        included_names,
        input_image_names,
        all_overlap_stats,
        all_whole_stats,
        custom_mean_factor,
        custom_std_factor,
        overlapping_pairs,
        min_overlap_size,
        debug_logs,
    )

    # Save adjustments
    if save_adjustments:
        _save_adjustments(
            save_path=save_adjustments,
            input_image_names=list(input_image_path_pairs.keys()),
            all_params=all_params,
            all_whole_stats=all_whole_stats,
            all_overlap_stats=all_overlap_stats,
            num_bands=num_bands,
            calculation_dtype=calculation_dtype,
        )

    # Apply corrections
    if debug_logs:
        print(f"Apply adjustments and saving results for:")
    parallel_args = [
        (
            name,
            img_path,
            output_image_path_pairs[name],
            np.array([all_params[b, 2 * idx, 0] for b in range(num_bands)]),
            np.array([all_params[b, 2 * idx + 1, 0] for b in range(num_bands)]),
            num_bands,
            nodata_val,
            window_size,
            calculation_dtype,
            output_dtype,
            window_parallel,
            window_backend,
            window_max_workers,
            save_as_cog,
            debug_logs,
        )
        for idx, (name, img_path) in enumerate(input_image_path_pairs.items())
    ]

    if image_parallel:
        with _get_executor(image_backend, image_max_workers) as executor:
            futures = [
                executor.submit(_apply_adjustments_process_image, *args)
                for args in parallel_args
            ]
            for future in as_completed(futures):
                future.result()
    else:
        for args in parallel_args:
            _apply_adjustments_process_image(*args)

    return output_image_paths


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
    min_overlap_size: int | None = None,
    debug_logs: bool = False,
    weight_by_size: bool = True,
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
        min_overlap_size: Minimum overlap size for each overlap. Defaults to None for no minimum overlap size.
        debug_logs: If True, prints debug information.

    Returns:
        np.ndarray: Adjustment parameters of shape (bands, 2 * num_images, 1).
    """
    all_params = np.zeros((num_bands, 2 * num_total, 1), dtype=float)
    image_names_with_id = [(i, name) for i, name in enumerate(all_image_names)]
    for b in range(num_bands):
        if debug_logs: print(f"\nProcessing band {b}:")
        A, y = [], []
        tot_overlap, overlap_count = 0, 0  # track both

        for i, name_i in image_names_with_id:
            for j, name_j in image_names_with_id[i + 1:]:
                stat = all_overlap_stats.get(name_i, {}).get(name_j)
                if stat is None: continue
                if name_i not in included_names and name_j not in included_names: continue

                s = stat[b]["size"]
                if min_overlap_size is not None and s < min_overlap_size: continue

                m1, v1 = stat[b]["mean"], stat[b]["std"]
                m2, v2 = all_overlap_stats[name_j][name_i][b]["mean"], all_overlap_stats[name_j][name_i][b]["std"]

                row_m = [0]*(2*num_total); row_s = [0]*(2*num_total)
                row_m[2*i:2*i+2] = [m1, 1]; row_m[2*j:2*j+2] = [-m2, -1]
                row_s[2*i], row_s[2*j] = v1, -v2

                w = (s if weight_by_size else 1.0)  # <- no size weighting when False
                A.extend([[v*w*custom_mean_factor for v in row_m],
                          [v*w*custom_std_factor for v in row_s]])
                y.extend([0, 0])

                tot_overlap += s
                overlap_count += 1

        # Whole-image constraint balancing:
        # original used tot_overlap/(2*num_total); unweighted uses overlap_count instead
        if weight_by_size:
            pjj = 1.0 if tot_overlap == 0 else tot_overlap/(2.0*num_total)
        else:
            pjj = 1.0 if overlap_count == 0 else overlap_count/(2.0*num_total)

        for name in included_names:
            mj = all_whole_stats[name][b]["mean"]; vj = all_whole_stats[name][b]["std"]
            j_idx = all_image_names.index(name)
            row_m = [0]*(2*num_total); row_s = [0]*(2*num_total)
            row_m[2*j_idx:2*j_idx+2] = [mj*pjj, 1*pjj]; row_s[2*j_idx] = vj*pjj
            A.extend([row_m, row_s]); y.extend([mj*pjj, vj*pjj])

        for name in input_image_names:
            if name in included_names: continue
            row = [0]*(2*num_total); A.append(row.copy()); y.append(0); A.append(row.copy()); y.append(0)

        A_arr = np.asarray(A); y_arr = np.asarray(y)
        res = least_squares(lambda p: A_arr @ p - y_arr, [1, 0]*num_total)
        all_params[b, :, 0] = res.x

        if debug_logs:
            _print_constraint_system(A_arr, res.x, y_arr, overlapping_pairs, image_names_with_id)

    return all_params


def _apply_adjustments_process_image(
    image_name: str,
    input_image_path: str,
    output_image_path: str,
    scale: np.ndarray,
    offset: np.ndarray,
    num_bands: int,
    nodata_val: int | float,
    window_size: Universal.WindowSizeWithBlock,
    calculation_dtype: str,
    output_dtype: str | None,
    window_parallel: bool,
    window_backend: str,
    window_max_workers: int,
    save_as_cog: Universal.SaveAsCog,
    debug_logs: bool = False,
):
    """
    Applies scale and offset adjustments to each band of an input image and writes the result to the output path.

    Args:
        image_name: Identifier for the image in the worker context.
        input_image_path: Path to the input raster image.
        output_image_path: Path to save the adjusted output image.
        scale: Per-band scale factors (1D array of length num_bands).
        offset: Per-band offset values (1D array of length num_bands).
        num_bands: Number of image bands.
        nodata_val: NoData value to preserve during adjustment.
        window_size: Tiling strategy for processing (None, int, tuple, or "internal").
        calculation_dtype: Data type for computation.
        output_dtype: Output data type (defaults to input type if None).
        window_parallel: Whether to parallelize over windows.
        window_backend: Backend to use for window-level parallelism ("process").
        window_max_workers: Number of parallel workers for window processing.
        debug_logs: If True, prints debug info during processing.

    Returns:
        None
    """
    if debug_logs:
        print(f"    Processing {image_name}")

    with rasterio.open(input_image_path) as src:
        block_y, block_x = (
            w := next(iter(_resolve_windows(src, window_size)))
        ).height, w.width

        meta = src.meta.copy()
        if save_as_cog:
            meta.update(
                {
                    "count": num_bands,
                    "dtype": output_dtype or src.dtypes[0],
                    "nodata": nodata_val,
                    "driver": "GTiff",
                    "BIGTIFF": "IF_SAFER",
                    "BLOCKXSIZE": block_x,
                    "BLOCKYSIZE": block_y,
                    "COMPRESS": "DEFLATE",
                    "TILED": True,
                }
            )
        else:
            meta.update(
                {
                    "count": num_bands,
                    "dtype": output_dtype or src.dtypes[0],
                    "nodata": nodata_val,
                    "driver": "GTiff",
                }
            )

        if os.path.exists(output_image_path):
            os.remove(output_image_path)
        windows = _resolve_windows(src, window_size)
        args = [
            (
                window,
                band,
                scale[band],
                offset[band],
                nodata_val,
                calculation_dtype,
                debug_logs,
                image_name,
            )
            for window in windows
            for band in range(num_bands)
        ]
        with rasterio.open(output_image_path, "w", **meta) as dst:
            if window_parallel:
                with _get_executor(
                        window_backend,
                        window_max_workers,
                        initializer=WorkerContext.init,
                        initargs=({image_name: ("raster", input_image_path)},),
                ) as executor:
                    future_to_params = {
                        executor.submit(_apply_adjustments_process_window, *arg): (arg[0], arg[1])  # window, band
                        for arg in args
                    }
                    future_iter = tqdm(
                        as_completed(future_to_params),
                        total=len(future_to_params),
                        desc="Apply Adjustments"
                    ) if debug_logs else as_completed(future_to_params)

                    for future in future_iter:
                        band, window, buf = future.result()
                        dst.write(buf.astype(meta["dtype"]), band + 1, window=window)
            else:
                WorkerContext.init({image_name: ("raster", input_image_path)})
                arg_iter = tqdm(args, desc="Apply Adjustments") if debug_logs else args
                for arg in arg_iter:
                    window, band, _, _, _, _, _, _ = arg
                    _, _, buf = _apply_adjustments_process_window(*arg)
                    dst.write(buf.astype(meta["dtype"]), band + 1, window=window)
                WorkerContext.close()


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


def _apply_adjustments_process_window(
    window: Window,
    band_idx: int,
    a: float,
    b: float,
    nodata: int | float,
    calculation_dtype: str,
    debug_logs: bool,
    image_name: str,
):
    """
    Applies a global linear transformation (scale and offset) to a raster tile.

    Args:
        window (Window): Rasterio window specifying the region to process.
        band_idx (int): Band index to read and adjust.
        a (float): Multiplicative factor for normalization.
        b (float): Additive offset for normalization.
        nodata (int | float): NoData value to ignore during processing.
        calculation_dtype (str): Data type to cast the block for computation.
        debug_logs (bool): If True, prints processing information.
        image_name (str): Key to fetch the raster from WorkerContext.

    Returns:
        Tuple[Window, np.ndarray]: Window and the adjusted data block.
    """

    ds = WorkerContext.get(image_name)
    block = ds.read(band_idx + 1, window=window).astype(calculation_dtype)

    mask = block != nodata
    block[mask] = a * block[mask] + b
    return band_idx, window, block


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
    image_bounds_dict: dict[str, rasterio.coords.BoundingBox],
) -> tuple[tuple[str, str], ...]:
    """
    Finds all pairs of image names with overlapping spatial bounds.

    Args:
        image_bounds_dict (dict): Dictionary mapping image names to their rasterio bounds.

    Returns:
        Tuple[Tuple[str, str], ...]: Pairs of image names with overlapping extents.
    """
    overlaps = []

    keys = sorted(image_bounds_dict)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            b1, b2 = image_bounds_dict[k1], image_bounds_dict[k2]

            if (
                b1.left < b2.right
                and b1.right > b2.left
                and b1.bottom < b2.top
                and b1.top > b2.bottom
            ):
                overlaps.append((k1, k2))

    return tuple(overlaps)


def _overlap_stats_process_image(
    parallel: bool,
    max_workers: int,
    backend: str,
    num_bands: int,
    input_image_path_i: str,
    input_image_path_j: str,
    name_i: str,
    name_j: str,
    bound_i: BoundingBox,
    bound_j: BoundingBox,
    nodata_i: int | float,
    nodata_j: int | float,
    vector_mask: (
        Tuple[Literal["include", "exclude"], str]
        | Tuple[Literal["include", "exclude"], str, str]
        | None
    ),
    window_size: int | Tuple[int, int] | Literal["internal"] | None,
    debug_logs: bool,
):
    """
    Computes per-band overlap statistics (mean, std, pixel count) between two images over their intersecting area.

    Args:
        parallel: Whether to use multiprocessing for window processing.
        max_workers: Number of workers for parallel processing.
        backend: Parallelization backend ("process").
        num_bands: Number of image bands.
        input_image_path_i: Path to the first image.
        input_image_path_j: Path to the second image.
        name_i: Identifier for the first image.
        name_j: Identifier for the second image.
        bound_i: BoundingBox of the first image.
        bound_j: BoundingBox of the second image.
        nodata_i: NoData value for the first image.
        nodata_j: NoData value for the second image.
        vector_mask: Optional mask to include/exclude regions, with optional field filter.
        window_size: Windowing strategy for tile processing.
        debug_logs: If True, prints overlap bounds and status.

    Returns:
        dict: Nested stats dictionary for each image pair and band.
    """

    stats = {name_i: {name_j: {}}, name_j: {name_i: {}}}

    with (
        rasterio.open(input_image_path_i) as src_i,
        rasterio.open(input_image_path_j) as src_j,
    ):
        # Parse geometry masks separately per image
        geoms_i = geoms_j = None
        invert = False
        if vector_mask:
            mode, path, *field = vector_mask
            invert = mode == "exclude"
            field_name = field[0] if field else None

            with fiona.open(path, "r") as vector:
                features = list(vector)
                if field_name:
                    geoms_i = [
                        f["geometry"]
                        for f in features
                        if field_name in f["properties"]
                        and name_i in str(f["properties"][field_name])
                    ]
                    geoms_j = [
                        f["geometry"]
                        for f in features
                        if field_name in f["properties"]
                        and name_j in str(f["properties"][field_name])
                    ]
                else:
                    geoms_i = geoms_j = [f["geometry"] for f in features]

        # Determine overlap bounds
        x_min = max(bound_i.left, bound_j.left)
        x_max = min(bound_i.right, bound_j.right)
        y_min = max(bound_i.bottom, bound_j.bottom)
        y_max = min(bound_i.top, bound_j.top)

        if debug_logs:
            print(
                f"Overlap bounds: x: {x_min:.2f} to {x_max:.2f}, y: {y_min:.2f} to {y_max:.2f}"
            )

        if x_min >= x_max or y_min >= y_max:
            return stats

        row_min_i, col_min_i = rowcol(src_i.transform, x_min, y_max)
        row_max_i, col_max_i = rowcol(src_i.transform, x_max, y_min)

        windows_image_i = _resolve_windows(src_i, window_size)
        fit_windows_image_i = _fit_windows_to_pixel_bounds(
            windows_image_i,
            row_min_i,
            row_max_i,
            col_min_i,
            col_max_i,
            row_min_i,
            col_min_i,
        )

        # Initialize stats structure
        stats = {name_i: {name_j: {}}, name_j: {name_i: {}}}
        completed = []

        # Generate all (window, band) combinations
        args = [
            (
                win,
                band,
                col_min_i,
                row_min_i,
                name_i,
                name_j,
                nodata_i,
                nodata_j,
                geoms_i,
                geoms_j,
                invert,
            )
            for win in fit_windows_image_i
            for band in range(num_bands)
        ]

        if parallel:
            with _get_executor(
                    backend,
                    max_workers,
                    initializer=WorkerContext.init,
                    initargs=(
                            {
                                name_i: ("raster", input_image_path_i),
                                name_j: ("raster", input_image_path_j),
                            },
                    ),
            ) as executor:
                future_to_params = {
                    executor.submit(_overlap_stats_process_window, *arg): arg[1]
                    for arg in args
                }
                future_iter = tqdm(
                    as_completed(future_to_params),
                    total=len(future_to_params),
                    desc="Overlap Stats"
                ) if debug_logs else as_completed(future_to_params)

                for future in future_iter:
                    result = future.result()
                    band = future_to_params[future]
                    if result is not None:
                        completed.append((band, result[0], result[1]))
        else:
            WorkerContext.init(
                {
                    name_i: ("raster", input_image_path_i),
                    name_j: ("raster", input_image_path_j),
                }
            )
            arg_iter = tqdm(args, desc="Overlap Stats") if debug_logs else args
            for arg in arg_iter:
                result = _overlap_stats_process_window(*arg)
                if result is not None:
                    band = arg[1]
                    completed.append((band, result[0], result[1]))
            WorkerContext.close()

        # Aggregate per-band statistics
        for band in range(num_bands):
            v_i = np.concatenate([arr_i for b, arr_i, _ in completed if b == band]) if completed else np.array([])
            v_j = np.concatenate([arr_j for b, _, arr_j in completed if b == band]) if completed else np.array([])

            stats[name_i][name_j][band] = {
                "mean": float(np.mean(v_i)) if v_i.size else 0,
                "std": float(np.std(v_i)) if v_i.size else 0,
                "size": int(v_i.size),
            }
            stats[name_j][name_i][band] = {
                "mean": float(np.mean(v_j)) if v_j.size else 0,
                "std": float(np.std(v_j)) if v_j.size else 0,
                "size": int(v_j.size),
            }
        return stats

def _overlap_stats_process_window(
    win: Window,
    band: int,
    col_min_i: int,
    row_min_i: int,
    name_i: str,
    name_j: str,
    nodata_i: float,
    nodata_j: float,
    geoms_i: list | None,
    geoms_j: list | None,
    invert: bool,
    interpolation_method: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Processes a single overlapping window between two images, applying masks and interpolation if needed,
    and returns valid pixel pairs for statistical comparison.

    Args:
        win: Window in image i's coordinate space.
        band: Band index to process.
        col_min_i: Column offset of overlap region in image i.
        row_min_i: Row offset of overlap region in image i.
        name_i: Image i identifier in WorkerContext.
        name_j: Image j identifier in WorkerContext.
        nodata_i: NoData value for image i.
        nodata_j: NoData value for image j.
        geoms_i: Optional list of geometries for masking image i.
        geoms_j: Optional list of geometries for masking image j.
        invert: Whether to invert the mask logic (exclude vs include).
        interpolation_method: OpenCV interpolation method for resampling. 1 is bilinear interpolation.

    Returns:
        Tuple of valid pixel arrays (image_i_values, image_j_values), or None if no valid pixels found.
    """

    src_i = WorkerContext.get(name_i)
    src_j = WorkerContext.get(name_j)

    win_i = Window(
        col_min_i + win.col_off, row_min_i + win.row_off, win.width, win.height
    )
    bounds = src_i.window_bounds(win_i)
    win_j = rasterio.windows.from_bounds(*bounds, transform=src_j.transform)

    block_i = src_i.read(band + 1, window=win_i)
    block_j = src_j.read(band + 1, window=win_j, boundless=True, fill_value=nodata_j)

    if np.all(block_i == nodata_i) or np.all(block_j == nodata_j):
        return None

    if geoms_i:
        transform_i_win = src_i.window_transform(win_i)
        mask_i = geometry_mask(
            geoms_i,
            transform=transform_i_win,
            invert=not invert,
            out_shape=block_i.shape,
        )
        block_i[~mask_i] = nodata_i

    if geoms_j:
        transform_j_win = src_j.window_transform(win_j)
        mask_j = geometry_mask(
            geoms_j,
            transform=transform_j_win,
            invert=not invert,
            out_shape=block_j.shape,
        )
        block_j[~mask_j] = nodata_j

    if block_j.shape != block_i.shape:
        block_j = resize(
            block_j,
            block_i.shape,
            order=interpolation_method,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(block_j.dtype)
    if block_j.shape != block_i.shape:
        raise ValueError(
            f"Block size mismatch after interpolation: block_i={block_i.shape}, block_j={block_j.shape}"
        )

    valid = (block_i != nodata_i) & (block_j != nodata_j)
    if np.any(valid):
        return block_i[valid], block_j[valid]

    return None


def _fit_windows_to_pixel_bounds(
    windows: list[Window],
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    row_offset: int,
    col_offset: int,
) -> list[Window]:
    """
    Adjusts a list of image-relative windows to ensure they fit within specified pixel bounds, based on a global offset.

    Args:
        windows: List of rasterio Window objects (relative to an image region).
        row_min, row_max: Global pixel row bounds to clip against.
        col_min, col_max: Global pixel column bounds to clip against.
        row_offset, col_offset: Offsets to convert window-relative coordinates to global coordinates.

    Returns:
        List[Window]: Windows cropped to the specified global bounds.
    """
    adjusted_windows = []
    for win in windows:
        win_row_start = row_offset + win.row_off
        win_row_end = win_row_start + win.height
        win_col_start = col_offset + win.col_off
        win_col_end = win_col_start + win.width

        clipped_row_start = max(win_row_start, row_min)
        clipped_row_end = min(win_row_end, row_max)
        clipped_col_start = max(win_col_start, col_min)
        clipped_col_end = min(win_col_end, col_max)

        new_width = clipped_col_end - clipped_col_start
        new_height = clipped_row_end - clipped_row_start

        win_row_off_adj = clipped_row_start - row_offset
        win_col_off_adj = clipped_col_start - col_offset

        if (
            new_width > 0
            and new_height > 0
            and win_row_off_adj >= 0
            and win_col_off_adj >= 0
        ):
            adjusted_windows.append(
                Window(
                    win_col_off_adj,
                    win_row_off_adj,
                    new_width,
                    new_height,
                )
            )

    return adjusted_windows


def _whole_stats_process_image(
    parallel: bool,
    max_workers: int,
    backend: str,
    input_image_path: str,
    nodata: int | float,
    num_bands: int,
    image_name: str,
    vector_mask: (
        Tuple[Literal["include", "exclude"], str]
        | Tuple[Literal["include", "exclude"], str, str]
        | None
    ) = None,
    window_size: int | Tuple[int, int] | Literal["internal"] | None = None,
    debug_logs: bool = False,
):
    """
    Calculates whole-image statistics (mean, std, and valid pixel count) per band, with optional masking and window-level parallelism.

    Args:
        parallel: Whether to enable multiprocessing for window processing.
        max_workers: Number of parallel workers to use.
        backend: Multiprocessing backend ("process").
        input_image_path: Path to the input raster.
        nodata: NoData value to exclude from stats.
        num_bands: Number of bands to process.
        image_name: Identifier for use in WorkerContext.
        vector_mask: Optional mask tuple to include/exclude specific regions, with optional field filter.
        window_size: Tiling strategy (None, int, tuple, or "internal").
        debug_logs: If True, prints debug messages.

    Returns:
        dict: Per-band statistics {image_name: {band: {mean, std, size}}}.
    """

    stats = {image_name: {}}

    with rasterio.open(input_image_path) as data:
        geoms = None
        invert = False

        if vector_mask:
            mode, path, *field = vector_mask
            invert = mode == "exclude"
            field_name = field[0] if field else None
            with fiona.open(path, "r") as vector:
                if field_name:
                    geoms = [
                        feat["geometry"]
                        for feat in vector
                        if field_name in feat["properties"]
                        and image_name in str(feat["properties"][field_name])
                    ]
                else:
                    geoms = [feat["geometry"] for feat in vector]

        if geoms and debug_logs:
            print("        Applied mask")

        for band_idx in range(num_bands):
            windows = _resolve_windows(data, window_size)

            args = [
                (win, band_idx, image_name, nodata, geoms, invert) for win in windows
            ]

            if parallel:
                with _get_executor(
                    backend,
                    max_workers,
                    initializer=WorkerContext.init,
                    initargs=({image_name: ("raster", input_image_path)},),
                ) as executor:
                    futures = [
                        executor.submit(_whole_stats_process_window, *arg)
                        for arg in args
                    ]
                    all_values = [
                        f.result()
                        for f in as_completed(futures)
                        if f.result() is not None
                    ]
            else:
                WorkerContext.init({image_name: ("raster", input_image_path)})
                all_values = []
                for arg in args:
                    values = _whole_stats_process_window(*arg)
                    if values is not None:
                        all_values.append(values)
                WorkerContext.close()

            if all_values:
                stacked = np.concatenate(all_values)
                mean = float(stacked.mean())
                std = float(stacked.std(ddof=1))
                count = int(stacked.size)
            else:
                mean = 0.0
                std = 0.0
                count = 0

            stats[image_name][band_idx] = {
                "mean": mean,
                "std": std,
                "size": count,
            }

    return stats


def _whole_stats_process_window(
    win: Window,
    band_idx: int,
    image_name: str,
    nodata: int | float,
    geoms: list | None,
    invert: bool,
) -> np.ndarray | None:
    """
    Extracts valid pixel values from a raster window, optionally applying a geometry mask.

    Args:
        win: Rasterio window to read.
        band_idx: Band index to read (0-based).
        image_name: Identifier for the image in WorkerContext.
        nodata: NoData value to exclude.
        geoms: Optional list of geometries for masking.
        invert: If True, inverts the mask (exclude instead of include).

    Returns:
        np.ndarray or None: 1D array of valid pixel values, or None if none found.
    """

    data = WorkerContext.get(image_name)
    block = data.read(band_idx + 1, window=win)
    if geoms:
        transform = data.window_transform(win)
        mask = geometry_mask(
            geoms,
            transform=transform,
            invert=not invert,
            out_shape=(int(win.height), int(win.width)),
        )
        block[~mask] = nodata
    valid = block != nodata
    return block[valid] if np.any(valid) else None
