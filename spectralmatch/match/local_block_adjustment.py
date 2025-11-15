import math
import tempfile
import numpy as np
import os

from scipy.ndimage import gaussian_filter
from concurrent.futures import as_completed
from typing import Tuple, Optional, List, Literal
from osgeo import gdal, osr

from ..utils import create_masked_vrts, _set_gdal_cache, _set_gdal_workers, _resolve_gdal_dtype, _resolve_window_size, \
    _gdal_dtype_str_to_enum, _get_valid_count
from ..handlers import _resolve_paths, _resolve_nodata_value, _check_raster_requirements
from ..utils_multiprocessing import _resolve_parallel_config, _get_executor
from ..types_and_validation import Universal, Match


def local_block_adjustment(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    *,
    calculation_dtype: Universal.CalculationDtype = "float32",
    output_dtype: Universal.CustomOutputDtype = None,
    vector_mask: Universal.VectorMask = None,
    debug_logs: Universal.DebugLogs = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
    cache: Universal.Cache = None,
    image_threads: Universal.Threads = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    window_size: Universal.WindowSize = None,
    save_as_cog: Universal.SaveAsCog = False,
    number_of_blocks: int | Tuple[int, int] | Literal["coefficient_of_variation"] = 100,
    alpha: float = 1.0,
    correction_method: Literal["gamma", "linear"] = "linear",
    save_block_maps: Tuple[str, str] | None = None,
    load_block_maps: (
        Tuple[str, List[str]] | Tuple[str, None] | Tuple[None, List[str]] | None
    ) = None,
    override_bounds_canvas_coords: Tuple[float, float, float, float] | None = None,
    block_valid_pixel_threshold: float = 0.001,
) -> list:
    """
    Performs local radiometric adjustment on a set of raster images using block-based statistics.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Global.tif), ["/input/one.tif", "/input/two.tif"].
        calculation_dtype (str, optional): Precision for internal calculations. Defaults to "float32".
        output_dtype (str | None, optional): Data type for output rasters. Defaults to input image dtype.
        vector_mask (Tuple[Literal["include", "exclude"], str, Optional[str]] | None): A mask limiting pixels to include when calculating stats for each block in the format of a tuple with two or three items: literal "include" or "exclude" the mask area, str path to the vector file, optional str of field name in vector file that *includes* (can be substring) input image name to filter geometry by. It is only applied when calculating local blocks, as the reference map is calculated as the mean of all local blocks. Loaded block maps won't have this applied unless it was used when calculating them. The matching solution is still applied to these areas in the output. Defaults to None for no mask.
        debug_logs (bool, optional): If True, prints debug info and progress. Defaults to False.
        custom_nodata_value (float | int | None, optional): Overrides detected NoData value. Defaults to None.
        cache (float | None): Controls GDAL cache size in GB. Defaults to preset cache size. Applied via GDAL_CACHEMAX.
        image_threads (Literal["cpu"] | int | None): Parallelism for per-image operations. "cpu" to get number of cores, int to assign number, and None to disable image level parallelism.
        io_threads (Literal["cpu"] | int | None): Parallelism for IO operations. "cpu" to get number of cores, int to assign number, and None to disable io level parallelism.
        tile_threads (Literal["cpu"] | int | None): "cpu" to get number of cores, int to assign number, and None to disable tile level parallelism.
        window_size (int | None): Output image tile size. Defaults to input image tile size.
        save_as_cog (bool): If True, saves output as a Cloud-Optimized GeoTIFF using proper band and block order.
        number_of_blocks (int | tuple | Literal["coefficient_of_variation"]): int as a target of blocks per image, tuple to set manually set total blocks width and height, coefficient_of_variation to find the number of blocks based on this metric.
        alpha (float, optional): Blending factor between reference and local means. Defaults to 1.0.
        correction_method (Literal["gamma", "linear"], optional): Local correction method. Defaults to "gamma".
        save_block_maps (tuple(str, str) | None): If enabled, saves block maps for review, to resume processing later, or to add additional images to the reference map.
            - First str is the path to save the global block map.
            - Second str is the path to save the local block maps, which must include "$" which will be replaced my the image name (because there are multiple local maps).
        load_block_maps (Tuple[str, List[str]] | Tuple[str, None] | Tuple[None, List[str]] | None, optional):
            Controls loading of precomputed block maps. Can be one of:
                - Tuple[str, List[str]]: Load both reference and local block maps.
                - Tuple[str, None]: Load only the reference block map.
                - Tuple[None, List[str]]: Load only the local block maps.
                - None: Do not load any block maps.
            This supports partial or full reuse of precomputed block maps:
                - Local block maps will still be computed for each input image that is not linked to a local block map by the images name being *included* in the local block maps name (file name).
                - The reference block map will only be calculated (mean of all local blocks) if not set.
                - The reference map defines the reference block statistics and the local maps define per-image local block statistics.
                - Both reference and local maps must have the same canvas extent and dimensions which will be used to set those values.
        override_bounds_canvas_coords (Tuple[float, float, float, float] | None): Manually set (min_x, min_y, max_x, max_y) bounds to override the computed/loaded canvas extent. If you wish to have a larger extent than the current images, you can manually set this, along with setting a fixed number of blocks, to anticipate images will expand beyond the current extent.
        block_valid_pixel_threshold (float): Minimum fraction of valid pixels required to include a block (0–1).

    Returns:
        List[str]: Paths to the locally adjusted output raster images.
    """

    print("Start local block adjustment")

    # Validate params
    Universal.validate(
        input_images=input_images,
        output_images=output_images,
        save_as_cog=save_as_cog,
        debug_logs=debug_logs,
        vector_mask=vector_mask,
        window_size=window_size,
        custom_nodata_value=custom_nodata_value,
        calculation_dtype=calculation_dtype,
        output_dtype=output_dtype,
        cache=cache,
        image_threads=image_threads,
        io_threads=io_threads,
        tile_threads=tile_threads,
    )

    Match.validate_local_block_adjustment(
        number_of_blocks=number_of_blocks,
        alpha=alpha,
        correction_method=correction_method,
        save_block_maps=save_block_maps,
        load_block_maps=load_block_maps,
        override_bounds_canvas_coords=override_bounds_canvas_coords,
        block_valid_pixel_threshold=block_valid_pixel_threshold,
    )

    # Setup gdal
    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_Local.tif",
        },
    )
    input_image_names = _resolve_paths("name", input_image_paths)

    input_image_path_pairs = dict(zip(input_image_names, input_image_paths))
    output_image_path_pairs = dict(zip(input_image_names, output_image_paths))
    if debug_logs: print(f"Input images: {input_image_paths}")
    if debug_logs: print(f"Output images: {output_image_paths}")

    input_image_path_pairs_masked = create_masked_vrts(input_image_path_pairs, vector_mask=vector_mask, debug_logs=debug_logs)

    # Dtype
    output_dtype = _resolve_gdal_dtype(output_dtype, input_image_paths[0])

    _check_raster_requirements(
        input_image_paths,
        debug_logs,
        check_geotransform=True,
        check_crs=True,
        check_bands=True,
        check_nodata=True,
    )

    nodata_val = _resolve_nodata_value(input_image_paths[0], custom_nodata_value)

    # Determine multiprocessing and worker count
    image_backend = "thread" # "thread" or "process"
    image_threads_on, image_thread_workers = _resolve_parallel_config(image_threads)
    tile_thread_on, tile_thread_workers = _resolve_parallel_config(tile_threads)

    if debug_logs:
        print(f"Global nodata value: {nodata_val}")
    num_bands = gdal.Open(next(iter(input_image_path_pairs.values()))).RasterCount

    # Load data from precomputed block maps if set
    if load_block_maps:
        (
            loaded_block_local_means,
            loaded_block_reference_mean,
            loaded_num_row,
            loaded_num_col,
            loaded_bounds_canvas_coords,
        ) = _get_pre_computed_block_maps(load_block_maps, calculation_dtype, debug_logs)
        loaded_names = list(loaded_block_local_means.keys())
        block_reference_mean = loaded_block_reference_mean

        matched = list(
            (
                soft_matches := {
                    input_name: loaded_name
                    for input_name in input_image_names
                    for loaded_name in loaded_names
                    if input_name in loaded_name
                }
            ).keys()
        )
        only_loaded = [
            l for l in loaded_names if not any(n in l for n in input_image_names)
        ]
        only_input = [
            n for n in input_image_names if not any(n in l for l in loaded_names)
        ]

    else:
        only_input = input_image_names
        matched = []
        only_loaded = []
        block_reference_mean = None

    if debug_logs:
        print(
            f"Total images: input images: {len(input_image_names)}, loaded local block maps: {len(loaded_names) if load_block_maps else 0}:"
        )
        print(
            f"    Matched local block maps (to override) ({len(matched)}):",
            sorted(matched),
        )
        print(
            f"    Only in loaded local block maps (to use) ({len(only_loaded)}):",
            sorted(only_loaded),
        )
        print(
            f"    Only in input (to compute) ({len(only_input)}):", sorted(only_input)
        )

    # Unpack path to save block maps
    if save_block_maps:
        reference_map_path, local_map_path = save_block_maps

    # Create image bounds dict not required

    # Get bounds canvas coords
    if not override_bounds_canvas_coords:
        if not load_block_maps:
            bounds_canvas_coords = _get_bounding_rectangle(input_image_paths)
        else:
            bounds_canvas_coords = loaded_bounds_canvas_coords
    else:
        bounds_canvas_coords = override_bounds_canvas_coords
        if load_block_maps:
            if bounds_canvas_coords != loaded_bounds_canvas_coords:
                raise ValueError(
                    "Override bounds canvas coordinates do not match loaded block maps bounds"
                )

    # Calculate the number of blocks
    if not load_block_maps:
        if isinstance(number_of_blocks, int):
            num_row, num_col = _compute_block_size(
                input_image_paths, number_of_blocks, bounds_canvas_coords
            )
        elif isinstance(number_of_blocks, tuple):
            num_row, num_col = number_of_blocks
        elif isinstance(number_of_blocks, str):
            num_row, num_col = _compute_mosaic_coefficient_of_variation(
                input_image_paths, nodata_val, debug_logs
            )  # This is the approach from the paper to compute bock size
    else:
        num_row, num_col = loaded_num_row, loaded_num_col

    if debug_logs:
        print("Computing local block maps:")

    # Compute local blocks
    local_blocks_to_calculate = {
        k: v for k, v in input_image_path_pairs_masked.items() if k in only_input
    }
    local_blocks_to_load = {
        **{k: loaded_block_local_means[soft_matches[k]] for k in matched},
        **{k: loaded_block_local_means[k] for k in only_loaded},
    }

    if local_blocks_to_calculate:
        args = [
            (
                name,
                path,
                bounds_canvas_coords,
                num_row,
                num_col,
                num_bands,
                debug_logs,
                nodata_val,
                calculation_dtype,
                tile_thread_on,
                tile_thread_workers
            )
            for name, path in local_blocks_to_calculate.items()
        ]

        if image_threads_on:
            with _get_executor(image_backend, image_thread_workers) as executor:
                futures = [
                    executor.submit(_calculate_block_process_image, *arg)
                    for arg in args
                ]
                results = [f.result() for f in futures]
        else:
            results = [_calculate_block_process_image(*arg) for arg in args]

        block_local_means = {name: mean for name, mean in results}

        overlap = set(block_local_means) & set(local_blocks_to_load)
        if overlap:
            raise ValueError(
                f"Duplicate keys when merging loaded and computed blocks: {overlap}"
            )

        block_local_means = {**block_local_means, **local_blocks_to_load}
    else:
        block_local_means = local_blocks_to_load

    # Compute reference block
    if debug_logs:
        print("Computing reference block map")
    if block_reference_mean is None:
        block_reference_mean = _compute_reference_blocks(
            block_local_means,
            calculation_dtype,
        )

    if save_block_maps:
        srs = gdal.Open(input_image_paths[0], gdal.GA_ReadOnly).GetProjection()
        _download_block_map(
            (
                np.nan_to_num(block_reference_mean, nan=nodata_val)
                if nodata_val is not None
                else block_reference_mean
            ),
            bounds_canvas_coords,
            reference_map_path,
            srs,
            calculation_dtype,
            nodata_val,
            num_col,
            num_row,
        )
        for name, block_local_mean in block_local_means.items():
            _download_block_map(
                (
                    np.nan_to_num(block_local_mean, nan=nodata_val)
                    if nodata_val is not None
                    else block_local_mean
                ),
                bounds_canvas_coords,
                local_map_path.replace("$", name),
                srs,
                calculation_dtype,
                nodata_val,
                num_col,
                num_row,
            )
            # _download_block_map(
            #     np.nan_to_num(block_local_count, nan=nodata_val),
            #     bounds_canvas_coords,
            #     os.path.join(output_image_folder, "BlockLocalCount", f"{input_image_name}_BlockLocalCount.tif"),
            #     projection,
            #     calculation_dtype,
            #     nodata_val,
            #     num_col,
            #     num_row,
            # )

    # block_local_mean = _smooth_array(block_local_mean, nodata_value=global_nodata_value)

    # Apply adjustments to images
    if debug_logs:
        print(f"Computing local correction, applying, and saving:")
    args = [
        (
            name,
            input_image_path_pairs[name],
            output_image_path_pairs[name],
            num_bands,
            block_reference_mean,
            block_local_means[name],
            bounds_canvas_coords,
            window_size,
            num_row,
            num_col,
            nodata_val,
            alpha,
            correction_method,
            calculation_dtype,
            _gdal_dtype_str_to_enum(output_dtype),
            debug_logs,
            tile_thread_on,
            tile_thread_workers,
            save_as_cog,
        )
        for name in input_image_path_pairs
    ]

    if image_threads_on:
        with _get_executor(image_backend, image_thread_workers) as executor:
            futures = [
                executor.submit(_apply_adjustment_process_image, *arg) for arg in args
            ]
            for future in as_completed(futures):
                future.result()
    else:
        for arg in args:
            _apply_adjustment_process_image(*arg)

    return output_image_paths


def _get_pre_computed_block_maps(
    load_block_maps: Tuple[Optional[str], Optional[List[str]]],
    calculation_dtype: str,
    debug_logs: bool,
) -> Tuple[
    dict[str, np.ndarray],
    Optional[np.ndarray],
    Optional[int],
    Optional[int],
    Optional[Tuple[float, float, float, float]],
]:
    """
    Load pre-computed block mean maps from files.

    Args:
        load_block_maps (Tuple[str, List[str]] | Tuple[str, None] | Tuple[None, List[str]]):
            - Tuple[str, List[str]]: Load both reference and local block maps.
            - Tuple[str, None]: Load only the reference block map.
            - Tuple[None, List[str]]: Load only the local block maps.
        calculation_dtype (str): Numpy dtype to use for reading.
        debug_logs (bool): To print debug statements or not.

    Returns:
        Tuple[
            dict[str, np.ndarray],             # block_local_means
            Optional[np.ndarray],              # block_reference_mean
            Optional[int],                     # num_row
            Optional[int],                     # num_col
            Optional[Tuple[float, float, float, float]]  # bounds_canvas_coords
        ]
    """
    ref_path, local_paths = load_block_maps

    shapes = set()
    extents = set()

    block_reference_mean = None

    # Load reference block map if provided
    if ref_path is not None:
        ds = gdal.Open(ref_path, gdal.GA_ReadOnly)
        arr = ds.ReadAsArray().astype(calculation_dtype)
        if arr.ndim == 2:  # If single band add band axis
            arr = arr[np.newaxis, ...]
        for i in range(arr.shape[0]):  # Replace nodata with NaN
            nd = ds.GetRasterBand(i + 1).GetNoDataValue()
            if nd is not None:
                arr[i][arr[i] == nd] = np.nan
        ref_data = np.transpose(arr, (1, 2, 0))  # H,W,B
        block_reference_mean = ref_data
        shapes.add(ref_data.shape)
        extents.add(_get_bounding_rectangle(ds))
        ds = None

    # Load local block maps if provided
    block_local_means = {}
    if local_paths is not None:
        for p in local_paths:
            name = os.path.splitext(os.path.basename(p))[0]
            ds = gdal.Open(p, gdal.GA_ReadOnly)
            arr = ds.ReadAsArray().astype(calculation_dtype)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            for i in range(arr.shape[0]):
                nd = ds.GetRasterBand(i + 1).GetNoDataValue()
                if nd is not None:
                    arr[i][arr[i] == nd] = np.nan
            data = np.transpose(arr, (1, 2, 0))  # H,W,B
            block_local_means[name] = data
            shapes.add(data.shape)
            extents.add(_get_bounding_rectangle(ds))
            ds = None

    if not shapes:
        raise ValueError("No block maps provided.")

    if len(shapes) != 1:
        raise ValueError(f"Inconsistent block map shapes: {shapes}")
    if len(extents) != 1:
        raise ValueError(f"Inconsistent block map extents: {extents}")

    num_row, num_col, _ = shapes.pop()
    bounds_canvas_coords = extents.pop()

    if debug_logs:
        print(
            f"Loaded block maps consistently have shape {(num_row, num_col)} and extent {bounds_canvas_coords}"
        )

    return (
        block_local_means,
        block_reference_mean,
        num_row,
        num_col,
        bounds_canvas_coords,
    )


def _get_bounding_rect_images_block_space(
    block_local_means: dict[str, np.ndarray],
) -> dict[str, tuple[int, int, int, int]]:
    """
    Compute block-space bounding rectangles for each image based on valid block values.

    Args:
        block_local_means (dict[str, np.ndarray]): Per-image block means
            with shape (num_row, num_col, num_bands).

    Returns:
        dict[str, tuple[int, int, int, int]]: Each entry maps image name to
            (min_row, min_col, max_row, max_col).
    """
    output = {}

    for name, arr in block_local_means.items():
        valid_mask = np.any(~np.isnan(arr), axis=2)
        rows, cols = np.where(valid_mask)

        if rows.size > 0 and cols.size > 0:
            min_row, max_row = rows.min(), rows.max() + 1
            min_col, max_col = cols.min(), cols.max() + 1
        else:
            min_row = max_row = min_col = max_col = 0

        output[name] = (min_row, min_col, max_row, max_col)

    return output


def _compute_reference_blocks(
    block_local_means: dict[str, np.ndarray],
    calculation_dtype: str,
) -> np.ndarray:
    """
    Computes reference block means across images by averaging non-NaN local block means.

    Args:
        block_local_means (dict[str, np.ndarray]): Per-image block mean arrays.
        calculation_dtype (str): Numpy dtype for output array.

    Returns:
        np.ndarray: Reference block map of shape (num_row, num_col, num_bands)
    """
    shape = next(iter(block_local_means.values())).shape
    stacked = np.stack(
        list(block_local_means.values()), axis=0
    )  # shape: (num_images, H, W, B)
    with np.errstate(invalid="ignore"):
        valid_mask = np.any(~np.isnan(stacked), axis=0)
        ref_block_mean = np.full(shape, np.nan, dtype=calculation_dtype)
        ref_block_mean[valid_mask] = np.nanmean(stacked[:, valid_mask], axis=0).astype(
            calculation_dtype
        )
    return ref_block_mean


def _apply_adjustment_process_image(
    name: str,
    img_path: str,
    out_path: str,
    num_bands: int,
    block_reference_mean: np.ndarray,
    block_local_mean: np.ndarray,
    bounds_canvas_coords: tuple,
    window_size,
    num_row: int,
    num_col: int,
    nodata_val: float,
    alpha: float,
    correction_method: Literal["gamma", "linear"],
    calculation_dtype: str,
    output_dtype,
    debug_logs: bool,
    tile_thread_on: bool,
    tile_thread_workers: int,
    save_as_cog: bool,
):
    """
    Apply local radiometric adjustment (linear or gamma) to a raster image using block-based reference and local mean surfaces. Builds parameter surfaces as rasters, warps them to the image grid, and creates a VRT with per-pixel expressions, then materializes the output as GTiff or COG.

    Args:
        name (str): Identifier for the image (basename, no extension).
        img_path (str): Path to the input raster image.
        out_path (str): Path where the adjusted raster will be written.
        num_bands (int): Number of spectral bands to process.
        block_reference_mean (np.ndarray): Block-level reference mean values per band.
        block_local_mean (np.ndarray): Block-level local mean values per band.
        bounds_canvas_coords (tuple): Geographic bounds of the image canvas (xmin, ymin, xmax, ymax).
        window_size (int | None): Output block size used for tiling.
        num_row (int): Number of block rows.
        num_col (int): Number of block columns.
        nodata_val (float): NoData value to assign to output bands.
        alpha (float): Scaling factor applied in gamma correction formula.
        correction_method (Literal["gamma", "linear"]): Radiometric correction method.
        calculation_dtype (str): Intermediate calculation data type (GDAL type string).
        output_dtype: GDAL output data type (enum or string).
        debug_logs (bool): If True, print debug information.
        tile_thread_on (bool): If True, enable multithreaded warp/translate operations.
        tile_thread_workers (int): Number of worker threads if `tile_thread_on=True`.
        save_as_cog (bool): If True, write output as Cloud-Optimized GeoTIFF (COG).

    Returns:
        None
    """
    if debug_logs:
        print(f"    {name}")

    # Open once to grab size & georef
    src = gdal.Open(img_path, gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Could not open {img_path}")
    w, h = src.RasterXSize, src.RasterYSize
    proj_wkt = src.GetProjectionRef()
    gt = src.GetGeoTransform()
    src = None

    xmin, ymin, xmax, ymax = bounds_canvas_coords
    tmpdir = tempfile.mkdtemp(prefix="spectralmatch_adjust_local_")

    # Helpers
    def _write_block_raster(arr2d: np.ndarray, path: str, dtype=gdal.GDT_Float32):
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(
            path, num_col, num_row, 1, dtype,
            options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6"]
        )
        gtx = (xmin, (xmax - xmin) / num_col, 0.0, ymax, 0.0, - (ymax - ymin) / num_row)
        ds.SetGeoTransform(gtx)
        if proj_wkt:
            srs = osr.SpatialReference(); srs.ImportFromWkt(proj_wkt)
            ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(arr2d.astype(np.float32))
        ds.GetRasterBand(1).SetNoDataValue(np.nan)
        ds = None

    def _warp_to_image_grid(src_path: str, dst_path: str):
        ds_out = gdal.Warp(
            dst_path, src_path,
            options=gdal.WarpOptions(
                format="VRT",
                dstSRS=proj_wkt or None,
                outputBounds=(gt[0], gt[3] + gt[5]*h, gt[0] + gt[1]*w, gt[3]),
                width=w, height=h,
                resampleAlg=gdal.GRIORA_Bilinear,
                multithread=tile_thread_on,
                warpOptions=(["SKIP_NOSOURCE=YES"] + ([f"NUM_THREADS={tile_thread_workers}"] if tile_thread_on else [])),
            )
        )
        if ds_out is None:
            raise RuntimeError("Warp failed building parameter surface")

    # Build full-resolution *single* parameter surfaces
    param_surfaces = []  # per-band path to VRT of G (linear) or GG (gamma)
    for b in range(num_bands):
        ref_blk = block_reference_mean[:, :, b].astype(np.float32, copy=False)
        loc_blk = block_local_mean[:, :, b].astype(np.float32, copy=False)

        if correction_method == "linear":
            # G = r / l ; invalid when l==0 or NaN
            with np.errstate(divide="ignore", invalid="ignore"):
                param_blk = ref_blk / loc_blk
        else:
            # GG = log(r) / log(l) ; invalid when r<=0 or l<=0
            with np.errstate(divide="ignore", invalid="ignore"):
                param_blk = np.log(ref_blk) / np.log(loc_blk)

        # keep invalids as NaN so they propagate
        param_blk[~np.isfinite(param_blk)] = np.nan

        # write block grid, then warp to image grid
        param_blk_tif = os.path.join(tmpdir, f"param_block_b{b + 1}.tif")
        _write_block_raster(param_blk, param_blk_tif)
        param_full_vrt = os.path.join(tmpdir, f"param_full_b{b + 1}.vrt")
        _warp_to_image_grid(param_blk_tif, param_full_vrt)
        param_surfaces.append(param_full_vrt)

    # --- VRT using the built-in expression pixel function --------------------
    out_vrt = os.path.join(tmpdir, f"{name}_local_adjust.vrt")
    calc_dtype_enum = _gdal_dtype_str_to_enum(calculation_dtype)
    calc_dtype_name = gdal.GetDataTypeName(calc_dtype_enum)

    # math expressions with guards (NaNs propagate automatically)
    if correction_method == "linear":
        if nodata_val is None:
            # Fastest case: no nodata check at all
            expr = "(v * P)"
        else:
            # Guard with mask for nodata handling
            expr = f"(m==0) ? {nodata_val} : (v * P)"
    else:  # "gamma"
        if nodata_val is None:
            expr = f"({alpha} * pow(v, P))"
        else:
            expr = f"(m==0) ? {nodata_val} : ({alpha} * pow(v, P))"

    bands_xml = []
    for b in range(1, num_bands + 1):
        bands_xml.append(f"""
    <VRTRasterBand dataType="{calc_dtype_name}" subClass="VRTDerivedRasterBand" band="{b}">
      <PixelFunctionType>expression</PixelFunctionType>
      <PixelFunctionArguments dialect="muparser" expression="{expr}"/>
      {f"<NoDataValue>{nodata_val}</NoDataValue>" if nodata_val is not None else ""}
      <!-- input band v -->
      <SimpleSource name="v">
        <SourceFilename relativeToVRT="0">{img_path}</SourceFilename>
        <SourceBand>{b}</SourceBand>
        <SourceTransferType>{calc_dtype_name}</SourceTransferType>
        <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
        <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
      </SimpleSource>
      <!-- parameter surface P: G (linear) or GG (gamma) -->
      <SimpleSource name="P">
        <SourceFilename relativeToVRT="0">{param_surfaces[b-1]}</SourceFilename>
        <SourceBand>1</SourceBand>
        <SourceTransferType>{calc_dtype_name}</SourceTransferType>
        <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
        <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
      </SimpleSource>
      <!-- input mask m (0 = invalid) -->
      <SimpleSource name="m">
        <SourceFilename relativeToVRT="0">{img_path}</SourceFilename>
        <SourceBand>mask</SourceBand>
        <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
        <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
      </SimpleSource>
    </VRTRasterBand>""")

    vrt_xml = f"""<VRTDataset rasterXSize="{w}" rasterYSize="{h}">
  <SRS>{proj_wkt or ''}</SRS>
  <GeoTransform>{gt[0]}, {gt[1]}, {gt[2]}, {gt[3]}, {gt[4]}, {gt[5]}</GeoTransform>
  {''.join(bands_xml)}
</VRTDataset>"""

    with open(out_vrt, "w", encoding="utf-8") as f:
        f.write(vrt_xml)

    # --- translate to output -------------------------------------------------
    if os.path.exists(out_path):
        try:
            gdal.Unlink(out_path)
        except Exception:
            pass

    window_size = _resolve_window_size(window_size, img_path, debug_logs)
    driver_name = "COG" if save_as_cog and gdal.GetDriverByName("COG") else "GTiff"
    co = (["COMPRESS=ZSTD", "LEVEL=9", f"BLOCKSIZE={window_size}", "OVERVIEWS=AUTO", "RESAMPLING=NEAREST"]
          if driver_name == "COG" else
          ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6", "BIGTIFF=IF_SAFER",
           f"BLOCKXSIZE={window_size}", f"BLOCKYSIZE={window_size}"])
    if tile_thread_on and driver_name == "GTiff":
        co.append(f"NUM_THREADS={tile_thread_workers}")

    ods = gdal.Translate(
        out_path, out_vrt,
        options=gdal.TranslateOptions(
            outputType=output_dtype,
            format=driver_name,
            creationOptions=co,
            noData=nodata_val if nodata_val is not None else None,
        )
    )
    if ods is None:
        raise RuntimeError("Failed to write adjusted image")
    ods = None


def _get_bounding_rectangle(
    images: List[str] | gdal.Dataset,
) -> Tuple[float, float, float, float]:
    """
    Return the combined extent (minx, miny, maxx, maxy) of rasters. Accepts a list of file paths, single path, or single GDAL dataset.

    Args:
        images (List[str] | gdal.Dataset): List of raster file paths, single path, or single GDAL dataset.

    Returns:
        Tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) of the combined extent.
    """

    if isinstance(images, gdal.Dataset):  # single dataset
        datasets = [images]
    elif isinstance(images, (list, tuple)):
        datasets = [gdal.Open(p) for p in images]
    else:  # single path
        datasets = [gdal.Open(images)]

    extents = []
    for ds in datasets:
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        corners = [
            (gt[0] + x * gt[1] + y * gt[2], gt[3] + x * gt[4] + y * gt[5])
            for x, y in [(0, 0), (w, 0), (0, h), (w, h)]
        ]
        xs, ys = zip(*corners)
        extents.append((min(xs), min(ys), max(xs), max(ys)))
        if not isinstance(images, gdal.Dataset):  # close only if we opened it
            ds = None

    return (
        min(e[0] for e in extents),
        min(e[1] for e in extents),
        max(e[2] for e in extents),
        max(e[3] for e in extents),
    )


def _compute_mosaic_coefficient_of_variation(
    image_paths: List[str],
    nodata_value: float,
    reference_std: float = 45.0,
    reference_mean: float = 125.0,
    base_block_size: Tuple[int, int] = (10, 10),
    band_index: int = 1,
    estimate_statistics: bool = True,
    debug_logs: bool = False,
) -> Tuple[int, int]:
    """
    Estimates block size for local adjustment using the coefficient of variation across input images.

    Args:
        image_paths (List[str]): List of input raster file paths.
        nodata_value (float): Value representing NoData in the input rasters.
        reference_std (float, optional): Reference standard deviation for comparison. Defaults to 45.0.
        reference_mean (float, optional): Reference mean for comparison. Defaults to 125.0.
        base_block_size (Tuple[int, int], optional): Base block size (rows, cols). Defaults to (10, 10).
        band_index (int, optional): Band index to use for statistics (1-based). Defaults to 1.
        estimate_statistics (bool, optional): If True, estimates statistics for each block. Defaults to True.
        debug_logs (bool, optional): If True, print logs.

    Returns:
        Tuple[int, int]: Estimated block size (rows, cols) adjusted based on coefficient of variation.
    """
    # Combine per-image stats: need mean_i, var_i, N_i
    total_N = 0.0
    total_mean = 0.0
    total_M2 = 0.0  # sum of squared deviations (for pooled variance)

    for p in image_paths:
        try:
            ds = gdal.Open(p, gdal.GA_ReadOnly)
            if ds is None:
                continue
            band = ds.GetRasterBand(band_index)
            # Ensure Nodata
            if nodata_value is not None:
                nd = band.GetNoDataValue()
                if nd is None:
                    band.SetNoDataValue(nodata_value)

            # Get stats

            mn, mx, mean_i, std_i = band.GetStatistics(1 if estimate_statistics else 0, 1)
            N_i = _get_valid_count(band, approx_ok=estimate_statistics)

            if N_i <= 0:
                ds = None
                continue

            var_i = float(std_i) ** 2

            # Pooled combine
            if total_N == 0:
                total_N = N_i
                total_mean = mean_i
                total_M2 = var_i * (N_i - 1)
            else:
                delta = mean_i - total_mean
                new_N = total_N + N_i
                total_mean += (N_i / new_N) * delta
                total_M2 += var_i * (N_i - 1) + (delta**2) * (total_N * N_i / new_N)
                total_N = new_N

            ds = None
        except Exception:
            continue

    if total_N <= 1:
        return base_block_size

    pooled_var = total_M2 / (total_N - 1)
    pooled_std = pooled_var**0.5
    if total_mean == 0:
        return base_block_size

    catar = pooled_std / total_mean  # coefficient of variation across the mosaic
    if debug_logs:
        print(f"Mosaic coefficient of variation (CAtar) = {catar:.4f}")

    caref = reference_std / reference_mean if reference_mean != 0 else 1.0
    r = catar / caref if caref != 0 else 1.0

    m, n = base_block_size
    return max(1, int(round(r * m))), max(1, int(round(r * n)))


def _calculate_block_process_image(
    name: str,
    image_path: str,
    bounds_canvas_coords: Tuple[float, float, float, float],
    num_row: int,
    num_col: int,
    num_bands: int,
    debug_logs: bool,
    nodata_value: float,
    calculation_dtype: str,
    tile_thread_on: bool,
    tile_thread_workers: int,
    ):
    """
    Compute area-weighted block means over a target grid using GDAL Warp.

    Args:
      name: Identifier carried through to the return tuple.
      image_path: Path to the source raster (VRT/GeoTIFF/etc.).
      bounds_canvas_coords: (x_min, y_min, x_max, y_max) in the source CRS (projection taken from `image_path`).
      num_row: Output grid height (rows).
      num_col: Output grid width (columns).
      num_bands: Number of bands to read from the warped raster.
      debug_logs: If True, emit progress and NaN counts to stdout.
      nodata_value: Source NoData value to treat as invalid (if present).
      calculation_dtype: Target NumPy dtype for the output array (e.g., "float32").
      tile_thread_on: If True, enable multithreaded warping.
      tile_thread_workers: Number of worker threads when `tile_thread_on` is True.

    Returns:
      Tuple[str, np.ndarray]: `(name, block_mean)` where `block_mean` has shape `(num_row, num_col, num_bands)` and dtype `calculation_dtype`. Cells with no valid input samples are NaN.
    """
    if debug_logs:
        print(f"    {name}")

    src_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Could not open {image_path}")
    proj_wkt = src_ds.GetProjectionRef()
    src_ds = None

    x_min, y_min, x_max, y_max = bounds_canvas_coords

    # Force float output so we can use NaN as dst nodata.
    mem_ds = gdal.Warp(
        "",
        image_path,
        options=gdal.WarpOptions(
            format="MEM",
            dstSRS=proj_wkt or None,
            outputBounds=(x_min, y_min, x_max, y_max),
            width=num_col,
            height=num_row,
            resampleAlg=gdal.GRIORA_Average,
            outputType=_gdal_dtype_str_to_enum(calculation_dtype),
            srcAlpha=True,
            srcNodata=nodata_value if nodata_value is not None else None,
            dstNodata=float("nan"),
            warpOptions=([
                "SKIP_NOSOURCE=YES",
                "UNIFIED_SRC_NODATA=YES",
                "INIT_DEST=NO_DATA",
            ] + ([f"NUM_THREADS={tile_thread_workers}"] if tile_thread_on else [])),
            multithread=tile_thread_on,
        )
    )
    if mem_ds is None:
        raise RuntimeError("Warp failed computing block means")

    block_mean = np.empty((num_row, num_col, num_bands), dtype=calculation_dtype)
    for b in range(1, num_bands + 1):
        arr = mem_ds.GetRasterBand(b).ReadAsArray()
        block_mean[:, :, b - 1] = arr.astype(calculation_dtype, copy=False)

    mem_ds = None
    return name, block_mean


def _download_block_map(
    block_map: np.ndarray,
    bounding_rect: Tuple[float, float, float, float],
    output_image_path: str,
    srs: str,
    dtype: str,
    nodata_value: float,
    width: int,
    height: int,
    write_bands: Tuple[int, ...] | None = None,
    delete_output: bool = True,
):
    """
    Writes a 3D block map to a raster file, creating or updating specified bands within a target window.

    Args:
        block_map (np.ndarray): Block data of shape (rows, cols, bands).
        bounding_rect (tuple): Spatial extent (minx, miny, maxx, maxy).
        output_image_path (str): Path to the output raster file.
        srs (str): SRS to save image with.
        dtype (str): Data type for output.
        nodata_value (float): NoData value to write.
        width (int): Full raster width.
        height (int): Full raster height.
        write_bands (tuple[int] | None): 0-based band indices to write; all if None.

    Output:
        Writes the `block_map` array to `output_image_path`, either creating a new raster or updating an existing one.
    """

    if block_map.ndim != 3:
        raise ValueError("block_map must be (rows, cols, bands)")
    h, w, nb = block_map.shape
    if (w, h) != (width, height):
        raise ValueError(f"block_map size {(w,h)} != target {(width,height)}")

    if write_bands is None:
        write_bands = tuple(range(nb))

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    if delete_output and os.path.exists(output_image_path):
        try: os.remove(output_image_path)
        except FileNotFoundError: pass

    # GeoTransform from bounds
    minx, miny, maxx, maxy = bounding_rect
    px_w = (maxx - minx) / width
    px_h = (maxy - miny) / height
    gt = (minx, px_w, 0.0, maxy, 0.0, -px_h)

    gdal_dtype = gdal.GetDataTypeByName(dtype) or gdal.GDT_Float32

    # Create (if needed)
    if not os.path.exists(output_image_path):
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(
            output_image_path, width, height, nb, gdal_dtype,
            options=["TILED=YES","COMPRESS=DEFLATE","PREDICTOR=2","ZLEVEL=6","BIGTIFF=IF_SAFER"]
        )
        ds.SetGeoTransform(gt)
        if srs:
            sref = osr.SpatialReference(); sref.ImportFromWkt(srs)
            ds.SetProjection(sref.ExportToWkt())
        for b in range(1, nb + 1):
            rb = ds.GetRasterBand(b)
            if nodata_value is not None:
                rb.SetNoDataValue(nodata_value); rb.Fill(nodata_value)
        ds = None

    # Write full bands
    ds = gdal.Open(output_image_path, gdal.GA_Update)
    for bi in write_bands:
        ds.GetRasterBand(bi + 1).WriteArray(block_map[:, :, bi])
    ds = None


def _compute_block_size(
    input_image_array_path: list,
    target_blocks_per_image: int | float,
    bounds_canvas_coords: tuple,
):
    """
    Calculates the number of rows and columns for dividing a bounding rectangle into target-sized blocks.

    Args:
        input_image_array_path (list): List of image paths to determine total image count.
        target_blocks_per_image (int | float): Desired number of blocks per image.
        bounds_canvas_coords (tuple): Bounding box covering all images (minx, miny, maxx, maxy).

    Returns:
        Tuple[int, int]: Number of rows (num_row) and columns (num_col) for the block grid.
    """

    num_images = len(input_image_array_path)

    # Total target blocks scaled by the number of images
    total_blocks = target_blocks_per_image * num_images

    x_min, y_min, x_max, y_max = bounds_canvas_coords
    bounding_width = x_max - x_min
    bounding_height = y_max - y_min

    # Aspect ratio of the bounding rectangle
    aspect_ratio = bounding_width / bounding_height

    # Start by assuming the number of columns (num_col)
    # We'll calculate num_col as the square root of total blocks scaled to the aspect ratio
    num_col = math.sqrt(total_blocks * aspect_ratio)
    num_col = max(1, round(num_col))  # Ensure at least one column

    # Calculate the number of rows (num_row) to match the number of blocks
    num_row = max(1, round(total_blocks / num_col))

    # Adjust for the closest fit to ensure num_row * num_col ≈ total_blocks
    while num_row * num_col < total_blocks:
        if bounding_width > bounding_height:
            num_col += 1
        else:
            num_row += 1

    while num_row * num_col > total_blocks:
        if bounding_width > bounding_height:
            num_col -= 1
        else:
            num_row -= 1

    return num_row, num_col


def _smooth_array(
    input_array: np.ndarray,
    nodata_value: Optional[float] = None,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """
    Applies Gaussian smoothing to an array while preserving NoData regions.

    Args:
        input_array (np.ndarray): 2D array to be smoothed.
        nodata_value (Optional[float], optional): Value representing NoData. Treated as NaN during smoothing. Defaults to None.
        scale_factor (float, optional): Sigma value for the Gaussian filter. Controls smoothing extent. Defaults to 1.0.

    Returns:
        np.ndarray: Smoothed array with NoData regions preserved or restored.
    """

    # Replace nodata_value with NaN for consistency
    if nodata_value is not None:
        input_array = np.where(input_array == nodata_value, np.nan, input_array)

    # Create a mask for valid (non-NaN) values
    valid_mask = ~np.isnan(input_array)

    # Replace NaN values with 0 to avoid affecting the smoothing
    array_with_nan_replaced = np.nan_to_num(input_array, nan=0.0)

    # Apply Gaussian smoothing
    smoothed = gaussian_filter(array_with_nan_replaced, sigma=scale_factor)

    # Normalize by the valid mask smoothed with the same kernel
    normalization_mask = gaussian_filter(valid_mask.astype(float), sigma=scale_factor)

    # Avoid division by zero in areas where the valid mask is 0
    smoothed_normalized = np.where(
        normalization_mask > 0, smoothed / normalization_mask, np.nan
    )

    # Reapply the nodata value (if specified) for output consistency
    if nodata_value is not None:
        smoothed_normalized = np.where(valid_mask, smoothed_normalized, nodata_value)

    return smoothed_normalized