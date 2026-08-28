import os
import math
import geopandas as gpd
import pandas as pd
import numpy as np

from typing import Optional, Literal, Tuple, List
from concurrent.futures import as_completed
from osgeo import gdal, ogr, osr

from .handlers import (
    _resolve_paths,
    _check_raster_requirements,
    _resolve_nodata_value,
    _existing_outputs_are_reusable,
    _resolve_reusable_output_paths,
)
from .types_and_validation import Universal, Utils as UtilsValidation
from .utils_multiprocessing import _get_executor, _resolve_parallel_config


def merge_vectors(
    input_vectors: Universal.SearchFolderOrListFiles,
    merged_vector_path: str,
    method: Literal["intersection", "union", "keep"],
    debug_logs: bool = False,
    create_name_attribute: Optional[Tuple[str, str]] = None,
) -> None:
    """
    Merge multiple vector files using the specified geometric method.

    Args:
        input_vectors (str | List[str]): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.gpkg", "/input/folder" (assumes *.gpkg), ["/input/one.tif", "/input/two.tif"].
        merged_vector_path (str): Path to save merged output.
        method (Literal["intersection", "union", "keep"]): Merge strategy.
        debug_logs (bool): If True, print debug information.
        create_name_attribute (Optional[Tuple[str, str]]): Tuple of (field_name, separator) to add a combined name field.

    Returns:
        None
    """
    print("Start vector merge")

    os.makedirs(os.path.dirname(merged_vector_path), exist_ok=True)
    input_vector_paths = _resolve_paths(
        "search", input_vectors, kwargs={"default_file_pattern": "*.gpkg"}
    )

    geoms = []
    input_names = []

    for path in input_vector_paths:
        gdf = gpd.read_file(path)
        if create_name_attribute:
            name = os.path.splitext(os.path.basename(path))[0]
            input_names.append(name)
        geoms.append(gdf)

    combined_name_value = None
    if create_name_attribute:
        field_name, sep = create_name_attribute
        combined_name_value = sep.join(input_names)

    if method == "keep":
        merged_dfs = []
        field_name = create_name_attribute[0] if create_name_attribute else None
        for path in input_vector_paths:
            gdf = gpd.read_file(path)
            if field_name:
                name = os.path.splitext(os.path.basename(path))[0]
                gdf[field_name] = name
            merged_dfs.append(gdf)
        merged = gpd.GeoDataFrame(
            pd.concat(merged_dfs, ignore_index=True), crs=merged_dfs[0].crs
        )

    elif method == "union":
        merged = gpd.GeoDataFrame(pd.concat(geoms, ignore_index=True), crs=geoms[0].crs)
        if create_name_attribute:
            merged[field_name] = combined_name_value

    elif method == "intersection":
        merged = geoms[0]
        for gdf in geoms[1:]:
            shared_cols = set(merged.columns).intersection(gdf.columns) - {"geometry"}
            gdf = gdf.drop(columns=shared_cols)
            merged = gpd.overlay(merged, gdf, how="intersection", keep_geom_type=True)
        if create_name_attribute:
            merged[field_name] = combined_name_value

    else:
        raise ValueError(f"Unsupported merge method: {method}")

    merged.to_file(merged_vector_path)


def align_rasters(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    *,
    resampling_method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    tap: bool = False,
    resolution: Universal.Resolution = None,
    window_size: Universal.WindowSize = None,
    debug_logs: Universal.DebugLogs = False,
    cache: Universal.Cache = None,
    image_threads: Universal.Threads = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    concurrent_processing_backend: Universal.ConcurrentProcessingBackend = "process_pool",
    dask_scheduler: Universal.DaskScheduler = None,
    resume_from_outputs: Literal["no", "yes", "validate"] = "no",
) -> None:
    """
    Aligns multiple rasters to a common resolution and grid using specified resampling.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Local.tif), ["/input/one.tif", "/input/two.tif"].
        resampling_method: "nearest" | "bilinear" | "cubic".
        tap: If True, snap output extent to target-aligned pixels (GDAL -tap behavior).
        resolution: Shared pixel size strategy, positive CRS-unit pixel size, or None to preserve native resolution.
        window_size: Tile size for output blocks; used for GTiff creation options.
        debug_logs: Verbose logging.
        cache: Cache for processing.
        image_threads: Python-level parallelism over images (e.g., ("process", 4)).
        io_threads: Sets GDAL_NUM_THREADS for internal GDAL multithreading (int or str).
        tile_threads: Sets GTiff/COG writer NUM_THREADS and Warp’s NUM_THREADS (int or str).
        concurrent_processing_backend: Use a local process pool or an existing Dask cluster.
        dask_scheduler: Existing Dask scheduler as ("file", path) or ("address", address).

    Returns:
        List[str]: Paths to the locally adjusted output raster images.
    """
    if debug_logs:
        print("Start align rasters")

    Universal._validate(
        input_images=input_images,
        output_images=output_images,
        debug_logs=debug_logs,
        window_size=window_size,
        cache=cache,
        image_threads=image_threads,
        io_threads=io_threads,
        tile_threads=tile_threads,
        concurrent_processing_backend=concurrent_processing_backend,
        dask_scheduler=dask_scheduler,
    )
    UtilsValidation._validate_align_rasters(
        resampling_method=resampling_method,
        tap=tap,
        resolution=resolution,
    )

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={
            "paths_or_bases": input_image_paths,
            "default_file_pattern": "$_Align.tif",
        },
    )
    input_image_names = [
        os.path.splitext(os.path.basename(p))[0] for p in input_image_paths
    ]
    reusable_output_paths = _resolve_reusable_output_paths(
        output_image_paths,
        resume_mode=resume_from_outputs,
        debug_logs=debug_logs,
        step_name="align_rasters",
    )
    if len(reusable_output_paths) == len(output_image_paths):
        return output_image_paths

    # Setup gdal
    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)

    # Setup parallel
    image_backend = "thread" # "process" or "thread"
    image_threads_on, image_thread_workers = _resolve_parallel_config(
        image_threads, concurrent_processing_backend, dask_scheduler
    )
    tile_thread_on, tile_thread_workers = _resolve_parallel_config(tile_threads)


    if debug_logs:
        print(f"{len(input_image_paths)} rasters to align")

    # Check requirements
    _check_raster_requirements(
        input_image_paths,
        debug_logs,
        check_geotransform=True,
        check_crs=True,
        check_bands=True,
        check_nodata=True,
    )

    # Get target resolution
    target_res = compute_resolution(input_image_paths, resolution)

    if debug_logs:
        print(f"Target resolution: {target_res}")

    # Prepare per-image args
    window_size = _resolve_window_size(window_size, input_image_paths[0], debug_logs)
    args = [
        (
            input_image_names[i],
            input_image_paths[i],
            output_image_paths[i],
            target_res,
            resampling_method,
            tap,
            window_size,
            tile_thread_workers,
            debug_logs,
            resume_from_outputs,
        )
        for i in range(len(input_image_paths))
        if output_image_paths[i] not in reusable_output_paths
    ]

    if image_threads_on:
        with _get_executor(
            image_backend,
            image_thread_workers,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
        ) as executor:
            futures = [
                executor.submit(_align_process_image, *arg) for arg in args
            ]
            for future in as_completed(futures):
                future.result()
    else:
        for args in args:
            _align_process_image(*args)
    return output_image_paths


def _align_process_image(
    image_name: str,
    in_path: str,
    out_path: str,
    target_res: Tuple[float, float] | None,
    resampling_method: Literal["nearest", "bilinear", "cubic"],
    tap: bool,
    window_size: int,
    tile_threads: Optional[int | str],
    debug_logs: bool,
    resume_from_outputs: Literal["no", "yes", "validate"],
    ) -> None:
    """
    Align a single raster to a target resolution and grid using GDAL Warp.

    Args:
        image_name (str): Identifier for the raster, used for logging.
        in_path (str): Path to the input raster file.
        out_path (str): Path where the aligned raster will be written.
        target_res (Tuple[float, float] | None): Target pixel resolution, or None to preserve the source resolution.
        resampling_method (Literal["nearest", "bilinear", "cubic"]): Resampling algorithm.
        tap (bool): If True, snaps bounds to target-aligned pixels (GDAL -tap behavior).
        window_size (int): Tile size in pixels for output blocks (BLOCKXSIZE/BLOCKYSIZE).
        tile_threads (Optional[int | str]): Number of threads for GTiff/COG writer and Warp tile processing. If None, defaults to GDAL's internal behavior.
        debug_logs (bool): If True, print debug information during processing.

    Returns:
        None
    """
    if debug_logs:
        print(f"Aligning: {image_name}")
    if _existing_outputs_are_reusable(
        [out_path],
        resume_mode=resume_from_outputs,
        debug_logs=debug_logs,
        step_name="align_rasters",
    ):
        return
    window_size = max(16, int(math.ceil(window_size / 16)) * 16)

    # Resolve metadata (extent, transform) via GDAL
    ds = gdal.Open(in_path, gdal.GA_ReadOnly)
    proj_wkt = ds.GetProjectionRef()
    transform = ds.GetGeoTransform()
    ds = None

    # Writer creation options
    co = [
        "TILED=YES",
        "BIGTIFF=IF_SAFER",
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "ZLEVEL=6",
        f"BLOCKXSIZE={window_size}",
        f"BLOCKYSIZE={window_size}",
    ]
    if tile_threads is not None and str(tile_threads).strip():
        co.append(f"NUM_THREADS={tile_threads}")

    # Resampling map
    resamp = {
        "nearest": gdal.GRIORA_NearestNeighbour,
        "bilinear": gdal.GRIORA_Bilinear,
        "cubic": gdal.GRIORA_Cubic,
    }[resampling_method]

    # Perform alignment with GDAL Warp
    try:
        if os.path.exists(out_path):
            gdal.Unlink(out_path)
    except Exception:
        pass

    if target_res is None and not tap:
        ods = gdal.Translate(
            out_path,
            in_path,
            options=gdal.TranslateOptions(format="GTiff", creationOptions=co),
        )
    else:
        xres, yres = target_res or (
            math.hypot(transform[1], transform[4]),
            math.hypot(transform[2], transform[5]),
        )
        ods = gdal.Warp(
            out_path,
            in_path,
            options=gdal.WarpOptions(
                format="GTiff",
                xRes=float(xres),
                yRes=float(yres),
                dstSRS=proj_wkt or None,
                resampleAlg=resamp,
                targetAlignedPixels=tap,
                creationOptions=co,
                multithread=True,
                warpOptions=(["SKIP_NOSOURCE=YES"]),
            )
        )

    if ods is None:
        raise RuntimeError(f"gdal.Warp failed for {in_path}")
    ods = None


def compute_resolution(
    paths: list[str],
    strategy: Universal.Resolution,
    ) -> Tuple[float, float] | None:
    """Resolve a shared square/named resolution, or preserve native grids with None."""
    UtilsValidation._validate_align_rasters(resolution=strategy)
    if strategy is None:
        return None
    if isinstance(strategy, float):
        return strategy, strategy
    res = []
    for p in paths:
        ds = gdal.Open(p, gdal.GA_ReadOnly)
        gt = ds.GetGeoTransform()
        res.append((math.hypot(gt[1], gt[4]), math.hypot(gt[2], gt[5])))
        ds = None
    res_arr = np.asarray(res, dtype=float)
    if strategy == "highest":
        return float(res_arr[:, 0].min()), float(res_arr[:, 1].min())
    if strategy == "lowest":
        return float(res_arr[:, 0].max()), float(res_arr[:, 1].max())
    return float(res_arr[:, 0].mean()), float(res_arr[:, 1].mean())


def merge_rasters(
    input_images: Universal.SearchFolderOrListFiles,
    output_image_path: str,
    *,
    cache: Universal.Cache = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    debug_logs: Universal.DebugLogs = False,
    output_dtype: Universal.CustomOutputDtype = None,
    custom_nodata_value: Universal.CustomNodataValue = None,
    resolution: Literal["highest", "average", "lowest"] = "highest",
    window_size: Universal.WindowSize = None,
    build_overviews: bool = False,
    resume_from_outputs: Literal["no", "yes", "validate"] = "no",
) -> str:
    """
    Merges multiple rasters into a single output.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_image_path (str): Path to output mosaic.
        cache (int | Tuple[int, str] | None, optional): Controls GDAL cache size. Examples: 2048 (MB), (2, "GB"). Set None to use GDAL’s default. Applied via GDAL_CACHEMAX.        window_parallel_workers (Tuple[Literal["process"], Literal["cpu"] | int] | None = None): Parallelization strategy at the window level within each image. Same format as image_parallel_workers. Threads are not supported. Set to None to disable.
        io_threads (Literal["cpu"] | int | None): Parallelism for IO operations. "cpu" to get number of cores, int to assign number, and None to disable io level parallelism.
        tile_threads (Literal["cpu"] | int | None): "cpu" to get number of cores, int to assign number, and None to disable tile level parallelism.
        debug_logs (bool, optional): If True, prints progress. Defaults to False.
        output_dtype (str | None, optional): Data type for output rasters. Defaults to input image dtype.
        custom_nodata_value (float | int | None, optional): Overrides detected NoData value. Defaults to None.
        resolution ("highest" | "average" | "lowest", optional): Strategy for computing merge resolution.
        window_size (int | None): Tile size for processing tiles. Defaults to None.
        build_overviews (bool, optional): If True, computes overviews. Defaults to False.

    Returns:
        str: Path of the merged raster.

    """

    Universal._validate(
        input_images=input_images,
        debug_logs=debug_logs,
        cache=cache,
        io_threads=io_threads,
        tile_threads=tile_threads,
        output_dtype=output_dtype,
        window_size=window_size,
        custom_nodata_value=custom_nodata_value,
    )
    UtilsValidation._validate_merge_rasters(
        resolution=resolution,
    )
    if _existing_outputs_are_reusable(
        [output_image_path],
        resume_mode=resume_from_outputs,
        debug_logs=debug_logs,
        step_name="merge_rasters",
    ):
        return output_image_path

    # Setup parallel
    tile_thread_on, tile_thread_workers = _resolve_parallel_config(tile_threads)

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )

    # Dtype
    output_dtype = _gdal_dtype_str_to_enum(_resolve_gdal_dtype(output_dtype, input_image_paths[0]))

    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)

    if debug_logs:
        print(f"Building VRT from {len(input_image_paths)} rasters")

    vrt_opts = gdal.BuildVRTOptions(
        resolution=resolution,
        srcNodata=custom_nodata_value,
        VRTNodata=custom_nodata_value,
    )

    vrt_ds = gdal.BuildVRT("", input_image_paths, options=vrt_opts)

    creation_options = [
        "TILED=YES",
        "BIGTIFF=YES",
        "COMPRESS=ZSTD",
    ]

    if window_size:
        creation_options += [
            f"BLOCKXSIZE={window_size}",
            f"BLOCKYSIZE={window_size}",
        ]

    if tile_thread_workers is not None and str(tile_thread_workers).strip():
        creation_options.append(f"NUM_THREADS={tile_thread_workers}")

    translate_opts = gdal.TranslateOptions(
        format="GTiff",
        outputType=output_dtype,
        noData=custom_nodata_value,
        creationOptions=creation_options,
    )

    gdal.Translate(
        destName=output_image_path,
        srcDS=vrt_ds,
        options=translate_opts,
    )

    vrt_ds = None

    if build_overviews: compute_overviews(
        input_images_paths=output_image_path,
        cache=cache,
        io_threads=io_threads,
        tile_threads=tile_threads,
        debug_logs=debug_logs,
        )
    return output_image_path


def mask_rasters(
    input_images: Universal.SearchFolderOrListFiles,
    output_images: Universal.CreateInFolderOrListFiles,
    vector_mask: Universal.VectorMask = None,
    window_size: Universal.WindowSize = None,
    debug_logs: Universal.DebugLogs = False,
    cache: Universal.Cache = None,
    image_threads: Universal.Threads = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    concurrent_processing_backend: Universal.ConcurrentProcessingBackend = "process_pool",
    dask_scheduler: Universal.DaskScheduler = None,
    include_touched_pixels: bool = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
    resume_from_outputs: Literal["no", "yes", "validate"] = "no",
    ) -> list:
    """
    Applies a vector-based mask to one or more rasters using GDAL Warp.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Local.tif), ["/input/one.tif", "/input/two.tif"].
        vector_mask (Universal.VectorMask, optional): Tuple ('include'|'exclude', vector_path, optional field name).
        window_size (int | None): Tile size for processing tiles. Defaults to None.
        debug_logs (bool, optional): If True, prints progress. Defaults to False.
        cache (int | Tuple[int, str] | None, optional): Controls GDAL cache size. Examples: 2048 (MB), (2, "GB"). Set None to use GDAL’s default. Applied via GDAL_CACHEMAX.        window_parallel_workers (Tuple[Literal["process"], Literal["cpu"] | int] | None = None): Parallelization strategy at the window level within each image. Same format as image_parallel_workers. Threads are not supported. Set to None to disable.
        image_threads (Literal["cpu"] | int | None): Parallelism for per-image operations. "cpu" to get number of cores, int to assign number, and None to disable image level parallelism.
        io_threads (Literal["cpu"] | int | None): Parallelism for IO operations. "cpu" to get number of cores, int to assign number, and None to disable io level parallelism.
        tile_threads (Literal["cpu"] | int | None): "cpu" to get number of cores, int to assign number, and None to disable tile level parallelism.
        concurrent_processing_backend: Use a local process pool or an existing Dask cluster.
        dask_scheduler: Existing Dask scheduler as ("file", path) or ("address", address).
        include_touched_pixels (bool, optional): If True, uses all touched pixels for cutline mask.
        custom_nodata_value (float | int | None, optional): Overrides detected NoData value. Defaults to None.

    Returns:
        list: Output image paths after masking.
    """

    if debug_logs:
        print("Start mask rasters")

    Universal._validate(
        input_images=input_images,
        output_images=output_images,
        debug_logs=debug_logs,
        vector_mask=vector_mask,
        window_size=window_size,
        image_threads=image_threads,
        io_threads=io_threads,
        tile_threads=tile_threads,
        custom_nodata_value=custom_nodata_value,
        cache=cache,
        concurrent_processing_backend=concurrent_processing_backend,
        dask_scheduler=dask_scheduler,
    )
    UtilsValidation._validate_mask_rasters(
        include_touched_pixels=include_touched_pixels,
    )

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={"paths_or_bases": input_image_paths, "default_file_pattern": "$_Mask.tif"},
    )
    reusable_output_paths = _resolve_reusable_output_paths(
        output_image_paths,
        resume_mode=resume_from_outputs,
        debug_logs=debug_logs,
        step_name="mask_rasters",
    )
    if len(reusable_output_paths) == len(output_image_paths):
        return output_image_paths

    input_image_names = [
        os.path.splitext(os.path.basename(p))[0] for p in input_image_paths
    ]

    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)

    # Determine multiprocessing and worker count
    image_backend = "thread" # "thread" or "process"
    image_threads_on, image_thread_workers = _resolve_parallel_config(
        image_threads, concurrent_processing_backend, dask_scheduler
    )
    tile_thread_on, tile_thread_workers = _resolve_parallel_config(tile_threads)

    args = [
        (
            input_image_paths[i],
            output_image_paths[i],
            input_image_names[i],
            vector_mask,
            debug_logs,
            include_touched_pixels,
            custom_nodata_value,
            tile_thread_workers,
            tile_thread_on,
            resume_from_outputs,
        )
        for i in range(len(input_image_paths))
        if output_image_paths[i] not in reusable_output_paths
    ]

    if image_threads_on:
        with _get_executor(
            image_backend,
            image_thread_workers,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
        ) as executor:
            futures = [executor.submit(_mask_raster_process_image, *arg) for arg in args]
            for future in as_completed(futures):
                future.result()
    else:
        for arg in args:
            _mask_raster_process_image(*arg)

    return output_image_paths


def _mask_raster_process_image(
    input_image_path: str,
    output_image_path: str,
    image_name: str,
    vector_mask: Universal.VectorMask,
    debug_logs: bool,
    include_touched_pixels: bool,
    custom_nodata_value: Universal.CustomNodataValue,
    tile_threads: int | str | None,
    tile_thread_on: bool,
    resume_from_outputs: Literal["no", "yes", "validate"],
) -> None:
    """
    Applies a GDAL Warp mask to a single image using cutline and nodata configuration.

    Args:
        input_image_path (str): Path to the input raster.
        output_image_path (str): Path to the output masked raster.
        image_name (str): Short name for logging/debugging.
        vector_mask: Cutline mode, source path, and optional basename field.
        debug_logs (bool): If True, prints processing information.
        include_touched_pixels (bool): If True, enables CUTLINE_ALL_TOUCHED for Warp.
        custom_nodata_value (Universal.CustomNodataValue): Nodata value for masked-out pixels.
        tile_threads (int | str | None): Number of threads for Warp block parallelism.
        tile_thread_on (bool): Whether tile-level multithreading is enabled.

    Returns:
        None
    """

    if debug_logs:
        print(f"Masking image: {image_name}")
    if _existing_outputs_are_reusable(
        [output_image_path],
        resume_mode=resume_from_outputs,
        debug_logs=debug_logs,
        step_name="mask_rasters",
    ):
        return

    warp_options = {
        "format": "GTiff",
        "dstNodata": custom_nodata_value if custom_nodata_value is not None else None,
        "warpOptions": [
            "SKIP_NOSOURCE=YES",
            "UNIFIED_SRC_NODATA=YES",
        ],
    }

    if tile_thread_on:
        warp_options["multithread"] = True
        warp_options["warpOptions"].append(f"NUM_THREADS={tile_threads}")

    mode, cutline_options = _resolve_cutline_options(vector_mask, image_name)
    if cutline_options:
        invert = (mode == "exclude")
        warp_options.update({
            **cutline_options,
            "cropToCutline": not invert,
            "cutlineBlend": 0,
            "dstAlpha": False,
            "copyMetadata": True,
        })
        if include_touched_pixels:
            warp_options["warpOptions"].append("CUTLINE_ALL_TOUCHED=TRUE")
    # Else no cutline for this image

    gdal.Warp(destNameOrDestDS=output_image_path, srcDSOrSrcDSTab=input_image_path, options=gdal.WarpOptions(**warp_options))


def _create_masked_vrt(
    image_name: str,
    image_path: str,
    *,
    vector_mask: Universal.VectorMask = None,
    nodata_value: Optional[float] = None,
    out_dir: str,
    debug_logs: bool = False,
) -> str:

    workdir = out_dir
    if debug_logs: print(f"Creating VRTs: {workdir}")

    # Pre-parse mask configuration once
    mask_mode, cutline_options = _resolve_cutline_options(vector_mask, image_name)

    if debug_logs:
        print(f"    {image_name}")

    src = gdal.Open(image_path, gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Could not open {image_path}")
    dst_wkt = src.GetProjectionRef() or ""
    src = None

    vrt_path = os.path.join(workdir, f"vrt_{image_name}.vrt")
    warp_kwargs = {
        "format": "VRT",
        "dstSRS": dst_wkt or None,
        "dstAlpha": True,
        "dstNodata": nodata_value,
        "cropToCutline": False,
        "resampleAlg": gdal.GRA_NearestNeighbour,
        "multithread": True,
        "warpOptions": [
            "SKIP_NOSOURCE=YES",
            "NUM_THREADS=ALL_CPUS",
            "UNIFIED_SRC_NODATA=YES",
        ],
    }

    if cutline_options:
        warp_kwargs.update(cutline_options)
        if mask_mode == "exclude":
            warp_kwargs["warpOptions"].append("CUTLINE_INVERT=YES")

    out_ds = gdal.Warp(vrt_path, image_path, options=gdal.WarpOptions(**warp_kwargs))
    if out_ds is None:
        raise RuntimeError(f"Failed to build masked VRT for {image_name}")
    out_ds = None
    return vrt_path


def _resolve_cutline_options(vector_mask, image_name):
    """Return GDAL cutline options for one image without creating temporary data."""
    if not vector_mask:
        return None, {}
    mode, path, *field = vector_mask
    options = {"cutlineDSName": path}
    if field:
        vector = ogr.Open(path)
        if vector is None:
            raise RuntimeError(f"Could not open cutline dataset: {path}")
        layer = vector.GetLayer(0)
        if layer is None or layer.GetLayerDefn().GetFieldIndex(field[0]) < 0:
            raise ValueError(f"Cutline field '{field[0]}' was not found in {path}")
        safe_name = image_name.replace("'", "''")
        safe_field = field[0].replace('"', '""')
        where = f'"{safe_field}" LIKE \'%{safe_name}%\''
        layer.SetAttributeFilter(where)
        if layer.GetFeatureCount() == 0:
            return None, {}
        options["cutlineWhere"] = where
    return mode, options


def _set_gdal_cache(
    cache: float | None,
    debug_logs: bool,
    ):
    if cache is not None:
        gdal.SetCacheMax(int(cache * 1024 ** 3))
    if debug_logs: print(f"Cache: {gdal.GetCacheMax() / (1024 ** 3):.2f} GB")

def _set_gdal_workers(
    io_threads: int | str | None,
    debug_logs: bool,
    ):
    if io_threads is not None:
        if io_threads == "cpu": io_threads = "ALL_CPUS"
        else: io_threads = str(io_threads)
        gdal.SetConfigOption("GDAL_NUM_THREADS", io_threads)
    if debug_logs: print(f'GDAL num threads: {gdal.GetConfigOption("GDAL_NUM_THREADS", "Not set")}')


def _resolve_gdal_dtype(
    override_dtype: str | None = None,
    input_image_path: str | None = None,
    debug_logs: bool = False,
    ) -> str:
    """
    Resolve a valid GDAL data type string or image path for output.

    Args:
        override_dtype (str | None): Desired GDAL dtype name (e.g., "UInt16"). If None, falls back to the dtype of the input image.
        input_image_path (str): Path to the input raster for fallback.
        debug_logs (bool): If True, prints debug information.

    Returns:
        str: GDAL data type name (e.g., "Byte", "UInt16", "Float32").
    """
    if override_dtype is not None:
        if debug_logs:
            print(f"User-specified output data type: {override_dtype}")
        return override_dtype

    if input_image_path is None:
        raise ValueError("input_image_path must be provided if override_dtype is None")

    ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
    dtype_str = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
    if debug_logs:
        print(f"Image-derived output data type: {dtype_str}")
    ds = None
    return dtype_str



def _resolve_window_size(
    window_size: int | None,
    input_image_path: str,
    debug_logs: bool,
    ) -> int:
    """
    Resolve the output tile size (window size) for processing.

    Args:
        window_size (int | None): Desired tile size. If None, fall back to the block size of the input raster (or full image size if untiled).
        input_image_path (str): Path to the input raster for fallback.
        debug_logs: bool: If True, prints debug information.

    Returns:
        int: Tile size in pixels (square, width == height).
    """
    if window_size is not None:
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer or None.")
        if debug_logs: print("User-specified window size: ", window_size)
        return int(window_size)

    ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open {input_image_path}")
    try:
        band = ds.GetRasterBand(1)
        if band is None:
            raise RuntimeError(f"No bands found in {input_image_path}")

        block_x, block_y = band.GetBlockSize()
        if block_x > 1 and block_y > 1:
            returned_window_size = block_x if block_x == block_y else max(block_x, block_y)
            if debug_logs: print(f"    Found window size: {returned_window_size}")
            return returned_window_size
    finally:
        ds = None


_STR2ENUM = {
    "byte": gdal.GDT_Byte,
    "uint16": gdal.GDT_UInt16, "int16": gdal.GDT_Int16,
    "uint32": gdal.GDT_UInt32, "int32": gdal.GDT_Int32,
    "float32": gdal.GDT_Float32, "float64": gdal.GDT_Float64,
}


def _gdal_dtype_str_to_enum(s: str, default=gdal.GDT_Float32) -> int:
    if s is None:
        return default
    return _STR2ENUM.get(str(s).strip().lower(), default)


def _gdal_dtype_enum_to_name(dt: int) -> str:
    return gdal.GetDataTypeName(int(dt))


def _get_gdal_bounds(path: str) -> tuple[float, float, float, float]:
    """
    Compute spatial bounds of a raster.

    Args:
        path (str): Path to the raster file.

    Returns:
        tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) bounds in dataset CRS.
    """
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    w, h = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    corners = [gdal.ApplyGeoTransform(gt, x, y) for x, y in [(0,0), (w,0), (w,h), (0,h)]]
    xs, ys = zip(*corners)
    ds = None
    return min(xs), min(ys), max(xs), max(ys)


def _get_valid_count(
    band,
    approx_ok=True,
    force=True,
):
    """
    Get the valid pixel count of a raster band.

    Args:
        band (gdal.Band): Raster band to compute stats for.
        approx_ok (bool): Allow approximate statistics (fast, may be inaccurate).
        force (bool): If True, force computation if stats are not cached.

    Returns:
        valid_count
    """
    # valid pixel count via mask-mean
    flags = band.GetMaskFlags()
    if flags & gdal.GMF_ALL_VALID:
        n_valid = band.XSize * band.YSize
    else:
        _, _, m_mean, _ = band.GetMaskBand().GetStatistics(
        1 if approx_ok else 0,
            1 if force else 0
        )
        valid_frac = 0.0 if (m_mean is None) else (m_mean / 255.0)
        n_valid = int(round(valid_frac * band.XSize * band.YSize))

    return n_valid


def compute_overviews(
    input_images_paths: Universal.SearchFolderOrListFiles,
    *,
    output_image_paths: Universal.CreateInFolderOrListFiles | None = None,
    window_scales: tuple[int] | None = (2, 4, 8, 16, 32),
    cache: Universal.Cache = None,
    image_threads: Universal.Threads = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
    concurrent_processing_backend: Universal.ConcurrentProcessingBackend = "process_pool",
    dask_scheduler: Universal.DaskScheduler = None,
    debug_logs: bool = False,
):
    """
    Compute and attach GDAL overviews for one or more raster images.

    Args:
        input_images_paths (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_image_paths (str | List[str] | None): Defines output files as None to update input images or from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Global.tif), ["/input/one.tif", "/input/two.tif"].
        window_scales: Overview decimation factors (default: (2, 4, 8, 16, 32)).
        cache: GDAL cache size configuration.
        image_threads: Number of parallel workers for image-level processing.
        io_threads: GDAL IO worker configuration.
        tile_threads: GDAL internal threads for overview computation.
        concurrent_processing_backend: Use a local process pool or an existing Dask cluster.
        dask_scheduler: Existing Dask scheduler as ("file", path) or ("address", address).
        debug_logs: Enable verbose logging.

    Returns:
        List[str]: Paths of images that received overviews.
    """
    print("Start overviews computation")
    if debug_logs: print(f"Input images: {input_images_paths}")
    if debug_logs and output_image_paths: print(f"Output images: {output_image_paths}")
    if debug_logs: print(f"Window scales: {window_scales}")

    Universal._validate(
        input_images=input_images_paths,
        cache=cache,
        image_threads=image_threads,
        io_threads=io_threads,
        tile_threads=tile_threads,
        concurrent_processing_backend=concurrent_processing_backend,
        dask_scheduler=dask_scheduler,
    )

    def _copy_files_if_needed(
            src_paths: List[str],
            dst_paths: List[str],
        ) -> List[str]:
        """
        Copy src to dst.
        """
        out: List[str] = []

        for src, dst in zip(src_paths, dst_paths):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            ds = gdal.Translate(dst, src)
            if ds is None:
                raise RuntimeError(f"Failed copying {src} to {dst}")
            ds = None
            out.append(dst)

        return out


    # Paths
    input_paths = _resolve_paths(
        "search",
        input_images_paths,
        kwargs={"default_file_pattern": "*.tif"},
    )

    if output_image_paths is None:
        target_paths = input_paths
    else:
        target_paths = _resolve_paths(
            "create",
            output_image_paths,
            kwargs={
                "paths_or_bases": input_paths,
                "default_file_pattern": "$.tif",
            },
        )
        _copy_files_if_needed(input_paths, target_paths)

    # GDAL config
    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)

    image_backend = "thread"
    image_threads_on, image_workers = _resolve_parallel_config(
        image_threads, concurrent_processing_backend, dask_scheduler
    )
    tile_thread_on, tile_workers = _resolve_parallel_config(tile_threads)


    # Execute
    if image_threads_on:
        with _get_executor(
            image_backend,
            image_workers,
            concurrent_processing_backend=concurrent_processing_backend,
            dask_scheduler=dask_scheduler,
        ) as ex:
            futures = [
                ex.submit(_process_image_overview, p, window_scales, tile_thread_on, tile_workers, debug_logs)
                for p in target_paths
            ]
            for f in as_completed(futures):
                f.result()
    else:
        for p in target_paths:
            _process_image_overview(p, window_scales, tile_thread_on, tile_workers, debug_logs)

    return target_paths


def _process_image_overview(path, window_scales, tile_threads_on, tile_workers, debug_logs):
    """Build overviews for one image; kept top-level for distributed serialization."""
    dataset = gdal.Open(path, gdal.GA_Update)
    if dataset is None:
        raise RuntimeError(f"Cannot open {path}")
    options = [f"NUM_THREADS={tile_workers}"] if tile_threads_on else []
    dataset.BuildOverviews("AVERAGE", window_scales, options=options)
    dataset = None
    if debug_logs:
        print(f"Overviews built for: {path}")
