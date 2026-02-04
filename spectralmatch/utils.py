import os
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile

from typing import Optional, Literal, Tuple, Dict
from concurrent.futures import as_completed
from osgeo import gdal, ogr, osr

from .handlers import _resolve_paths, _check_raster_requirements, _resolve_nodata_value
from .types_and_validation import Universal
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
    resolution: Literal["highest", "average", "lowest"] = "highest",
    window_size: Universal.WindowSize = None,
    debug_logs: Universal.DebugLogs = False,
    cache: Universal.Cache = None,
    image_threads: Universal.Threads = None,
    io_threads: Universal.Threads = None,
    tile_threads: Universal.Threads = None,
) -> None:
    """
    Aligns multiple rasters to a common resolution and grid using specified resampling.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Local.tif), ["/input/one.tif", "/input/two.tif"].
        resampling_method: "nearest" | "bilinear" | "cubic".
        tap: If True, snap output extent to target-aligned pixels (GDAL -tap behavior).
        resolution: "highest" (min px size), "average", or "lowest" (max px size).
        window_size: Tile size for output blocks; used for GTiff creation options.
        debug_logs: Verbose logging.
        cache: Cache for processing.
        image_threads: Python-level parallelism over images (e.g., ("process", 4)).
        io_threads: Sets GDAL_NUM_THREADS for internal GDAL multithreading (int or str).
        tile_threads: Sets GTiff/COG writer NUM_THREADS and Warp’s NUM_THREADS (int or str).

    Returns:
        List[str]: Paths to the locally adjusted output raster images.
    """
    if debug_logs:
        print("Start align rasters")

    Universal.validate(
        input_images=input_images,
        output_images=output_images,
        debug_logs=debug_logs,
        window_size=window_size,
        cache=cache,
        image_threads=image_threads,
        io_threads=io_threads,
        tile_threads=tile_threads,
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

    # Setup gdal
    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)

    # Setup parallel
    image_backend = "thread" # "process" or "thread"
    image_threads_on, image_thread_workers = _resolve_parallel_config(image_threads)
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
        )
        for i in range(len(input_image_paths))
    ]

    if image_threads:
        with _get_executor(image_backend, image_thread_workers) as executor:
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
    target_res: Tuple[float, float],
    resampling_method: Literal["nearest", "bilinear", "cubic"],
    tap: bool,
    window_size: int,
    tile_threads: Optional[int | str],
    debug_logs: bool,
    ) -> None:
    """
    Align a single raster to a target resolution and grid using GDAL Warp.

    Args:
        image_name (str): Identifier for the raster, used for logging.
        in_path (str): Path to the input raster file.
        out_path (str): Path where the aligned raster will be written.
        target_res (Tuple[float, float]): Target pixel resolution as ``(xres, yres)``.
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

    # Resolve metadata (extent, transform) via GDAL
    ds = gdal.Open(in_path, gdal.GA_ReadOnly)
    proj_wkt = ds.GetProjectionRef()
    ds = None

    xres, yres = float(target_res[0]), float(target_res[1])

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

    ods = gdal.Warp(
        out_path,
        in_path,
        options=gdal.WarpOptions(
            format="GTiff",
            xRes=xres,
            yRes=yres,
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
    strategy: str
    ) -> Tuple[float, float]:
    res = []
    for p in paths:
        ds = gdal.Open(p, gdal.GA_ReadOnly)
        gt = ds.GetGeoTransform()
        # Approximate pixel size (assume no rotation)
        res.append((abs(gt[1]), abs(gt[5])))
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
) -> str:
    """
    Merges multiple rasters into a single output using GDAL Warp (C++), aligning them to the union extent and a unified resolution. Supports parallelism via GDAL threading knobs.

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
    Returns:
        str: Path of the merged raster.

    """

    Universal.validate(
        input_images=input_images,
        debug_logs=debug_logs,
        cache=cache,
        io_threads=io_threads,
        tile_threads=tile_threads,
        output_dtype=output_dtype,
        window_size=window_size,
        custom_nodata_value=custom_nodata_value,
    )

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
    include_touched_pixels: bool = False,
    custom_nodata_value: Universal.CustomNodataValue = None,
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
        include_touched_pixels (bool, optional): If True, uses all touched pixels for cutline mask.
        custom_nodata_value (float | int | None, optional): Overrides detected NoData value. Defaults to None.

    Returns:
        list: Output image paths after masking.
    """

    if debug_logs:
        print("Start mask rasters")

    Universal.validate(
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
    )

    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    output_image_paths = _resolve_paths(
        "create",
        output_images,
        kwargs={"paths_or_bases": input_image_paths, "default_file_pattern": "$_Mask.tif"},
    )

    input_image_names = [
        os.path.splitext(os.path.basename(p))[0] for p in input_image_paths
    ]

    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)

    # Determine multiprocessing and worker count
    image_backend = "thread" # "thread" or "process"
    image_threads_on, image_thread_workers = _resolve_parallel_config(image_threads)
    tile_thread_on, tile_thread_workers = _resolve_parallel_config(tile_threads)

    mode, per_image_cutlines, original_vector_path, field_given = _prepare_cutline_sources(
        vector_mask, input_image_names, debug_logs
    )

    args = [
        (
            input_image_paths[i],
            output_image_paths[i],
            input_image_names[i],
            mode,
            (per_image_cutlines[input_image_names[i]] if field_given else original_vector_path),
            field_given,
            debug_logs,
            include_touched_pixels,
            custom_nodata_value,
            tile_thread_workers,
            tile_thread_on,
        )
        for i in range(len(input_image_paths))
    ]

    if image_threads_on:
        with _get_executor(image_backend, image_thread_workers) as executor:
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
    mode: str | None,
    cutline_path: str | None,
    field_given: bool,
    debug_logs: bool,
    include_touched_pixels: bool,
    custom_nodata_value: Universal.CustomNodataValue,
    tile_threads: int | str | None,
    tile_thread_on: bool,
) -> None:
    """
    Applies a GDAL Warp mask to a single image using cutline and nodata configuration.

    Args:
        input_image_path (str): Path to the input raster.
        output_image_path (str): Path to the output masked raster.
        image_name (str): Short name for logging/debugging.
        mode (str | None): Weather to "include" or "exclude".
        cutline_path (str | None): Path to the cutline.
        field_given (bool): If a filter field was provided.
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

    if cutline_path:
        invert = (mode == "exclude")
        warp_options.update({
            "cutlineDSName": cutline_path,
            "cropToCutline": not invert if field_given else not invert,  # same behavior
            "cutlineBlend": 0,
            "dstAlpha": False,
            "copyMetadata": True,
        })
        if include_touched_pixels:
            warp_options["warpOptions"].append("CUTLINE_ALL_TOUCHED=TRUE")
    # Else no cutline for this image

    gdal.Warp(destNameOrDestDS=output_image_path, srcDSOrSrcDSTab=input_image_path, options=gdal.WarpOptions(**warp_options))


def _prepare_cutline_sources(vector_mask, image_names, debug_logs=False):
    """
    Returns: (mode, per_image_path_or_None, original_vector_path, field_given)
      - If field is given: dict[image_name] -> '/vsimem/<name>.geojson' or None if no match
      - If no field: returns None dict, and you should pass the original vector as-is.
    """
    if not vector_mask:
        return None, None, None, False

    mode, vector_path, *maybe_field = vector_mask
    field_given = bool(maybe_field)
    if not field_given:
        return mode, None, vector_path, False

    field_name = maybe_field[0]
    vds = ogr.Open(vector_path)
    if vds is None:
        raise RuntimeError(f"Failed to open vector: {vector_path}")
    layer = vds.GetLayer(0)
    layer_srs = layer.GetSpatialRef()
    defn = layer.GetLayerDefn()
    fidx = defn.GetFieldIndex(field_name)
    if fidx < 0:
        raise ValueError(f"Field '{field_name}' not found in {vector_path}")

    wanted = set(image_names)
    out = {name: None for name in image_names}

    drv = ogr.GetDriverByName("GeoJSON")
    for feat in layer:
        key = str(feat.GetField(fidx))
        if key not in wanted:
            continue
        geom = feat.GetGeometryRef()
        if not geom or geom.IsEmpty():
            continue
        # write single-feature GeoJSON in /vsimem
        vs = f"/vsimem/{key}_cut.geojson"
        try:
            drv.DeleteDataSource(vs)
        except:
            pass
        ds = drv.CreateDataSource(vs)
        lyr = ds.CreateLayer("cut", srs=layer_srs, geom_type=geom.GetGeometryType())
        of = ogr.Feature(lyr.GetLayerDefn()); of.SetGeometry(geom.Clone())
        lyr.CreateFeature(of); of = None; ds = None
        out[key] = vs
        if debug_logs:
            print(f"cutline[{key}] -> {vs}")

    return mode, out, vector_path, True


def create_masked_vrts(
    input_image_path_pairs: Dict[str, str],
    *,
    vector_mask: Universal.VectorMask = None,
    out_dir: Optional[str] = None,
    debug_logs: bool = False,
) -> Dict[str, str]:
    """
    For each (name -> image_path), write:
      - mask_{name}.geojson  (cutline: include polys OR exclude complement)
      - vrt_{name}.vrt       (alpha = band1 nodata U cutline-outside)
    Returns: dict[name, vrt_path]
    """
    # temp dir for all masks+VRTs
    workdir = out_dir or tempfile.mkdtemp(prefix="spectralmatch_masks_")
    if debug_logs: print(f"Creating VRTs: {workdir}")

    out_vrts: Dict[str, str] = {}

    for image_name, image_path in input_image_path_pairs.items():
        if debug_logs:
            print(f"    {image_name}")

        src = gdal.Open(image_path, gdal.GA_ReadOnly)
        if src is None:
            raise RuntimeError(f"Could not open {image_path}")

        # raster geo + nodata
        nodata = src.GetRasterBand(1).GetNoDataValue()
        dst_wkt = src.GetProjectionRef() or ""
        dst_srs = osr.SpatialReference(); dst_srs.ImportFromWkt(dst_wkt)
        gt = src.GetGeoTransform()
        xmin, xmax = gt[0], gt[0] + gt[1] * src.RasterXSize
        ymax, ymin = gt[3], gt[3] + gt[5] * src.RasterYSize

        # build cutline only if requested
        cutline_ds = None
        if vector_mask:
            mode, vpath, *field = vector_mask
            field_name = field[0] if field else None

            vds = ogr.Open(vpath)
            if vds is None:
                raise RuntimeError(f"Could not open vector: {vpath}")
            lyr = vds.GetLayer(0)
            lyr.SetSpatialFilterRect(xmin, ymin, xmax, ymax)

            src_srs = lyr.GetSpatialRef()
            tx = osr.CoordinateTransformation(src_srs, dst_srs) if (src_srs and not src_srs.IsSame(dst_srs)) else None

            # extent polygon once
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(xmin, ymin); ring.AddPoint(xmin, ymax)
            ring.AddPoint(xmax, ymax); ring.AddPoint(xmax, ymin); ring.AddPoint(xmin, ymin)
            extent_poly = ogr.Geometry(ogr.wkbPolygon); extent_poly.AddGeometry(ring)

            # collect selected geoms in raster CRS
            selected = []
            for feat in lyr:
                if field_name:
                    val = feat.GetField(field_name)
                    if val is None or (image_name not in str(val)):
                        continue
                g = feat.GetGeometryRef()
                if not g:
                    continue
                gc = g.Clone()
                if tx: gc.Transform(tx)
                if gc.GetGeometryType() not in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
                    gc = gc.Buffer(0)
                gc = gc.Intersection(extent_poly)
                if gc and not gc.IsEmpty():
                    selected.append(gc)

            # decide what to write as the cutline geometry
            if mode == "include":
                geoms_to_write = selected
                suffix = "include"
            else:
                if selected:
                    mp = ogr.Geometry(ogr.wkbMultiPolygon)
                    for g in selected: mp.AddGeometry(g)
                    union_geom = mp.UnionCascaded()
                else:
                    union_geom = None
                complement = extent_poly if union_geom is None else extent_poly.Difference(union_geom)
                geoms_to_write = [complement]
                suffix = "exclude_complement"

            if geoms_to_write:
                cutline_path = os.path.join(workdir, f"mask_{image_name}.geojson")
                drv = ogr.GetDriverByName("GeoJSON")
                ods = drv.CreateDataSource(cutline_path)
                ol = ods.CreateLayer("cut", srs=dst_srs, geom_type=ogr.wkbPolygon)
                defn = ol.GetLayerDefn()
                for g in geoms_to_write:
                    if g and not g.IsEmpty():
                        of = ogr.Feature(defn); of.SetGeometry(g)
                        ol.CreateFeature(of); of = None
                ol = None; ods = None
                cutline_ds = cutline_path

            lyr = None; vds = None

        # build the VRT with alpha (nodata + cutline-outside)
        vrt_path = os.path.join(workdir, f"vrt_{image_name}.vrt")
        warp_opts = gdal.WarpOptions(
            format="VRT",
            dstSRS=dst_wkt or None,
            dstAlpha=True,
            dstNodata=None,
            cutlineDSName=cutline_ds,
            cropToCutline=False,
            resampleAlg=gdal.GRA_NearestNeighbour,
            multithread=True,
            warpOptions=[
                "SKIP_NOSOURCE=YES",
                "NUM_THREADS=ALL_CPUS",
                "UNIFIED_SRC_NODATA=YES",
            ],
        )
        out_ds = gdal.Warp(vrt_path, image_path, options=warp_opts)
        if out_ds is None:
            raise RuntimeError(f"Failed to build masked VRT for {image_name}")
        out_ds = None
        src = None

        out_vrts[image_name] = vrt_path
    return out_vrts


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