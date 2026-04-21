import os
import shutil
import tempfile
import math
import time

from datetime import datetime

from typing import Any, Literal

from .handlers import _resolve_paths
from .match.global_regression import global_regression
from .match.local_block_adjustment import local_block_adjustment
from .seamline.voronoi_center_seamline import voronoi_center_seamline
from .types_and_validation import Universal, Match, Pipeline as PipelineValidation, Utils as UtilsValidation, Seamline as SeamlineValidation
from .utils import align_rasters, mask_rasters, merge_rasters

AutoCache = Universal.Cache | Literal["auto"]
AutoThreads = Universal.Threads | Literal["auto"]


def pipeline(
    shared_input_images: Universal.SearchFolderOrListFiles,
    shared_output_image_path: str,
    *,
    shared_temp_dir: str | None = None,
    delete_temp_dir: bool = True,
    shared_debug_logs: Universal.DebugLogs = False,
    shared_cache: AutoCache = "auto",
    shared_custom_nodata_value: Universal.CustomNodataValue = None,
    shared_window_size: Universal.WindowSize = 1024,
    shared_image_threads: AutoThreads = "auto",
    shared_io_threads: AutoThreads = "auto",
    shared_tile_threads: AutoThreads = "auto",
    shared_calculation_dtype: Universal.CalculationDtype = "float32",
    shared_output_dtype: Universal.CustomOutputDtype = None,
    shared_save_as_cog: Universal.SaveAsCog = False,
    matching_global_method: Literal["global_regression"] | None = "global_regression",
    matching_local_method: Literal["local_block_adjustment"] | None = "local_block_adjustment",
    align_method: Literal["align_rasters"] | None = "align_rasters",
    seamline_method: Literal["voronoi_center_seamline"] | None = "voronoi_center_seamline",
    clip_method: Literal["mask_rasters"] | None = "mask_rasters",
    merge_method: Literal["merge_rasters"] | None = "merge_rasters",
    global_regression_output_images: Universal.CreateInFolderOrListFiles | None = None,
    global_regression_vector_mask: Universal.VectorMask = None,
    global_regression_estimate_stats: bool = True,
    global_regression_specify_model_images: tuple[Literal["exclude", "include"], list[str]] | None = None,
    global_regression_custom_mean_factor: float = 1.0,
    global_regression_custom_std_factor: float = 1.0,
    global_regression_save_adjustments: str | None = None,
    global_regression_load_adjustments: str | None = None,
    global_regression_pif_method: Literal["entire", "conjugate"] = "entire",
    global_regression_pif_red_band_index: int | None = None,
    global_regression_pif_nir_band_index: int | None = None,
    global_regression_pif_vegetation_threshold: float = 0.2,
    global_regression_pif_inz_threshold: float = 0.25,
    global_regression_pif_region_radius: int = 5,
    global_regression_pif_max_samples: int = 100000,
    global_regression_pif_min_samples: int = 32,
    global_regression_pif_feature_method: Literal["orb"] = "orb",
    global_regression_build_overviews: bool = False,
    local_block_adjustment_output_images: Universal.CreateInFolderOrListFiles | None = None,
    local_block_adjustment_vector_mask: Universal.VectorMask = None,
    local_block_adjustment_number_of_blocks: int | tuple[int, int] | Literal["coefficient_of_variation"] = 100,
    local_block_adjustment_alpha: float = 1.0,
    local_block_adjustment_correction_method: Literal["gamma", "linear", "offset"] = "offset",
    local_block_adjustment_save_block_maps: tuple[str, str] | None = None,
    local_block_adjustment_load_block_maps: tuple[str | None, list[str] | None] | None = None,
    local_block_adjustment_override_bounds_canvas_coords: tuple[float, float, float, float] | None = None,
    local_block_adjustment_build_overviews: bool = False,
    align_rasters_output_images: Universal.CreateInFolderOrListFiles | None = None,
    align_rasters_resampling_method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    align_rasters_tap: bool = True,
    align_rasters_resolution: Literal["highest", "average", "lowest"] = "highest",
    voronoi_center_seamline_output_mask: str | None = None,
    voronoi_center_seamline_aoi_path: str | None = None,
    voronoi_center_seamline_vector_mask: tuple[str, str] | None = None,
    voronoi_center_seamline_image_field_name: str = "image",
    voronoi_center_seamline_min_point_spacing: float = 10,
    voronoi_center_seamline_min_cut_length: float = 0,
    voronoi_center_seamline_debug_vectors_path: str | None = None,
    mask_rasters_output_images: Universal.CreateInFolderOrListFiles | None = None,
    mask_rasters_vector_mask: Universal.VectorMask = None,
    mask_rasters_include_touched_pixels: bool = False,
    merge_rasters_resolution: Literal["highest", "average", "lowest"] = "highest",
    merge_rasters_build_overviews: bool = True,
) -> dict[str, Any]:
    """
    Run the standard spectral matching mosaic workflow as a single pipeline.

    The current pipeline order is: global matching -> local matching -> align -> seamline -> clip -> merge

    Each stage can be disabled by setting its method parameter to ``None``. Intermediate outputs default to a temporary directory that is created automatically unless ``shared_temp_dir`` is provided. Set``delete_temp_dir=True`` to remove the temp directory after processing.
    """
    temp_dir = shared_temp_dir or tempfile.mkdtemp(prefix="spectralmatch_pipeline_")
    os.makedirs(temp_dir, exist_ok=True)
    input_image_paths = _resolve_paths(
        "search", shared_input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    shared_cache, shared_image_threads, shared_io_threads, shared_tile_threads = (
        _resolve_auto_shared_settings(
            shared_input_images=shared_input_images,
            shared_cache=shared_cache,
            shared_image_threads=shared_image_threads,
            shared_io_threads=shared_io_threads,
            shared_tile_threads=shared_tile_threads,
            shared_debug_logs=shared_debug_logs,
        )
    )
    PipelineValidation.validate_shared_pipeline(
        shared_output_image_path=shared_output_image_path,
        shared_temp_dir=shared_temp_dir,
        delete_temp_dir=delete_temp_dir,
    )
    for method_name, method_value, allowed_values in [
        ("matching_global_method", matching_global_method, {None, "global_regression"}),
        ("matching_local_method", matching_local_method, {None, "local_block_adjustment"}),
        ("align_method", align_method, {None, "align_rasters"}),
        ("seamline_method", seamline_method, {None, "voronoi_center_seamline"}),
        ("clip_method", clip_method, {None, "mask_rasters"}),
        ("merge_method", merge_method, {None, "merge_rasters"}),
    ]:
        PipelineValidation.validate_method_choice(
            method_name=method_name,
            method_value=method_value,
            allowed_values=allowed_values,
        )
    Universal.validate(
        input_images=shared_input_images,
        debug_logs=shared_debug_logs,
        window_size=shared_window_size,
        custom_nodata_value=shared_custom_nodata_value,
        calculation_dtype=shared_calculation_dtype,
        output_dtype=shared_output_dtype,
        cache=shared_cache,
        image_threads=shared_image_threads,
        io_threads=shared_io_threads,
        tile_threads=shared_tile_threads,
        save_as_cog=shared_save_as_cog,
    )
    if matching_global_method == "global_regression":
        Universal.validate(
            input_images=shared_input_images,
            output_images=global_regression_output_images or os.path.join(temp_dir, "global"),
            save_as_cog=shared_save_as_cog,
            debug_logs=shared_debug_logs,
            vector_mask=global_regression_vector_mask,
            window_size=shared_window_size,
            custom_nodata_value=shared_custom_nodata_value,
            calculation_dtype=shared_calculation_dtype,
            output_dtype=shared_output_dtype,
            cache=shared_cache,
            image_threads=shared_image_threads,
            io_threads=shared_io_threads,
            tile_threads=shared_tile_threads,
            estimate_stats=global_regression_estimate_stats,
        )
        Match.validate_match(specify_model_images=global_regression_specify_model_images)
        Match.validate_global_regression(
            custom_mean_factor=global_regression_custom_mean_factor,
            custom_std_factor=global_regression_custom_std_factor,
            save_adjustments=global_regression_save_adjustments,
            load_adjustments=global_regression_load_adjustments,
            pif_method=global_regression_pif_method,
            pif_feature_method=global_regression_pif_feature_method,
        )
    if matching_local_method == "local_block_adjustment":
        Universal.validate(
            input_images=shared_input_images,
            output_images=local_block_adjustment_output_images or os.path.join(temp_dir, "local"),
            save_as_cog=shared_save_as_cog,
            debug_logs=shared_debug_logs,
            vector_mask=local_block_adjustment_vector_mask,
            window_size=shared_window_size,
            custom_nodata_value=shared_custom_nodata_value,
            calculation_dtype=shared_calculation_dtype,
            output_dtype=shared_output_dtype,
            cache=shared_cache,
            image_threads=shared_image_threads,
            io_threads=shared_io_threads,
            tile_threads=shared_tile_threads,
        )
        Match.validate_local_block_adjustment(
            number_of_blocks=local_block_adjustment_number_of_blocks,
            alpha=local_block_adjustment_alpha,
            correction_method=local_block_adjustment_correction_method,
            save_block_maps=local_block_adjustment_save_block_maps,
            load_block_maps=local_block_adjustment_load_block_maps,
            override_bounds_canvas_coords=local_block_adjustment_override_bounds_canvas_coords,
        )
    if align_method == "align_rasters":
        Universal.validate(
            input_images=shared_input_images,
            output_images=align_rasters_output_images or os.path.join(temp_dir, "aligned"),
            debug_logs=shared_debug_logs,
            window_size=shared_window_size,
            cache=shared_cache,
            image_threads=shared_image_threads,
            io_threads=shared_io_threads,
            tile_threads=shared_tile_threads,
        )
        UtilsValidation.validate_align_rasters(
            resampling_method=align_rasters_resampling_method,
            tap=align_rasters_tap,
            resolution=align_rasters_resolution,
        )
    if seamline_method == "voronoi_center_seamline":
        Universal.validate(input_images=shared_input_images)
        SeamlineValidation.validate_voronoi_center_seamline(
            output_mask=voronoi_center_seamline_output_mask or os.path.join(temp_dir, "seamline", "ImageMasks.gpkg"),
            aoi_path=voronoi_center_seamline_aoi_path,
            vector_mask=voronoi_center_seamline_vector_mask,
            image_field_name=voronoi_center_seamline_image_field_name,
            min_point_spacing=voronoi_center_seamline_min_point_spacing,
            min_cut_length=voronoi_center_seamline_min_cut_length,
            debug_vectors_path=voronoi_center_seamline_debug_vectors_path,
        )
    if clip_method == "mask_rasters":
        clip_vector_mask = mask_rasters_vector_mask
        if clip_vector_mask is None and seamline_method == "voronoi_center_seamline":
            clip_vector_mask = (
                "include",
                voronoi_center_seamline_output_mask or os.path.join(temp_dir, "seamline", "ImageMasks.gpkg"),
                voronoi_center_seamline_image_field_name,
            )
        if clip_vector_mask is None:
            raise ValueError(
                "mask_rasters requires a vector mask. Set mask_rasters_vector_mask or enable the seamline stage."
            )
        Universal.validate(
            input_images=shared_input_images,
            output_images=mask_rasters_output_images or os.path.join(temp_dir, "clip"),
            debug_logs=shared_debug_logs,
            vector_mask=clip_vector_mask,
            window_size=shared_window_size,
            custom_nodata_value=shared_custom_nodata_value,
            cache=shared_cache,
            image_threads=shared_image_threads,
            io_threads=shared_io_threads,
            tile_threads=shared_tile_threads,
        )
        UtilsValidation.validate_mask_rasters(
            include_touched_pixels=mask_rasters_include_touched_pixels,
        )
    if merge_method == "merge_rasters":
        Universal.validate(
            input_images=shared_input_images,
            debug_logs=shared_debug_logs,
            cache=shared_cache,
            io_threads=shared_io_threads,
            tile_threads=shared_tile_threads,
            output_dtype=shared_output_dtype,
            window_size=shared_window_size,
            custom_nodata_value=shared_custom_nodata_value,
        )
        UtilsValidation.validate_merge_rasters(
            resolution=merge_rasters_resolution,
        )

    start_dt = datetime.now()
    start_perf = time.perf_counter()
    print(f"Pipeline start: {start_dt.isoformat(timespec='seconds')}")
    print(f"Pipeline temp dir: {temp_dir}")
    print(f"Number of input images: {len(input_image_paths)}")

    current_images = shared_input_images
    seamline_mask_path = None

    results: dict[str, Any] = {
        "temp_dir": temp_dir,
        "input_images": shared_input_images,
        "resolved_shared_cache": shared_cache,
        "resolved_shared_image_threads": shared_image_threads,
        "resolved_shared_io_threads": shared_io_threads,
        "resolved_shared_tile_threads": shared_tile_threads,
        "num_input_images": len(input_image_paths),
        "start_time": start_dt.isoformat(timespec="seconds"),
    }
    try:
        if matching_global_method == "global_regression":
            global_output_images = global_regression_output_images or os.path.join(
                temp_dir, "global"
            )
            current_images = global_regression(
                input_images=current_images,
                output_images=global_output_images,
                calculation_dtype=shared_calculation_dtype,
                output_dtype=shared_output_dtype,
                vector_mask=global_regression_vector_mask,
                debug_logs=shared_debug_logs,
                custom_nodata_value=shared_custom_nodata_value,
                cache=shared_cache,
                image_threads=shared_image_threads,
                io_threads=shared_io_threads,
                tile_threads=shared_tile_threads,
                estimate_stats=global_regression_estimate_stats,
                window_size=shared_window_size,
                save_as_cog=shared_save_as_cog,
                specify_model_images=global_regression_specify_model_images,
                custom_mean_factor=global_regression_custom_mean_factor,
                custom_std_factor=global_regression_custom_std_factor,
                save_adjustments=global_regression_save_adjustments,
                load_adjustments=global_regression_load_adjustments,
                pif_method=global_regression_pif_method,
                pif_red_band_index=global_regression_pif_red_band_index,
                pif_nir_band_index=global_regression_pif_nir_band_index,
                pif_vegetation_threshold=global_regression_pif_vegetation_threshold,
                pif_inz_threshold=global_regression_pif_inz_threshold,
                pif_region_radius=global_regression_pif_region_radius,
                pif_max_samples=global_regression_pif_max_samples,
                pif_min_samples=global_regression_pif_min_samples,
                pif_feature_method=global_regression_pif_feature_method,
                build_overviews=global_regression_build_overviews,
            )
            results["global_regression"] = current_images
        elif matching_global_method is None:
            results["global_regression"] = None
        else:
            raise ValueError(f"Unsupported matching_global_method: {matching_global_method}")

        if matching_local_method == "local_block_adjustment":
            local_output_images = local_block_adjustment_output_images or os.path.join(
                temp_dir, "local"
            )
            current_images = local_block_adjustment(
                input_images=current_images,
                output_images=local_output_images,
                calculation_dtype=shared_calculation_dtype,
                output_dtype=shared_output_dtype,
                vector_mask=local_block_adjustment_vector_mask,
                debug_logs=shared_debug_logs,
                custom_nodata_value=shared_custom_nodata_value,
                cache=shared_cache,
                image_threads=shared_image_threads,
                io_threads=shared_io_threads,
                tile_threads=shared_tile_threads,
                window_size=shared_window_size,
                save_as_cog=shared_save_as_cog,
                number_of_blocks=local_block_adjustment_number_of_blocks,
                alpha=local_block_adjustment_alpha,
                correction_method=local_block_adjustment_correction_method,
                save_block_maps=local_block_adjustment_save_block_maps,
                load_block_maps=local_block_adjustment_load_block_maps,
                override_bounds_canvas_coords=local_block_adjustment_override_bounds_canvas_coords,
                build_overviews=local_block_adjustment_build_overviews,
            )
            results["local_block_adjustment"] = current_images
        elif matching_local_method is None:
            results["local_block_adjustment"] = None
        else:
            raise ValueError(f"Unsupported matching_local_method: {matching_local_method}")

        if align_method == "align_rasters":
            align_output_images = align_rasters_output_images or os.path.join(
                temp_dir, "aligned"
            )
            current_images = align_rasters(
                input_images=current_images,
                output_images=align_output_images,
                resampling_method=align_rasters_resampling_method,
                tap=align_rasters_tap,
                resolution=align_rasters_resolution,
                window_size=shared_window_size,
                debug_logs=shared_debug_logs,
                cache=shared_cache,
                image_threads=shared_image_threads,
                io_threads=shared_io_threads,
                tile_threads=shared_tile_threads,
            )
            results["align_rasters"] = current_images
        elif align_method is None:
            results["align_rasters"] = None
        else:
            raise ValueError(f"Unsupported align_method: {align_method}")

        if seamline_method == "voronoi_center_seamline":
            seamline_mask_path = voronoi_center_seamline_output_mask or os.path.join(
                temp_dir, "seamline", "ImageMasks.gpkg"
            )
            voronoi_center_seamline(
                input_images=current_images,
                output_mask=seamline_mask_path,
                aoi_path=voronoi_center_seamline_aoi_path,
                vector_mask=voronoi_center_seamline_vector_mask,
                image_field_name=voronoi_center_seamline_image_field_name,
                min_point_spacing=voronoi_center_seamline_min_point_spacing,
                min_cut_length=voronoi_center_seamline_min_cut_length,
                debug_logs=shared_debug_logs,
                debug_vectors_path=voronoi_center_seamline_debug_vectors_path,
            )
            results["voronoi_center_seamline"] = seamline_mask_path
        elif seamline_method is None:
            results["voronoi_center_seamline"] = None
        else:
            raise ValueError(f"Unsupported seamline_method: {seamline_method}")

        if clip_method == "mask_rasters":
            mask_output_images = mask_rasters_output_images or os.path.join(temp_dir, "clip")
            clip_vector_mask = mask_rasters_vector_mask
            if clip_vector_mask is None and seamline_mask_path is not None:
                clip_vector_mask = (
                    "include",
                    seamline_mask_path,
                    voronoi_center_seamline_image_field_name,
                )
            if clip_vector_mask is None:
                raise ValueError(
                    "mask_rasters requires a vector mask. Set mask_rasters_vector_mask "
                    "or enable the seamline stage."
                )

            current_images = mask_rasters(
                input_images=current_images,
                output_images=mask_output_images,
                vector_mask=clip_vector_mask,
                window_size=shared_window_size,
                debug_logs=shared_debug_logs,
                cache=shared_cache,
                image_threads=shared_image_threads,
                io_threads=shared_io_threads,
                tile_threads=shared_tile_threads,
                include_touched_pixels=mask_rasters_include_touched_pixels,
                custom_nodata_value=shared_custom_nodata_value,
            )
            results["mask_rasters"] = current_images
        elif clip_method is None:
            results["mask_rasters"] = None
        else:
            raise ValueError(f"Unsupported clip_method: {clip_method}")

        if merge_method == "merge_rasters":
            merged_output = merge_rasters(
                input_images=current_images,
                output_image_path=shared_output_image_path,
                cache=shared_cache,
                io_threads=shared_io_threads,
                tile_threads=shared_tile_threads,
                debug_logs=shared_debug_logs,
                output_dtype=shared_output_dtype,
                custom_nodata_value=shared_custom_nodata_value,
                resolution=merge_rasters_resolution,
                window_size=shared_window_size,
                build_overviews=merge_rasters_build_overviews,
            )
            results["merge_rasters"] = merged_output
            results["output"] = merged_output
        elif merge_method is None:
            results["merge_rasters"] = None
            results["output"] = current_images
        else:
            raise ValueError(f"Unsupported merge_method: {merge_method}")

        end_dt = datetime.now()
        duration_seconds = round(time.perf_counter() - start_perf, 2)

        return {
            "output": results["output"],
            "temp_dir": temp_dir,
            "num_input_images": len(input_image_paths),
            "start_time": start_dt.isoformat(timespec="seconds"),
            "end_time": end_dt.isoformat(timespec="seconds"),
            "duration_seconds": duration_seconds,
            "resolved_shared_cache": shared_cache,
            "resolved_shared_image_threads": shared_image_threads,
            "resolved_shared_io_threads": shared_io_threads,
            "resolved_shared_tile_threads": shared_tile_threads,
        }
    finally:
        if delete_temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def _resolve_auto_shared_settings(
    *,
    shared_input_images: Universal.SearchFolderOrListFiles,
    shared_cache: AutoCache,
    shared_image_threads: AutoThreads,
    shared_io_threads: AutoThreads,
    shared_tile_threads: AutoThreads,
    shared_debug_logs: bool,
) -> tuple[Universal.Cache, Universal.Threads, Universal.Threads, Universal.Threads]:
    input_image_paths = _resolve_paths(
        "search", shared_input_images, kwargs={"default_file_pattern": "*.tif"}
    )
    image_count = max(1, len(input_image_paths))
    cpu_count = max(1, os.cpu_count() or 1)
    auto_cache = _resolve_auto_cache_gb()
    auto_image_threads = min(image_count, max(1, math.ceil(cpu_count * 0.90)))
    auto_worker_threads = max(1, min(int(cpu_count / auto_image_threads), 4))

    resolved_cache = auto_cache if shared_cache == "auto" else shared_cache
    resolved_image_threads = (
        auto_image_threads if shared_image_threads == "auto" else shared_image_threads
    )
    resolved_io_threads = (
        auto_worker_threads if shared_io_threads == "auto" else shared_io_threads
    )
    resolved_tile_threads = (
        auto_worker_threads if shared_tile_threads == "auto" else shared_tile_threads
    )

    if shared_debug_logs:
        print(
            "Resolved shared settings:",
            {
                "cache": resolved_cache,
                "image_threads": resolved_image_threads,
                "io_threads": resolved_io_threads,
                "tile_threads": resolved_tile_threads,
            },
        )

    return (
        resolved_cache,
        resolved_image_threads,
        resolved_io_threads,
        resolved_tile_threads,
    )


def _resolve_auto_cache_gb() -> int:
    total_memory_bytes = _get_total_memory_bytes()
    if total_memory_bytes is None:
        return 4096
    total_memory_mb = total_memory_bytes / (1024 ** 2)
    return max(1, math.ceil(total_memory_mb * 0.90))


def _get_total_memory_bytes() -> int | None:
    try:
        if hasattr(os, "sysconf"):
            if "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
                page_size = os.sysconf("SC_PAGE_SIZE")
                phys_pages = os.sysconf("SC_PHYS_PAGES")
                if isinstance(page_size, int) and isinstance(phys_pages, int):
                    total_bytes = page_size * phys_pages
                    if total_bytes > 0:
                        return total_bytes
    except (ValueError, OSError, AttributeError):
        pass

    try:
        import ctypes

        libc = ctypes.CDLL("libc.dylib")
        size = ctypes.c_uint64()
        size_len = ctypes.c_size_t(ctypes.sizeof(size))
        if libc.sysctlbyname(
            b"hw.memsize",
            ctypes.byref(size),
            ctypes.byref(size_len),
            None,
            0,
        ) == 0:
            total_bytes = int(size.value)
            if total_bytes > 0:
                return total_bytes
    except Exception:
        pass

    return None
