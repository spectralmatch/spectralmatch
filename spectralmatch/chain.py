import os
import shutil
import tempfile
import math
import time

from datetime import datetime

from typing import Any, Literal

from .handlers import _resolve_paths
from .joint_coregistration import joint_coregistration
from .match.match import Match
from .seamline.seamline import Seamline
from .types_and_validation import Universal, Pipeline as PipelineValidation
from .utils import align_rasters, mask_rasters, merge_rasters

AutoCache = Universal.Cache | Literal["auto"]
AutoThreads = Universal.Threads | Literal["auto"]
PipelineStep = Literal[
    "joint_coregistration",
    "global_regression",
    "local_block_adjustment",
    "align",
    "voronoi_center_seamline",
    "weighted_seamline",
    "mask",
    "merge",
]
PIPELINE_STEP_ORDER: tuple[PipelineStep, ...] = (
    "joint_coregistration",
    "align",
    "global_regression",
    "local_block_adjustment",
    "voronoi_center_seamline",
    "weighted_seamline",
    "mask",
    "merge",
)

MULTI_RASTER_STEPS = {
    "joint_coregistration",
    "global_regression",
    "local_block_adjustment",
    "align",
    "mask",
}
SEAMLINE_STEPS = {"voronoi_center_seamline", "weighted_seamline"}
DEFAULT_PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    "joint_coregistration",
    "global_regression",
    "local_block_adjustment",
    "voronoi_center_seamline",
    "mask",
    "merge",
)
STEP_TEMP_OUTPUTS = {
    "joint_coregistration": os.path.join("coregistered"),
    "global_regression": os.path.join("global"),
    "local_block_adjustment": os.path.join("local"),
    "align": os.path.join("aligned"),
    "mask": os.path.join("clip"),
    "voronoi_center_seamline": os.path.join("seamline", "ImageMasks.gpkg"),
    "weighted_seamline": os.path.join("seamline", "ImageMasks.gpkg"),
}


def pipeline(
    shared_input_images: Universal.SearchFolderOrListFiles,
    shared_output_image_path: Universal.CreateInFolderOrListFiles,
    *,
    shared_temp_dir: str | None = None,
    delete_temp_dir: bool = True,
    delete_previous_step: bool = False,
    shared_resume_from_steps: Literal["no", "yes", "validate"] = "no",
    shared_debug_logs: Universal.DebugLogs = False,
    shared_cache: AutoCache = "auto",
    shared_custom_nodata_value: Universal.CustomNodataValue = None,
    shared_window_size: Universal.WindowSize = 1024,
    shared_window_scales: tuple[int, ...] | None = (2, 4, 8, 16, 32),
    shared_image_threads: AutoThreads = "auto",
    shared_io_threads: AutoThreads = "auto",
    shared_tile_threads: AutoThreads = "auto",
    shared_concurrent_processing_backend: Universal.ConcurrentProcessingBackend = "process_pool",
    shared_dask_scheduler: Universal.DaskScheduler = None,
    shared_calculation_dtype: Universal.CalculationDtype = "float32",
    shared_output_dtype: Universal.CustomOutputDtype = None,
    shared_save_as_cog: Universal.SaveAsCog = False,
    steps: list[PipelineStep] | tuple[PipelineStep, ...] = DEFAULT_PIPELINE_STEPS,
    joint_coregistration_global_model: Literal["none", "translation", "similarity", "affine"] = "translation",
    joint_coregistration_global_image_position_preservation_weights: dict[str, float] | None = None,
    joint_coregistration_global_tie_point_alignment_strength: float = 1.0,
    joint_coregistration_local_model: Literal["none", "bilinear", "piecewise_affine"] = "piecewise_affine",
    joint_coregistration_local_image_position_preservation_weights: dict[str, float] | None = None,
    joint_coregistration_local_tie_point_alignment_strength: float = 1.0,
    joint_coregistration_local_grid_spacing: float = 500.0,
    joint_coregistration_local_smoothness_weight: float = 1.0,
    joint_coregistration_local_bending_weight: float = 1.0,
    joint_coregistration_local_anchor_falloff_distance: float = 500.0,
    joint_coregistration_feature_method: Literal["orb"] = "orb",
    joint_coregistration_maximum_tie_point_displacement: float | None = None,
    joint_coregistration_ransac_reprojection_threshold: float | None = None,
    joint_coregistration_robust_loss: Literal["none", "huber", "soft_l1", "cauchy"] = "huber",
    joint_coregistration_robust_loss_scale: float | None = None,
    joint_coregistration_save_adjustments: str | None = None,
    joint_coregistration_load_adjustments: str | None = None,
    joint_coregistration_resampling_method: Literal["nearest", "bilinear", "cubic", "lanczos"] = "bilinear",
    joint_coregistration_tap: bool = False,
    joint_coregistration_resolution: Universal.Resolution = None,
    joint_coregistration_build_overviews: bool = False,
    align_rasters_resampling_method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    align_rasters_tap: bool = False,
    align_rasters_resolution: Universal.Resolution = None,
    global_regression_vector_mask: Universal.VectorMask = None,
    global_regression_estimate_stats: bool = True,
    global_regression_specify_model_images: tuple[Literal["exclude", "include"], list[str]] | None = None,
    global_regression_custom_mean_factor: float = 1.0,
    global_regression_custom_std_factor: float = 1.0,
    global_regression_save_adjustments: str | None = None,
    global_regression_load_adjustments: str | None = None,
    global_regression_pif_method: Literal["entire", "flood_from_match_points"] = "entire",
    global_regression_pif_red_band_index: int | None = None,
    global_regression_pif_nir_band_index: int | None = None,
    global_regression_pif_vegetation_threshold: float = 0.2,
    global_regression_pif_inz_threshold: float = 0.25,
    global_regression_pif_region_radius: int = 5,
    global_regression_pif_max_samples: int | None = 100000,
    global_regression_pif_min_samples: int | None = 32,
    global_regression_pif_feature_method: Literal["orb"] = "orb",
    global_regression_pif_load_tie_points: str | None = None,
    global_regression_pif_save_inz: str | None = None,
    global_regression_build_overviews: bool = False,
    local_block_adjustment_vector_mask: Universal.VectorMask = None,
    local_block_adjustment_number_of_blocks: int | tuple[int, int] | Literal["coefficient_of_variation"] = 100,
    local_block_adjustment_alpha: float = 1.0,
    local_block_adjustment_correction_method: Literal["gamma", "linear", "offset"] = "offset",
    local_block_adjustment_save_block_maps: tuple[str, str] | None = None,
    local_block_adjustment_load_block_maps: tuple[str | None, list[str] | None] | None = None,
    local_block_adjustment_override_bounds_canvas_coords: tuple[float, float, float, float] | None = None,
    local_block_adjustment_build_overviews: bool = False,
    voronoi_center_seamline_aoi_path: str | None = None,
    voronoi_center_seamline_vector_mask: tuple[str, str] | None = None,
    voronoi_center_seamline_image_field_name: str = "image",
    voronoi_center_seamline_min_point_spacing: float = 10,
    voronoi_center_seamline_min_cut_length: float = 0,
    voronoi_center_seamline_debug_vectors_path: str | None = None,
    weighted_seamline_input_polygons: str | None = None,
    weighted_seamline_rank_function: str | None = None,
    weighted_seamline_image_field_name: str = "image",
    weighted_seamline_input_layer: str | None = None,
    weighted_seamline_output_layer: str = "seamlines",
    weighted_seamline_rank_descending: bool = True,
    mask_rasters_vector_mask: Universal.VectorMask = None,
    mask_rasters_include_touched_pixels: bool = False,
    merge_rasters_resolution: Literal["highest", "average", "lowest"] = "highest",
    merge_rasters_build_overviews: bool = True,
) -> dict[str, Any]:
    """
    Run the spectral matching workflow as an ordered pipeline.

    ``steps`` defines the exact step order. Intermediate outputs are written
    inside the pipeline temp directory. The final step writes to
    ``shared_output_image_path``:

    - If the final step writes multiple rasters, ``shared_output_image_path``
      must be a folder, a template containing ``$``, or a list of paths.
    - If the final step writes a single raster or vector, it must be a single
      file path without ``$``.

    Args:
        shared_window_scales: Overview factors shared by all steps with build_overviews enabled, default (2, 4, 8, 16, 32); None or an empty tuple disables overview creation for those steps.
    """
    Universal._validate(window_scales=shared_window_scales)
    temp_dir = shared_temp_dir or tempfile.mkdtemp(prefix="spectralmatch_pipeline_")
    if delete_temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
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
    PipelineValidation._validate_shared_pipeline(
        shared_output_image_path=shared_output_image_path,
        shared_temp_dir=shared_temp_dir,
        delete_temp_dir=delete_temp_dir,
        delete_previous_step=delete_previous_step,
        shared_resume_from_steps=shared_resume_from_steps,
    )
    Universal._validate(
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
        concurrent_processing_backend=shared_concurrent_processing_backend,
        dask_scheduler=shared_dask_scheduler,
    )

    resolved_steps = _validate_pipeline_steps(steps)
    last_step = resolved_steps[-1] if resolved_steps else None
    _validate_shared_output_for_last_step(
        shared_output_image_path=shared_output_image_path,
        last_step=last_step,
    )

    start_dt = datetime.now()
    start_perf = time.perf_counter()
    print(f"Pipeline start: {start_dt.isoformat(timespec='seconds')}")
    print(f"Pipeline temp dir: {temp_dir}")
    print(f"Number of input images: {len(input_image_paths)}")

    current_images: Universal.SearchFolderOrListFiles = shared_input_images
    seamline_mask_path: str | None = None
    seamline_mask_image_field_name: str | None = None
    previous_cleanup_paths: list[str] = []

    results: dict[str, Any] = {
        "temp_dir": temp_dir,
        "input_images": shared_input_images,
        "resolved_shared_cache": shared_cache,
        "resolved_shared_image_threads": shared_image_threads,
        "resolved_shared_io_threads": shared_io_threads,
        "resolved_shared_tile_threads": shared_tile_threads,
        "shared_concurrent_processing_backend": shared_concurrent_processing_backend,
        "shared_dask_scheduler": shared_dask_scheduler,
        "num_input_images": len(input_image_paths),
        "start_time": start_dt.isoformat(timespec="seconds"),
        "steps": resolved_steps,
        "shared_resume_from_steps": shared_resume_from_steps,
    }

    try:
        for step_index, step_name in enumerate(resolved_steps):
            is_last_step = step_index == len(resolved_steps) - 1
            step_cleanup_paths: list[str] = []

            if step_name == "joint_coregistration":
                output_images = (
                    shared_output_image_path
                    if is_last_step
                    else _step_temp_output(step_name, temp_dir)
                )
                current_images = joint_coregistration(
                    input_images=current_images,
                    output_images=output_images,
                    global_model=joint_coregistration_global_model,
                    global_image_position_preservation_weights=joint_coregistration_global_image_position_preservation_weights,
                    global_tie_point_alignment_strength=joint_coregistration_global_tie_point_alignment_strength,
                    local_model=joint_coregistration_local_model,
                    local_image_position_preservation_weights=joint_coregistration_local_image_position_preservation_weights,
                    local_tie_point_alignment_strength=joint_coregistration_local_tie_point_alignment_strength,
                    local_grid_spacing=joint_coregistration_local_grid_spacing,
                    local_smoothness_weight=joint_coregistration_local_smoothness_weight,
                    local_bending_weight=joint_coregistration_local_bending_weight,
                    local_anchor_falloff_distance=joint_coregistration_local_anchor_falloff_distance,
                    feature_method=joint_coregistration_feature_method,
                    maximum_tie_point_displacement=joint_coregistration_maximum_tie_point_displacement,
                    ransac_reprojection_threshold=joint_coregistration_ransac_reprojection_threshold,
                    robust_loss=joint_coregistration_robust_loss,
                    robust_loss_scale=joint_coregistration_robust_loss_scale,
                    save_adjustments=joint_coregistration_save_adjustments,
                    load_adjustments=joint_coregistration_load_adjustments,
                    resampling_method=joint_coregistration_resampling_method,
                    tap=joint_coregistration_tap,
                    resolution=joint_coregistration_resolution,
                    output_dtype=shared_output_dtype,
                    custom_nodata_value=shared_custom_nodata_value,
                    window_size=shared_window_size,
                    save_as_cog=shared_save_as_cog,
                    build_overviews=joint_coregistration_build_overviews,
                    window_scales=shared_window_scales,
                    cache=shared_cache,
                    image_threads=shared_image_threads,
                    io_threads=shared_io_threads,
                    tile_threads=shared_tile_threads,
                    debug_logs=shared_debug_logs,
                    resume_from_outputs=shared_resume_from_steps,
                )
                results["joint_coregistration"] = current_images
                step_cleanup_paths = _collect_step_cleanup_paths(
                    step_name, current_images, temp_dir
                )

            elif step_name == "global_regression":
                output_images = (
                    shared_output_image_path
                    if is_last_step
                    else _step_temp_output(step_name, temp_dir)
                )
                current_images = Match.global_regression(
                    input_images=current_images,
                    output_images=output_images,
                    calculation_dtype=shared_calculation_dtype,
                    output_dtype=shared_output_dtype,
                    vector_mask=global_regression_vector_mask,
                    debug_logs=shared_debug_logs,
                    custom_nodata_value=shared_custom_nodata_value,
                    cache=shared_cache,
                    image_threads=shared_image_threads,
                    io_threads=shared_io_threads,
                    tile_threads=shared_tile_threads,
                    concurrent_processing_backend=shared_concurrent_processing_backend,
                    dask_scheduler=shared_dask_scheduler,
                    window_size=shared_window_size,
                    save_as_cog=shared_save_as_cog,
                    estimate_stats=global_regression_estimate_stats,
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
                    pif_load_tie_points=global_regression_pif_load_tie_points,
                    pif_save_inz=global_regression_pif_save_inz,
                    build_overviews=global_regression_build_overviews,
                    window_scales=shared_window_scales,
                    resume_from_outputs=shared_resume_from_steps,
                )
                results["global_regression"] = current_images
                step_cleanup_paths = _collect_step_cleanup_paths(
                    step_name, current_images, temp_dir
                )

            elif step_name == "local_block_adjustment":
                output_images = (
                    shared_output_image_path
                    if is_last_step
                    else _step_temp_output(step_name, temp_dir)
                )
                current_images = Match.local_block_adjustment(
                    input_images=current_images,
                    output_images=output_images,
                    calculation_dtype=shared_calculation_dtype,
                    output_dtype=shared_output_dtype,
                    vector_mask=local_block_adjustment_vector_mask,
                    debug_logs=shared_debug_logs,
                    custom_nodata_value=shared_custom_nodata_value,
                    cache=shared_cache,
                    image_threads=shared_image_threads,
                    io_threads=shared_io_threads,
                    tile_threads=shared_tile_threads,
                    concurrent_processing_backend=shared_concurrent_processing_backend,
                    dask_scheduler=shared_dask_scheduler,
                    window_size=shared_window_size,
                    save_as_cog=shared_save_as_cog,
                    number_of_blocks=local_block_adjustment_number_of_blocks,
                    alpha=local_block_adjustment_alpha,
                    correction_method=local_block_adjustment_correction_method,
                    save_block_maps=local_block_adjustment_save_block_maps,
                    load_block_maps=local_block_adjustment_load_block_maps,
                    override_bounds_canvas_coords=local_block_adjustment_override_bounds_canvas_coords,
                    build_overviews=local_block_adjustment_build_overviews,
                    window_scales=shared_window_scales,
                    resume_from_outputs=shared_resume_from_steps,
                )
                results["local_block_adjustment"] = current_images
                step_cleanup_paths = _collect_step_cleanup_paths(
                    step_name, current_images, temp_dir
                )

            elif step_name == "align":
                output_images = (
                    shared_output_image_path
                    if is_last_step
                    else _step_temp_output(step_name, temp_dir)
                )
                current_images = align_rasters(
                    input_images=current_images,
                    output_images=output_images,
                    resampling_method=align_rasters_resampling_method,
                    tap=align_rasters_tap,
                    resolution=align_rasters_resolution,
                    window_size=shared_window_size,
                    debug_logs=shared_debug_logs,
                    cache=shared_cache,
                    image_threads=shared_image_threads,
                    io_threads=shared_io_threads,
                    tile_threads=shared_tile_threads,
                    concurrent_processing_backend=shared_concurrent_processing_backend,
                    dask_scheduler=shared_dask_scheduler,
                    resume_from_outputs=shared_resume_from_steps,
                )
                results["align"] = current_images
                results["align_rasters"] = current_images
                step_cleanup_paths = _collect_step_cleanup_paths(
                    step_name, current_images, temp_dir
                )

            elif step_name == "voronoi_center_seamline":
                seamline_mask_path = (
                    shared_output_image_path
                    if is_last_step
                    else _step_temp_output(step_name, temp_dir)
                )
                if not isinstance(seamline_mask_path, str):
                    raise ValueError(
                        "shared_output_image_path must be a single file path when the final step is a seamline."
                    )
                seamline_mask_image_field_name = voronoi_center_seamline_image_field_name
                Seamline.voronoi(
                    input_images=current_images,
                    output_mask=seamline_mask_path,
                    aoi_path=voronoi_center_seamline_aoi_path,
                    vector_mask=voronoi_center_seamline_vector_mask,
                    image_field_name=voronoi_center_seamline_image_field_name,
                    min_point_spacing=voronoi_center_seamline_min_point_spacing,
                    min_cut_length=voronoi_center_seamline_min_cut_length,
                    debug_logs=shared_debug_logs,
                    debug_vectors_path=voronoi_center_seamline_debug_vectors_path,
                    resume_from_outputs=shared_resume_from_steps,
                )
                results["voronoi_center_seamline"] = seamline_mask_path
                step_cleanup_paths = _collect_step_cleanup_paths(
                    step_name, seamline_mask_path, temp_dir
                )

            elif step_name == "weighted_seamline":
                seamline_mask_path = (
                    shared_output_image_path
                    if is_last_step
                    else _step_temp_output(step_name, temp_dir)
                )
                if not isinstance(seamline_mask_path, str):
                    raise ValueError(
                        "shared_output_image_path must be a single file path when the final step is a seamline."
                    )
                seamline_mask_image_field_name = weighted_seamline_image_field_name
                Seamline.weighted(
                    input_polygons=weighted_seamline_input_polygons,
                    output_mask=seamline_mask_path,
                    rank_function=weighted_seamline_rank_function,
                    image_field_name=weighted_seamline_image_field_name,
                    input_layer=weighted_seamline_input_layer,
                    output_layer=weighted_seamline_output_layer,
                    rank_descending=weighted_seamline_rank_descending,
                    debug_logs=shared_debug_logs,
                    resume_from_outputs=shared_resume_from_steps,
                )
                results["weighted_seamline"] = seamline_mask_path
                step_cleanup_paths = _collect_step_cleanup_paths(
                    step_name, seamline_mask_path, temp_dir
                )

            elif step_name == "mask":
                output_images = (
                    shared_output_image_path
                    if is_last_step
                    else _step_temp_output(step_name, temp_dir)
                )
                clip_vector_mask = mask_rasters_vector_mask
                if clip_vector_mask is None and seamline_mask_path is not None:
                    clip_vector_mask = (
                        "include",
                        seamline_mask_path,
                        seamline_mask_image_field_name,
                    )
                if clip_vector_mask is None:
                    raise ValueError(
                        "mask_rasters requires a vector mask. Set mask_rasters_vector_mask "
                        "or run a seamline step earlier in the pipeline."
                    )

                current_images = mask_rasters(
                    input_images=current_images,
                    output_images=output_images,
                    vector_mask=clip_vector_mask,
                    window_size=shared_window_size,
                    debug_logs=shared_debug_logs,
                    cache=shared_cache,
                    image_threads=shared_image_threads,
                    io_threads=shared_io_threads,
                    tile_threads=shared_tile_threads,
                    concurrent_processing_backend=shared_concurrent_processing_backend,
                    dask_scheduler=shared_dask_scheduler,
                    include_touched_pixels=mask_rasters_include_touched_pixels,
                    custom_nodata_value=shared_custom_nodata_value,
                    resume_from_outputs=shared_resume_from_steps,
                )
                results["mask"] = current_images
                results["mask_rasters"] = current_images
                step_cleanup_paths = _collect_step_cleanup_paths(
                    step_name, current_images, temp_dir
                )

            elif step_name == "merge":
                if not isinstance(shared_output_image_path, str):
                    raise ValueError(
                        "shared_output_image_path must be a single file path when the final step is merge."
                    )
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
                    window_scales=shared_window_scales,
                    resume_from_outputs=shared_resume_from_steps,
                )
                current_images = merged_output
                results["merge"] = merged_output
                results["merge_rasters"] = merged_output
                step_cleanup_paths = _collect_step_cleanup_paths(
                    step_name, merged_output, temp_dir
                )

            else:
                raise ValueError(f"Unsupported pipeline step: {step_name}")

            if delete_previous_step and previous_cleanup_paths:
                _delete_step_outputs_if_inactive(
                    previous_cleanup_paths=previous_cleanup_paths,
                    current_images=current_images,
                    seamline_mask_path=seamline_mask_path,
                    temp_dir=temp_dir,
                    debug_logs=shared_debug_logs,
                )
            previous_cleanup_paths = step_cleanup_paths

        if not resolved_steps:
            results["output"] = current_images
        elif last_step in SEAMLINE_STEPS:
            results["output"] = seamline_mask_path
        else:
            results["output"] = current_images

        end_dt = datetime.now()
        duration_seconds = round(time.perf_counter() - start_perf, 2)

        final_results = {
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
        for key, value in results.items():
            if key not in final_results and value is not None:
                final_results[key] = value
        return final_results
    finally:
        if delete_temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def _validate_pipeline_steps(
    steps: list[PipelineStep] | tuple[PipelineStep, ...],
) -> list[PipelineStep]:
    if not isinstance(steps, (list, tuple)):
        raise ValueError("steps must be a list or tuple of pipeline step names.")
    resolved_steps = list(steps)
    allowed_steps = set(PIPELINE_STEP_ORDER)
    for step_name in resolved_steps:
        if step_name not in allowed_steps:
            raise ValueError(f"Unsupported pipeline step: {step_name}")
    if len(set(resolved_steps)) != len(resolved_steps):
        raise ValueError("steps cannot contain duplicate values.")
    if len(SEAMLINE_STEPS.intersection(resolved_steps)) > 1:
        raise ValueError(
            "steps can include at most one seamline step: "
            "'voronoi_center_seamline' or 'weighted_seamline'."
        )
    return resolved_steps


def _validate_shared_output_for_last_step(
    *,
    shared_output_image_path: Universal.CreateInFolderOrListFiles,
    last_step: PipelineStep | None,
) -> None:
    if last_step is None:
        return

    if last_step in MULTI_RASTER_STEPS:
        Universal._validate(output_images=shared_output_image_path)
        if isinstance(shared_output_image_path, str):
            basename = os.path.basename(shared_output_image_path)
            if basename.count(".") and "$" not in shared_output_image_path:
                raise ValueError(
                    "When the final pipeline step writes multiple rasters, "
                    "shared_output_image_path must be a folder, a template containing '$', "
                    "or a list of output paths."
                )
        return

    if not isinstance(shared_output_image_path, str):
        raise ValueError(
            "When the final pipeline step writes a single output, "
            "shared_output_image_path must be a single file path."
        )
    if "$" in shared_output_image_path:
        raise ValueError(
            "When the final pipeline step writes a single output, "
            "shared_output_image_path cannot contain '$'."
        )
    if not os.path.basename(shared_output_image_path).count("."):
        raise ValueError(
            "When the final pipeline step writes a single output, "
            "shared_output_image_path must be a file path, not a folder."
        )
def _step_temp_output(step_name: PipelineStep, temp_dir: str) -> str:
    return os.path.join(temp_dir, STEP_TEMP_OUTPUTS[step_name])


def _collect_step_cleanup_paths(
    step_name: str,
    step_output: Any,
    temp_dir: str,
) -> list[str]:
    if step_name in MULTI_RASTER_STEPS:
        if isinstance(step_output, list):
            dirs = {
                os.path.dirname(path)
                for path in step_output
                if _path_is_within(path, temp_dir)
            }
            return sorted(dirs) if dirs else []
        return []
    if isinstance(step_output, str) and _path_is_within(step_output, temp_dir):
        return [step_output]
    return []


def _delete_step_outputs_if_inactive(
    *,
    previous_cleanup_paths: list[str],
    current_images: Universal.SearchFolderOrListFiles,
    seamline_mask_path: str | None,
    temp_dir: str,
    debug_logs: bool,
) -> None:
    active_paths = set()
    if isinstance(current_images, list):
        active_paths.update(current_images)
    elif isinstance(current_images, str):
        active_paths.add(current_images)
    if seamline_mask_path:
        active_paths.add(seamline_mask_path)

    for path in previous_cleanup_paths:
        if not _path_is_within(path, temp_dir):
            continue
        if any(_paths_overlap(path, active_path) for active_path in active_paths):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            if debug_logs:
                print(f"Deleted previous step directory: {path}")
        elif os.path.exists(path):
            os.remove(path)
            if debug_logs:
                print(f"Deleted previous step file: {path}")


def _path_is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def _paths_overlap(path_a: str, path_b: str) -> bool:
    abs_a = os.path.abspath(path_a)
    abs_b = os.path.abspath(path_b)
    if abs_a == abs_b:
        return True
    if os.path.isdir(abs_a):
        return os.path.commonpath([abs_a, abs_b]) == abs_a
    if os.path.isdir(abs_b):
        return os.path.commonpath([abs_a, abs_b]) == abs_b
    return False


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


def _resolve_auto_cache_gb() -> float:
    total_memory_bytes = _get_total_memory_bytes()
    if total_memory_bytes is None:
        return 4.0
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    return max(1.0, total_memory_gb * 0.90)


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
