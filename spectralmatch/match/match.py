import json

import numpy as np
from concurrent.futures import as_completed
from osgeo import gdal
from typing import Literal, Tuple, List

from ..handlers import _resolve_paths, _resolve_nodata_value, _check_raster_requirements
from ..pif.pif import Pif
from ..types_and_validation import Universal, Match as MatchValidation
from ..utils import (
    create_masked_vrts,
    _set_gdal_cache,
    _set_gdal_workers,
    _resolve_gdal_dtype,
    compute_overviews,
    _get_gdal_bounds,
    _gdal_dtype_str_to_enum,
)
from ..utils_multiprocessing import _resolve_parallel_config, _get_executor
from .global_regression import (
    _solve_global_model,
    _apply_adjustments_process_image,
    _save_adjustments,
    _validate_adjustment_model_structure,
    _find_overlaps,
    _overlap_stats_process_image,
    _whole_stats_process_image,
)
from .local_block_adjustment import (
    _get_pre_computed_block_maps,
    _get_bounding_rectangle,
    _compute_mosaic_coefficient_of_variation,
    _calculate_block_process_image,
    _compute_reference_blocks,
    _download_block_map,
    _apply_adjustment_process_image as _apply_local_adjustment_process_image,
    _compute_block_size,
)


class Match:
    @staticmethod
    def _setup_images(
        *,
        input_images,
        output_images,
        default_output_pattern: str,
        calculation_dtype,
        output_dtype,
        vector_mask,
        debug_logs,
        custom_nodata_value,
        cache,
        image_threads,
        io_threads,
        tile_threads,
        window_size,
        save_as_cog,
        estimate_stats=None,
    ) -> dict:
        validate_kwargs = {
            "input_images": input_images,
            "output_images": output_images,
            "save_as_cog": save_as_cog,
            "debug_logs": debug_logs,
            "vector_mask": vector_mask,
            "window_size": window_size,
            "custom_nodata_value": custom_nodata_value,
            "calculation_dtype": calculation_dtype,
            "output_dtype": output_dtype,
            "cache": cache,
            "image_threads": image_threads,
            "io_threads": io_threads,
            "tile_threads": tile_threads,
        }
        if estimate_stats is not None:
            validate_kwargs["estimate_stats"] = estimate_stats
        Universal.validate(**validate_kwargs)

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
                "default_file_pattern": default_output_pattern,
            },
        )
        input_image_names = _resolve_paths("name", input_image_paths)

        input_image_path_pairs = dict(zip(input_image_names, input_image_paths))
        output_image_path_pairs = dict(zip(input_image_names, output_image_paths))
        if debug_logs:
            print(f"Input images: {input_image_paths}")
            print(f"Output images: {output_image_paths}")

        _check_raster_requirements(
            input_image_paths,
            debug_logs,
            check_geotransform=True,
            check_crs=True,
            check_bands=True,
            check_nodata=True,
        )

        resolved_output_dtype = _resolve_gdal_dtype(
            output_dtype,
            input_image_paths[0],
            debug_logs,
        )
        nodata_val = _resolve_nodata_value(input_image_paths[0], custom_nodata_value)
        image_backend = "thread"
        image_threads_on, image_thread_workers = _resolve_parallel_config(image_threads)
        tile_thread_on, tile_thread_workers = _resolve_parallel_config(tile_threads)

        return {
            "input_image_paths": input_image_paths,
            "output_image_paths": output_image_paths,
            "input_image_names": input_image_names,
            "input_image_path_pairs": input_image_path_pairs,
            "output_image_path_pairs": output_image_path_pairs,
            "output_dtype": resolved_output_dtype,
            "nodata_val": nodata_val,
            "image_backend": image_backend,
            "image_threads_on": image_threads_on,
            "image_thread_workers": image_thread_workers,
            "tile_thread_on": tile_thread_on,
            "tile_thread_workers": tile_thread_workers,
        }

    @staticmethod
    def global_regression(
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
        estimate_stats: bool = True,
        specify_model_images: MatchValidation.SpecifyModelImages = None,
        custom_mean_factor: float = 1.0,
        custom_std_factor: float = 1.0,
        save_adjustments: str | None = None,
        load_adjustments: str | None = None,
        pif_method: Literal["entire", "flood_from_match_points"] = "flood_from_match_points",
        pif_red_band_index: int | None = None,
        pif_nir_band_index: int | None = None,
        pif_vegetation_threshold: float = 0.2,
        pif_inz_threshold: float = 0.25,
        pif_region_radius: int = 5,
        pif_max_samples: int | None = 10000,
        pif_min_samples: int | None = 10,
        pif_feature_method: Literal["orb"] = "orb",
        pif_save_inz: str | None = None,
        build_overviews: bool = False,
    ) -> list:
        """Performs global radiometric normalization across overlapping images using least squares regression.

Args:
    input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
    output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Global.tif), ["/input/one.tif", "/input/two.tif"].
    calculation_dtype (str, optional): Precision for internal calculations. Defaults to "float32".
    output_dtype (str | None, optional): Data type for output rasters. Defaults to input image dtype.
    vector_mask (Tuple[Literal["include", "exclude"], str, Optional[str]] | None): Mask to limit stats calculation to specific areas in the format of a tuple with two or three items: literal "include" or "exclude" the mask area, str path to the vector file, optional str of field name in vector file that *includes* (can be substring) input image name to filter geometry by. Loaded stats won't have this applied to them. The matching solution is still applied to these areas in the output. Defaults to None for no mask.
    debug_logs (bool, optional): If True, prints debug info and progress. Defaults to False.
    custom_nodata_value (float | int | None, optional): Overrides detected NoData value. Defaults to None.
    cache (float | None): Controls GDAL cache size in GB. Defaults to preset cache size. Applied via GDAL_CACHEMAX.
    image_threads (Literal["cpu"] | int | None): Parallelism for per-image operations. "cpu" to get number of cores, int to assign number, and None to disable image level parallelism.
    io_threads (Literal["cpu"] | int | None): Parallelism for IO operations. "cpu" to get number of cores, int to assign number, and None to disable io level parallelism.
    tile_threads (Literal["cpu"] | int | None): "cpu" to get number of cores, int to assign number, and None to disable tile level parallelism.
    window_size (int | None): Output image tile size. Defaults to input image tile size.
    save_as_cog (bool): If True, saves output as a Cloud-Optimized GeoTIFF using proper band and block order.
    estimate_stats (bool): If True, use an estimate algorithm to calculate the mean and sd to increase processing speeds. If False, use the exact algorithm. Defaults to True.
    specify_model_images (Tuple[Literal["exclude", "include"], List[str]] | None ): First item in tuples sets weather to 'include' or 'exclude' the listed images from model building statistics. Second item is the list of image names (without their extension) to apply criteria to. For example, if this param is only set to 'include' one image, all other images will be matched to that one image. Defaults to no exclusion.
    custom_mean_factor (float, optional): Weight for mean constraints in regression. Defaults to 1.0.
    custom_std_factor (float, optional): Weight for standard deviation constraints in regression. Defaults to 1.0.
    save_adjustments (str | None, optional): The output path of a .json file to save adjustments parameters. Defaults to not saving.
    load_adjustments (str | None, optional): If set, loads saved whole and overlapping statistics only for images that exist in the .json file. Other images will still have their statistics calculated. Defaults to None.
    pif_method (Literal["entire", "flood_from_match_points"], optional): Method used to select overlap pixels for the matching solution. Defaults to "entire".
    pif_red_band_index (int | None, optional): Index of the red band used for NDVI-based vegetation filtering. Defaults to None.
    pif_nir_band_index (int | None, optional): Index of the NIR band used for NDVI-based vegetation filtering. Defaults to None.
    pif_vegetation_threshold (float, optional): Vegetation threshold used with NDVI filtering. Defaults to 0.2.
    pif_inz_threshold (float, optional): Integrated normalized Z-score threshold used for flood_from_match_points. Defaults to 0.25.
    pif_region_radius (int, optional): Radius used to expand matched points into PIF seed areas. Defaults to 5.
    pif_max_samples (int | None, optional): Maximum number of PIF samples to keep. Defaults to 100000.
    pif_min_samples (int | None, optional): Minimum number of PIF samples required. Defaults to 32.
    pif_feature_method (Literal["orb"], optional): Feature matching method used for flood_from_match_points. Defaults to "orb".
    pif_save_inz (str | None, optional): Output path to save the INZ raster. If two "$" are given, the first is the main basename and the second is the reference basename. Defaults to None.
    build_overviews (bool, optional): If True, computes overviews. Defaults to False.

Returns:
    List[str]: Paths to the globally adjusted output raster images."""
        print("Start global regression")

        MatchValidation.validate_match(
            specify_model_images=specify_model_images,
        )
        MatchValidation.validate_global_regression(
            custom_mean_factor=custom_mean_factor,
            custom_std_factor=custom_std_factor,
            save_adjustments=save_adjustments,
            load_adjustments=load_adjustments,
            pif_method=pif_method,
            pif_feature_method=pif_feature_method,
            pif_save_inz=pif_save_inz,
        )

        setup = Match._setup_images(
            input_images=input_images,
            output_images=output_images,
            default_output_pattern="$_Global.tif",
            calculation_dtype=calculation_dtype,
            output_dtype=output_dtype,
            vector_mask=vector_mask,
            debug_logs=debug_logs,
            custom_nodata_value=custom_nodata_value,
            cache=cache,
            image_threads=image_threads,
            io_threads=io_threads,
            tile_threads=tile_threads,
            window_size=window_size,
            save_as_cog=save_as_cog,
            estimate_stats=estimate_stats,
        )
        input_image_paths = setup["input_image_paths"]
        output_image_paths = setup["output_image_paths"]
        input_image_names = setup["input_image_names"]
        input_image_path_pairs = setup["input_image_path_pairs"]
        output_image_path_pairs = setup["output_image_path_pairs"]
        output_dtype = setup["output_dtype"]
        nodata_val = setup["nodata_val"]
        image_backend = setup["image_backend"]
        image_threads_on = setup["image_threads_on"]
        image_thread_workers = setup["image_thread_workers"]
        tile_thread_on = setup["tile_thread_on"]
        tile_thread_workers = setup["tile_thread_workers"]

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
            print(f"    Matched adjustments (to override) ({len(matched)}):", sorted(matched))
            print(
                f"    Only in loaded adjustments (to add) ({len(only_loaded)}):",
                sorted(only_loaded),
            )
            print(f"    Only in input (to calculate) ({len(only_input)}):", sorted(only_input))

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
            print(f"    Included in model ({len(included_names)}): {sorted(included_names)}")
            if specify_model_images:
                print(f"    Excluded from model ({len(excluded_names)}): {sorted(excluded_names)}")
            else:
                print("    Excluded from model (0): []")

        input_image_masked_path_pairs = create_masked_vrts(
            input_image_path_pairs,
            vector_mask=vector_mask,
            nodata_value=nodata_val,
            debug_logs=debug_logs,
        )

        if debug_logs:
            print("Calculating statistics")
        num_bands = gdal.Open(next(iter(input_image_path_pairs.values()))).RasterCount
        all_bounds = {name: _get_gdal_bounds(path) for name, path in input_image_path_pairs.items()}
        overlapping_pairs = _find_overlaps(all_bounds)

        all_overlap_stats = {}
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

        parallel_args = [
            (
                tile_thread_on,
                tile_thread_workers,
                num_bands,
                input_image_masked_path_pairs[name_i],
                input_image_masked_path_pairs[name_j],
                name_i,
                name_j,
                all_bounds[name_i],
                all_bounds[name_j],
                estimate_stats,
                debug_logs,
            )
            for name_i, name_j in overlapping_pairs
            if name_i not in loaded_model
            or name_j not in loaded_model.get(name_i, {}).get("overlap_stats", {})
        ]
        if image_threads_on:
            with _get_executor(image_backend, image_thread_workers) as executor:
                futures = [executor.submit(_overlap_stats_process_image, *args) for args in parallel_args]
                for future in as_completed(futures):
                    stats = future.result()
                    for outer, inner in stats.items():
                        all_overlap_stats.setdefault(outer, {}).update(inner)
        else:
            for args in parallel_args:
                stats = _overlap_stats_process_image(*args)
                for outer, inner in stats.items():
                    all_overlap_stats.setdefault(outer, {}).update(inner)

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

        parallel_args = [
            (
                tile_thread_on,
                tile_thread_workers,
                image_path,
                num_bands,
                image_name,
                estimate_stats,
                debug_logs,
            )
            for image_name, image_path in input_image_masked_path_pairs.items()
            if image_name not in loaded_model
        ]
        if image_threads_on:
            with _get_executor(image_backend, image_thread_workers) as executor:
                futures = [executor.submit(_whole_stats_process_image, *args) for args in parallel_args]
                for future in as_completed(futures):
                    all_whole_stats.update(future.result())
        else:
            for args in parallel_args:
                all_whole_stats.update(_whole_stats_process_image(*args))

        all_image_names = list(dict.fromkeys(input_image_names + list(loaded_model.keys())))
        num_total = len(all_image_names)
        if debug_logs:
            print(
                f"\nCreating model for {len(all_image_names)} total images from {len(included_names)} included:"
            )
            print(f"    {'ID':<4}\t{'Source':<6}\t{'Inclusion':<8}\tName")
            for i, name in enumerate(all_image_names):
                source = "load" if name in (matched | only_loaded) else "calc"
                included = "incl" if name in included_names else "excl"
                print(f"    {i:<4}\t{source:<6}\t{included:<8}\t{name}")

        if pif_method == "flood_from_match_points":
            if debug_logs:
                print("Using flood_from_match_points PIF adjustment parameters")
            all_params = Pif.flood_from_match_points(
                input_images=input_image_paths,
                input_image_names=input_image_names,
                included_names=included_names,
                overlapping_pairs=overlapping_pairs,
                calculation_dtype=calculation_dtype,
                custom_nodata_value=custom_nodata_value,
                red_band_index=pif_red_band_index,
                nir_band_index=pif_nir_band_index,
                vegetation_threshold=pif_vegetation_threshold,
                inz_threshold=pif_inz_threshold,
                region_radius=pif_region_radius,
                max_samples=pif_max_samples,
                min_samples=pif_min_samples,
                feature_method=pif_feature_method,
                custom_mean_factor=custom_mean_factor,
                custom_std_factor=custom_std_factor,
                debug_logs=debug_logs,
                cache=cache,
                image_threads=image_threads,
                io_threads=io_threads,
                tile_threads=tile_threads,
                save_inz=pif_save_inz,
            )
        else:
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
                debug_logs,
            )

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

        if debug_logs:
            print("Apply adjustments and saving results for:")
        parallel_args = [
            (
                tile_thread_on,
                tile_thread_workers,
                name,
                img_path,
                output_image_path_pairs[name],
                np.array([all_params[b, 2 * idx, 0] for b in range(num_bands)]),
                np.array([all_params[b, 2 * idx + 1, 0] for b in range(num_bands)]),
                num_bands,
                nodata_val,
                window_size,
                output_dtype,
                calculation_dtype,
                save_as_cog,
                debug_logs,
            )
            for idx, (name, img_path) in enumerate(input_image_path_pairs.items())
        ]
        if image_threads_on:
            with _get_executor(image_backend, image_thread_workers) as executor:
                futures = [executor.submit(_apply_adjustments_process_image, *args) for args in parallel_args]
                for future in as_completed(futures):
                    future.result()
        else:
            for args in parallel_args:
                _apply_adjustments_process_image(*args)

        if build_overviews:
            compute_overviews(
                input_images_paths=output_image_paths,
                cache=cache,
                io_threads=io_threads,
                image_threads=image_threads,
                tile_threads=tile_threads,
                debug_logs=debug_logs,
            )
        return output_image_paths


    @staticmethod
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
        correction_method: Literal["gamma", "linear", "offset"] = "offset",
        save_block_maps: Tuple[str, str] | None = None,
        load_block_maps: (
            Tuple[str, List[str]] | Tuple[str, None] | Tuple[None, List[str]] | None
        ) = None,
        override_bounds_canvas_coords: Tuple[float, float, float, float] | None = None,
        build_overviews: bool = False,
    ) -> list:
        """Performs local radiometric adjustment on a set of raster images using block-based statistics.

Args:
    input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
    output_images (str | List[str], required): Defines output files from a template path, folder, or list of paths (with the same length as the input). Specify like: "/input/files/$.tif", "/input/folder" (assumes $_Global.tif), ["/input/one.tif", "/input/two.tif"].
    calculation_dtype (str, optional): Precision for internal calculations. Defaults to "float32".
    output_dtype (str | None, optional): Data type for output rasters. Defaults to input image dtype.
    vector_mask (Tuple[Literal["include", "exclude"], str, Optional[str]] | None): Mask to limit stats calculation to specific areas in the format of a tuple with two or three items: literal "include" or "exclude" the mask area, str path to the vector file, optional str of field name in vector file that *includes* (can be substring) input image name to filter geometry by. Loaded stats won't have this applied to them. The matching solution is still applied to these areas in the output. Defaults to None for no mask.
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
    correction_method (Literal["gamma", "linear", "offset"], optional): Local correction method. Defaults to "gamma". Offset is commended for images with negative values.
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
    build_overviews (bool, optional): If True, computes overviews. Defaults to False.

Returns:
    List[str]: Paths to the locally adjusted output raster images."""
        print("Start local block adjustment")

        MatchValidation.validate_local_block_adjustment(
            number_of_blocks=number_of_blocks,
            alpha=alpha,
            correction_method=correction_method,
            save_block_maps=save_block_maps,
            load_block_maps=load_block_maps,
            override_bounds_canvas_coords=override_bounds_canvas_coords,
        )

        setup = Match._setup_images(
            input_images=input_images,
            output_images=output_images,
            default_output_pattern="$_Local.tif",
            calculation_dtype=calculation_dtype,
            output_dtype=output_dtype,
            vector_mask=vector_mask,
            debug_logs=debug_logs,
            custom_nodata_value=custom_nodata_value,
            cache=cache,
            image_threads=image_threads,
            io_threads=io_threads,
            tile_threads=tile_threads,
            window_size=window_size,
            save_as_cog=save_as_cog,
        )
        input_image_paths = setup["input_image_paths"]
        output_image_paths = setup["output_image_paths"]
        input_image_names = setup["input_image_names"]
        input_image_path_pairs = setup["input_image_path_pairs"]
        output_image_path_pairs = setup["output_image_path_pairs"]
        output_dtype = setup["output_dtype"]
        nodata_val = setup["nodata_val"]
        image_backend = setup["image_backend"]
        image_threads_on = setup["image_threads_on"]
        image_thread_workers = setup["image_thread_workers"]
        tile_thread_on = setup["tile_thread_on"]
        tile_thread_workers = setup["tile_thread_workers"]

        input_image_path_pairs_masked = create_masked_vrts(
            input_image_path_pairs,
            vector_mask=vector_mask,
            nodata_value=nodata_val,
            debug_logs=debug_logs,
        )
        if debug_logs:
            print(f"Global nodata value: {nodata_val}")
        num_bands = gdal.Open(next(iter(input_image_path_pairs.values()))).RasterCount

        loaded_names = []
        if load_block_maps:
            (
                loaded_block_local_means,
                loaded_block_reference_mean,
                loaded_num_row,
                loaded_num_col,
                loaded_bounds_canvas_coords,
            ) = _get_pre_computed_block_maps(
                load_block_maps,
                calculation_dtype,
                debug_logs,
            )
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
            only_loaded = [l for l in loaded_names if not any(n in l for n in input_image_names)]
            only_input = [n for n in input_image_names if not any(n in l for l in loaded_names)]
        else:
            only_input = input_image_names
            matched = []
            only_loaded = []
            block_reference_mean = None

        if debug_logs:
            print(
                f"Total images: input images: {len(input_image_names)}, loaded local block maps: {len(loaded_names) if load_block_maps else 0}:"
            )
            print(f"    Matched local block maps (to override) ({len(matched)}):", sorted(matched))
            print(
                f"    Only in loaded local block maps (to use) ({len(only_loaded)}):",
                sorted(only_loaded),
            )
            print(f"    Only in input (to compute) ({len(only_input)}):", sorted(only_input))

        if save_block_maps:
            reference_map_path, local_map_path = save_block_maps

        if not override_bounds_canvas_coords:
            if not load_block_maps:
                bounds_canvas_coords = _get_bounding_rectangle(input_image_paths)
            else:
                bounds_canvas_coords = loaded_bounds_canvas_coords
        else:
            bounds_canvas_coords = override_bounds_canvas_coords
            if load_block_maps and bounds_canvas_coords != loaded_bounds_canvas_coords:
                raise ValueError(
                    "Override bounds canvas coordinates do not match loaded block maps bounds"
                )

        if not load_block_maps:
            if isinstance(number_of_blocks, int):
                num_row, num_col = _compute_block_size(input_image_paths, number_of_blocks, bounds_canvas_coords)
            elif isinstance(number_of_blocks, tuple):
                num_row, num_col = number_of_blocks
            else:
                num_row, num_col = _compute_mosaic_coefficient_of_variation(
                    input_image_paths, nodata_val, debug_logs
                )
        else:
            num_row, num_col = loaded_num_row, loaded_num_col

        if debug_logs:
            print("Computing local block maps:")
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
                    tile_thread_workers,
                )
                for name, path in local_blocks_to_calculate.items()
            ]
            if image_threads_on:
                with _get_executor(image_backend, image_thread_workers) as executor:
                    results = [
                        f.result()
                        for f in [executor.submit(_calculate_block_process_image, *arg) for arg in args]
                    ]
            else:
                results = [_calculate_block_process_image(*arg) for arg in args]
            block_local_means = {name: mean for name, mean in results}
            overlap = set(block_local_means) & set(local_blocks_to_load)
            if overlap:
                raise ValueError(f"Duplicate keys when merging loaded and computed blocks: {overlap}")
            block_local_means = {**block_local_means, **local_blocks_to_load}
        else:
            block_local_means = local_blocks_to_load

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
                np.nan_to_num(block_reference_mean, nan=nodata_val)
                if nodata_val is not None
                else block_reference_mean,
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
                    np.nan_to_num(block_local_mean, nan=nodata_val)
                    if nodata_val is not None
                    else block_local_mean,
                    bounds_canvas_coords,
                    local_map_path.replace("$", name),
                    srs,
                    calculation_dtype,
                    nodata_val,
                    num_col,
                    num_row,
                )

        if debug_logs:
            print("Computing local correction, applying, and saving:")
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
                futures = [executor.submit(_apply_local_adjustment_process_image, *arg) for arg in args]
                for future in as_completed(futures):
                    future.result()
        else:
            for arg in args:
                _apply_local_adjustment_process_image(*arg)

        if build_overviews:
            compute_overviews(
                input_images_paths=output_image_paths,
                cache=cache,
                io_threads=io_threads,
                image_threads=image_threads,
                tile_threads=tile_threads,
                debug_logs=debug_logs,
            )
        return output_image_paths
