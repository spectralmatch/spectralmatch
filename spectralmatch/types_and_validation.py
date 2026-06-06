from typing import Tuple, List, Literal, Optional

_UNSET = object()


# Universal types
class Universal:
    SearchFolderOrListFiles = str | List[str]
    CreateInFolderOrListFiles = str | List[str]
    SaveAsCog = bool  # Default: True
    DebugLogs = bool  # Default: False
    VectorMask = Tuple[Literal["include", "exclude"], str, Optional[str]] | None
    WindowSize = int | None
    CustomNodataValue = float | int | None
    Threads = Literal["cpu"] | int | None
    Cache = float | None
    CalculationDtype = str
    CustomOutputDtype = str | None
    CreateNameAttribute: Tuple[str, str] | None

    @staticmethod
    def _validate(
        *,
        input_images=_UNSET,
        output_images=_UNSET,
        save_as_cog=_UNSET,
        debug_logs=_UNSET,
        vector_mask=_UNSET,
        window_size=_UNSET,
        custom_nodata_value=_UNSET,
        calculation_dtype=_UNSET,
        custom_output_dtype=_UNSET,
        create_name_attribute=_UNSET,
        output_dtype=_UNSET,
        cache=_UNSET,
        image_threads=_UNSET,
        io_threads=_UNSET,
        tile_threads=_UNSET,
        estimate_stats=_UNSET,
    ):
        if input_images is not _UNSET:
            if not isinstance(input_images, (str, list)):
                raise ValueError(
                    "input_images must be a string (path or glob pattern) or a list of strings."
                )
            if isinstance(input_images, list) and not all(
                isinstance(p, str) for p in input_images
            ):
                raise ValueError("All elements in input_images list must be strings.")

        if output_images is not _UNSET:
            if not isinstance(output_images, (str, list)):
                raise ValueError(
                    "output_images must be a string (path or template) or a list of strings."
                )
            if isinstance(output_images, list) and not all(
                isinstance(p, str) for p in output_images
            ):
                raise ValueError("All elements in output_images list must be strings.")

        if save_as_cog is not _UNSET:
            if not isinstance(save_as_cog, bool):
                raise ValueError("save_as_cog must be a boolean.")

            if save_as_cog:
                if window_size is _UNSET or window_size is None:
                    raise ValueError("When save_as_cog=True, window_size must be set.")
                if window_size % 16 != 0:
                    raise ValueError("When save_as_cog=True, window_size must be a multiple of 16.")

        if debug_logs is not _UNSET:
            if not isinstance(debug_logs, bool):
                raise ValueError("debug_logs must be a boolean.")

        if vector_mask is not _UNSET and vector_mask is not None:
            if not isinstance(vector_mask, tuple) or len(vector_mask) not in {2, 3}:
                raise ValueError("vector_mask must be a tuple of 2 or 3 elements.")
            if vector_mask[0] not in {"include", "exclude"}:
                raise ValueError(
                    "The first element of vector_mask must be 'include' or 'exclude'."
                )
            if not isinstance(vector_mask[1], str):
                raise ValueError(
                    "The second element must be a string (vector file path)."
                )
            if len(vector_mask) == 3 and not isinstance(vector_mask[2], str):
                raise ValueError(
                    "The third element, if provided, must be a string (field name)."
                )

        if window_size is not _UNSET:
            if window_size is None:
                pass
            elif isinstance(window_size, int):
                if window_size <= 0:
                    raise ValueError("window_size must be > 0.")
            else:
                raise ValueError("window_size must be an int or None.")

        if custom_nodata_value is not _UNSET:
            if custom_nodata_value is not None and not isinstance(
                custom_nodata_value, (int, float)
            ):
                raise ValueError("custom_nodata_value must be a number or None.")

        if calculation_dtype is not _UNSET:
            if not isinstance(calculation_dtype, str):
                raise ValueError("calculation_dtype must be a string.")

        if custom_output_dtype is not _UNSET and custom_output_dtype is not None:
            if not isinstance(custom_output_dtype, str):
                raise ValueError("custom_output_dtype must be a string or None.")

        if create_name_attribute is not _UNSET:
            if (
                not isinstance(Universal.CreateNameAttribute, tuple)
                or len(Universal.CreateNameAttribute) != 2
            ):
                raise ValueError(
                    "CreateNameAttribute must be a tuple of two strings or None."
                )
            if not all(isinstance(s, str) for s in Universal.CreateNameAttribute):
                raise ValueError(
                    "Both elements of CreateNameAttribute must be strings."
                )
        if output_dtype is not _UNSET and output_dtype is not None:
            if not isinstance(output_dtype, str):
                raise ValueError("output_dtype must be a string or None.")

        if cache is not _UNSET:
            if cache is None:
                return
            if isinstance(cache, (int, float)) and not isinstance(cache, bool):
                if cache <= 0:
                    raise ValueError("cache must be > 0 (in GB).")
                return
            raise ValueError("cache must be a number in GB, or None.")

        if image_threads is not _UNSET:
            _validate_threads(image_threads, "image_threads")

        if io_threads is not _UNSET:
            _validate_threads(io_threads, "io_threads")

        if tile_threads is not _UNSET:
            _validate_threads(tile_threads, "tile_threads")

        if estimate_stats is not _UNSET:
            if not isinstance(estimate_stats, bool):
                raise ValueError("estimate_stats must be a boolean.")

def _validate_threads(x, name):
    if x is None:
        return
    if isinstance(x, int):
        if x < 1:
            raise ValueError(f"{name} must be a positive integer, got {x}.")
        return
    if isinstance(x, str) and x == "cpu":
        return
    raise ValueError(f'{name} must be "cpu", an int, or None.')


# Match-specific only
class Match:
    SpecifyModelImages = Tuple[Literal["exclude", "include"], List[str]] | None

    @staticmethod
    def _validate_match(
        *,
        specify_model_images=_UNSET,
    ):
        if specify_model_images is not _UNSET and specify_model_images is not None:
            if (
                not isinstance(specify_model_images, tuple)
                or len(specify_model_images) != 2
                or specify_model_images[0] not in {"include", "exclude"}
                or not isinstance(specify_model_images[1], list)
                or not all(isinstance(s, str) for s in specify_model_images[1])
            ):
                raise ValueError(
                    "specify_model_images must be a tuple of ('include'|'exclude', list of strings)."
                )

    @staticmethod
    def _validate_global_regression(
        *,
        custom_mean_factor=_UNSET,
        custom_std_factor=_UNSET,
        save_adjustments=_UNSET,
        load_adjustments=_UNSET,
        pif_method=_UNSET,
        pif_feature_method=_UNSET,
        pif_save_inz=_UNSET,
    ):
        if pif_method is not _UNSET:
            if pif_method not in {"entire", "flood_from_match_points"}:
                raise ValueError("pif_method must be 'entire' or 'flood_from_match_points'.")
        if pif_feature_method is not _UNSET:
            if pif_feature_method not in {"orb"}:
                raise ValueError("pif_feature_method must be 'orb'.")

        if custom_mean_factor is not _UNSET:
            if not isinstance(custom_mean_factor, (int, float)):
                raise ValueError("custom_mean_factor must be a number.")

        if custom_std_factor is not _UNSET:
            if not isinstance(custom_std_factor, (int, float)):
                raise ValueError("custom_std_factor must be a number.")

        if save_adjustments is not _UNSET and save_adjustments is not None:
            if not isinstance(save_adjustments, str):
                raise ValueError("save_adjustments must be a string or None.")

        if load_adjustments is not _UNSET and load_adjustments is not None:
            if not isinstance(load_adjustments, str):
                raise ValueError("load_adjustments must be a string or None.")

        if pif_save_inz is not _UNSET and pif_save_inz is not None:
            if not isinstance(pif_save_inz, str):
                raise ValueError("pif_save_inz must be a string or None.")
            placeholder_count = pif_save_inz.count("$")
            if placeholder_count not in {0, 2}:
                raise ValueError(
                    "pif_save_inz must contain either no '$' placeholders or exactly two '$' placeholders."
                )

    @staticmethod
    def _validate_local_block_adjustment(
        *,
        number_of_blocks=_UNSET,
        alpha=_UNSET,
        correction_method=_UNSET,
        save_block_maps=_UNSET,
        load_block_maps=_UNSET,
        override_bounds_canvas_coords=_UNSET,
    ):
        if number_of_blocks is not _UNSET:
            if not (
                isinstance(number_of_blocks, int)
                or (
                    isinstance(number_of_blocks, tuple)
                    and len(number_of_blocks) == 2
                    and all(isinstance(i, int) for i in number_of_blocks)
                )
                or number_of_blocks == "coefficient_of_variation"
            ):
                raise ValueError(
                    "number_of_blocks must be an int, a (width, height) tuple, or 'coefficient_of_variation'."
                )

        if alpha is not _UNSET:
            if not isinstance(alpha, (float, int)):
                raise ValueError("alpha must be a float or int.")

        if correction_method is not _UNSET:
            if correction_method not in {"gamma", "linear", "offset"}:
                raise ValueError(
                    "correction_method must be either 'gamma' or 'linear'."
                )

        if save_block_maps is not _UNSET:
            if save_block_maps is not None:
                if not (
                    isinstance(save_block_maps, tuple)
                    and len(save_block_maps) == 2
                    and all(isinstance(s, str) for s in save_block_maps)
                ):
                    raise ValueError(
                        "save_block_maps must be a tuple of two strings or None."
                    )

        if load_block_maps is not _UNSET:
            if load_block_maps is not None:
                if not (
                    isinstance(load_block_maps, tuple)
                    and len(load_block_maps) == 2
                    and (
                        (
                            isinstance(load_block_maps[0], str)
                            or load_block_maps[0] is None
                        )
                        and (
                            isinstance(load_block_maps[1], list)
                            or load_block_maps[1] is None
                        )
                    )
                ):
                    raise ValueError(
                        "load_block_maps must be a tuple (str|None, list[str]|None) or None."
                    )

        if override_bounds_canvas_coords is not _UNSET:
            if override_bounds_canvas_coords is not None:
                if not (
                    isinstance(override_bounds_canvas_coords, tuple)
                    and len(override_bounds_canvas_coords) == 4
                    and all(
                        isinstance(v, (float, int))
                        for v in override_bounds_canvas_coords
                    )
                ):
                    raise ValueError(
                        "override_bounds_canvas_coords must be a tuple of four floats or ints, or None."
                    )


class Pipeline:
    @staticmethod
    def _validate_shared_pipeline(
        *,
        shared_output_image_path=_UNSET,
        shared_temp_dir=_UNSET,
        delete_temp_dir=_UNSET,
    ):
        if shared_output_image_path is not _UNSET:
            if not isinstance(shared_output_image_path, str):
                raise ValueError("shared_output_image_path must be a string.")
        if shared_temp_dir is not _UNSET and shared_temp_dir is not None:
            if not isinstance(shared_temp_dir, str):
                raise ValueError("shared_temp_dir must be a string or None.")
        if delete_temp_dir is not _UNSET:
            if not isinstance(delete_temp_dir, bool):
                raise ValueError("delete_temp_dir must be a boolean.")

    @staticmethod
    def _validate_method_choice(
        *,
        method_name: str,
        method_value: str | None,
        allowed_values: set[str | None],
    ):
        if method_value not in allowed_values:
            raise ValueError(
                f"{method_name} must be one of {sorted(allowed_values, key=str)}."
            )


class Utils:
    @staticmethod
    def _validate_align_rasters(
        *,
        resampling_method=_UNSET,
        tap=_UNSET,
        resolution=_UNSET,
    ):
        if resampling_method is not _UNSET:
            if resampling_method not in {"nearest", "bilinear", "cubic"}:
                raise ValueError(
                    "resampling_method must be one of 'nearest', 'bilinear', or 'cubic'."
                )
        if tap is not _UNSET:
            if not isinstance(tap, bool):
                raise ValueError("tap must be a boolean.")
        if resolution is not _UNSET:
            if resolution not in {"highest", "average", "lowest"}:
                raise ValueError(
                    "resolution must be one of 'highest', 'average', or 'lowest'."
                )

    @staticmethod
    def _validate_mask_rasters(
        *,
        include_touched_pixels=_UNSET,
    ):
        if include_touched_pixels is not _UNSET:
            if not isinstance(include_touched_pixels, bool):
                raise ValueError("include_touched_pixels must be a boolean.")

    @staticmethod
    def _validate_merge_rasters(
        *,
        resolution=_UNSET,
    ):
        if resolution is not _UNSET:
            if resolution not in {"highest", "average", "lowest"}:
                raise ValueError(
                    "resolution must be one of 'highest', 'average', or 'lowest'."
                )


class Seamline:
    @staticmethod
    def _validate_voronoi_center_seamline(
        *,
        output_mask=_UNSET,
        aoi_path=_UNSET,
        vector_mask=_UNSET,
        image_field_name=_UNSET,
        min_point_spacing=_UNSET,
        min_cut_length=_UNSET,
        debug_vectors_path=_UNSET,
    ):
        if output_mask is not _UNSET:
            if not isinstance(output_mask, str):
                raise ValueError("output_mask must be a string.")
        if aoi_path is not _UNSET and aoi_path is not None:
            if not isinstance(aoi_path, str):
                raise ValueError("aoi_path must be a string or None.")
        if vector_mask is not _UNSET and vector_mask is not None:
            if (
                not isinstance(vector_mask, tuple)
                or len(vector_mask) != 2
                or not all(isinstance(value, str) for value in vector_mask)
            ):
                raise ValueError(
                    "vector_mask must be a tuple of (vector_path, field_name) or None."
                )
        if image_field_name is not _UNSET:
            if not isinstance(image_field_name, str):
                raise ValueError("image_field_name must be a string.")
        if min_point_spacing is not _UNSET:
            if not isinstance(min_point_spacing, (int, float)):
                raise ValueError("min_point_spacing must be a number.")
            if min_point_spacing <= 0:
                raise ValueError("min_point_spacing must be > 0.")
        if min_cut_length is not _UNSET:
            if not isinstance(min_cut_length, (int, float)):
                raise ValueError("min_cut_length must be a number.")
        if debug_vectors_path is not _UNSET and debug_vectors_path is not None:
            if not isinstance(debug_vectors_path, str):
                raise ValueError("debug_vectors_path must be a string or None.")

    @staticmethod
    def _validate_weighted_seamline(
        *,
        input_polygons=_UNSET,
        output_mask=_UNSET,
        rank_function=_UNSET,
        image_field_name=_UNSET,
        input_layer=_UNSET,
        output_layer=_UNSET,
        rank_descending=_UNSET,
    ):
        if input_polygons is not _UNSET:
            if not isinstance(input_polygons, str):
                raise ValueError("input_polygons must be a string.")
        if output_mask is not _UNSET:
            if not isinstance(output_mask, str):
                raise ValueError("output_mask must be a string.")
        if rank_function is not _UNSET:
            if not isinstance(rank_function, str) or not rank_function.strip():
                raise ValueError("rank_function must be a non-empty string.")
        if image_field_name is not _UNSET:
            if not isinstance(image_field_name, str) or not image_field_name.strip():
                raise ValueError("image_field_name must be a non-empty string.")
        if input_layer is not _UNSET and input_layer is not None:
            if not isinstance(input_layer, str) or not input_layer.strip():
                raise ValueError("input_layer must be a non-empty string or None.")
        if output_layer is not _UNSET:
            if not isinstance(output_layer, str) or not output_layer.strip():
                raise ValueError("output_layer must be a non-empty string.")
        if rank_descending is not _UNSET:
            if not isinstance(rank_descending, bool):
                raise ValueError("rank_descending must be a bool.")
