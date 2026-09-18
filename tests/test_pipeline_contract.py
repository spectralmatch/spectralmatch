import inspect

import pytest

from spectralmatch import chain

from .utils_test import create_dummy_raster


STEPS = [
    ("joint_coregistration", "joint_coregistration", chain, "joint_coregistration"),
    ("align", "align_rasters", chain, "align_rasters"),
    ("global_regression", "global_regression", chain.Match, "global_regression"),
    ("local_block_adjustment", "local_block_adjustment", chain.Match, "local_block_adjustment"),
    ("create_footprints", "create_footprints", chain.Seamline, "create_footprints"),
    ("postprocess_footprints", "postprocess_footprints", chain.Seamline, "postprocess_footprints"),
    ("voronoi_center_seamline", "voronoi_center_seamline", chain.Seamline, "voronoi"),
    ("weighted_seamline", "weighted_seamline", chain.Seamline, "weighted"),
    ("mask", "mask_rasters", chain, "mask_rasters"),
    ("merge", "merge_rasters", chain, "merge_rasters"),
]
PATH_PARAMETERS = {"input_images", "output_images", "output_image_path", "output_mask", "output_polygons"}


@pytest.mark.parametrize("step,prefix,owner,name", STEPS)
def test_pipeline_forwards_every_function_parameter(tmp_path, monkeypatch, step, prefix, owner, name):
    target = getattr(owner, name)
    signature = inspect.signature(target)
    source = tmp_path / "input.tif"
    create_dummy_raster(source, count=1)
    vector_step = step in chain.SEAMLINE_STEPS | chain.FOOTPRINT_STEPS
    output = str(tmp_path / ("output.gpkg" if vector_step else "output"))
    options = {
        "shared_input_images": [str(source)],
        "shared_output_image_path": output,
        "shared_temp_dir": str(tmp_path / "temp"),
        "steps": (step,),
        "shared_cache": None,
        "shared_image_threads": None,
        "shared_io_threads": 1,
        "shared_tile_threads": 1,
        "shared_window_size": 16,
        "shared_window_scales": (2, 4),
        "shared_concurrent_processing_backend": "dask",
        "shared_dask_scheduler": ("address", "tcp://localhost:8786"),
        "shared_resume_from_steps": "validate",
        "shared_custom_nodata_value": 255,
        "shared_output_dtype": "uint16",
        "shared_calculation_dtype": "float64",
        "shared_save_as_cog": True,
        "shared_debug_logs": True,
    }
    for parameter_name in inspect.signature(chain.pipeline).parameters:
        if parameter_name.startswith(prefix + "_") and step != "merge":
            # Unique values detect swapped or ignored arguments without executing the step.
            options[parameter_name] = object()
    if step == "merge":
        options.update(merge_rasters_output_tiles=True, merge_rasters_overlap=4, merge_rasters_create_vrts="custom.vrt")
    expected = inspect.signature(chain.pipeline).bind(**options)
    expected.apply_defaults()
    calls = []

    def capture(**kwargs):
        signature.bind(**kwargs)
        calls.append(kwargs)
        assert set(kwargs) == set(signature.parameters)
        for parameter_name, actual in kwargs.items():
            if parameter_name == "input_images":
                assert actual == options["shared_input_images"]
            elif parameter_name in PATH_PARAMETERS:
                assert actual == output
            else:
                dedicated = f"{prefix}_{parameter_name}"
                pipeline_name = dedicated if dedicated in expected.arguments else (
                    "shared_resume_from_steps" if parameter_name == "resume_from_outputs" else f"shared_{parameter_name}"
                )
                assert actual == expected.arguments[pipeline_name], pipeline_name
        return output if step == "merge" or vector_step else [output + "/result.tif"]

    monkeypatch.setattr(owner, name, capture)
    chain.pipeline(**options)
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["create_footprints", "postprocess_footprints"])
def test_pipeline_footprint_defaults_match_function(name):
    pipeline_parameters = inspect.signature(chain.pipeline).parameters
    for parameter_name, parameter in inspect.signature(getattr(chain.Seamline, name)).parameters.items():
        if parameter.default is inspect.Parameter.empty:
            continue
        dedicated = f"{name}_{parameter_name}"
        if dedicated in pipeline_parameters:
            assert pipeline_parameters[dedicated].default == parameter.default
