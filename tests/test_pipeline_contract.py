import inspect
from typing import Literal, get_args, get_type_hints

import pytest

from spectralmatch import chain

from .utils_test import create_dummy_raster


STEPS = [
    ("joint_coregistration", "joint_coregistration", chain, "joint_coregistration"),
    ("align", "align_rasters", chain, "align_rasters"),
    ("global_regression", "global_regression", chain.Match, "global_regression"),
    ("local_block_adjustment", "local_block_adjustment", chain.Match, "local_block_adjustment"),
    ("voronoi_center_seamline", "voronoi_center_seamline", chain.Seamline, "voronoi"),
    ("weighted_seamline", "weighted_seamline", chain.Seamline, "weighted"),
    ("mask", "mask_rasters", chain, "mask_rasters"),
    ("merge", "merge_rasters", chain, "merge_rasters"),
]
PATH_PARAMETERS = {"input_images", "output_images", "output_image_path", "output_mask"}


@pytest.mark.parametrize("step,prefix,owner,name", STEPS)
def test_pipeline_parameter_contract(step, prefix, owner, name):
    target = getattr(owner, name)
    target_parameters = inspect.signature(target).parameters
    pipeline_parameters = inspect.signature(chain.pipeline).parameters
    target_types = get_type_hints(target)
    pipeline_types = get_type_hints(chain.pipeline)

    for parameter_name, parameter in target_parameters.items():
        if parameter_name in PATH_PARAMETERS:
            continue
        dedicated = f"{prefix}_{parameter_name}"
        pipeline_name = dedicated if dedicated in pipeline_parameters else (
            "shared_resume_from_steps" if parameter_name == "resume_from_outputs" else f"shared_{parameter_name}"
        )
        assert pipeline_name in pipeline_parameters, f"Missing pipeline parameter for {prefix}.{parameter_name}"
        mapped = pipeline_parameters[pipeline_name]
        mapped_type = pipeline_types[pipeline_name]
        if parameter.default is inspect.Parameter.empty:
            # Required inputs of optional steps cannot be required for every pipeline.
            assert step == "weighted_seamline"
            assert mapped.default is None
            assert mapped_type == target_types[parameter_name] | None
        elif parameter_name in {"cache", "image_threads", "io_threads", "tile_threads"}:
            assert set(get_args(mapped_type)) - {Literal["auto"]} == set(get_args(target_types[parameter_name]))
            assert mapped.default == "auto"
        elif parameter_name == "concurrent_processing_backend" and step == "merge":
            # Merge receives None in single-file mode and the shared backend in tile mode.
            assert mapped.default == "process_pool"
            assert mapped_type | None == target_types[parameter_name]
        else:
            assert mapped_type == target_types[parameter_name], pipeline_name
            assert mapped.default == (1024 if parameter_name == "window_size" else parameter.default), pipeline_name

    for pipeline_name in pipeline_parameters:
        if pipeline_name.startswith(prefix + "_"):
            assert pipeline_name[len(prefix) + 1:] in target_parameters, f"Unmapped pipeline parameter: {pipeline_name}"


@pytest.mark.parametrize("step,prefix,owner,name", STEPS)
def test_pipeline_forwards_every_function_parameter(tmp_path, monkeypatch, step, prefix, owner, name):
    target = getattr(owner, name)
    signature = inspect.signature(target)
    source = tmp_path / "input.tif"
    create_dummy_raster(source, count=1)
    output = str(tmp_path / ("output.gpkg" if "seamline" in step else "output"))
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
        return output if step == "merge" or "seamline" in step else [output + "/result.tif"]

    monkeypatch.setattr(owner, name, capture)
    chain.pipeline(**options)
    assert len(calls) == 1
