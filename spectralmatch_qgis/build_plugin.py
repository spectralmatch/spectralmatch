import inspect
import json
import importlib
import os
import shutil
import tomllib
from types import FunctionType
from typing import List


def _format_annotation(annotation):
    if annotation is inspect.Signature.empty:
        return None
    ann = repr(annotation)
    if ann.startswith("typing."):
        return ann.replace("typing.", "")
    if ann.startswith("<class '") and ann.endswith("'>"):
        return ann[8:-2]
    return ann


def _build_params(obj):
    params = []
    for param in inspect.signature(obj).parameters.values():
        params.append(
            {
                "name": param.name,
                "display_name": param.name.replace("_", " ").capitalize(),
                "kind": str(param.kind),
                "default": repr(param.default) if param.default is not param.empty else None,
                "annotation": _format_annotation(param.annotation),
                "param_type": "folder"
                if param.name in {"input_images", "output_images"}
                else "string",
            }
        )
    return params


def _is_public(name: str, exclude_functions: set[str], exclude_internal_functions: bool) -> bool:
    if name in exclude_functions:
        return False
    if exclude_internal_functions and name.startswith("_"):
        return False
    return True


def _append_function_header(output, function_path, obj):
    output.append(
        {
            "function": function_path,
            "docstring": inspect.getdoc(obj) or "",
            "parameters": _build_params(obj),
        }
    )


def _iter_public_class_functions(cls, module_name, exclude_functions, exclude_internal_functions):
    for method_name, descriptor in cls.__dict__.items():
        if not _is_public(method_name, exclude_functions, exclude_internal_functions):
            continue
        if not isinstance(descriptor, (staticmethod, classmethod)):
            continue
        method = getattr(cls, method_name)
        if getattr(method, "__module__", None) != module_name:
            continue
        yield method_name, method


def generate_function_headers(
    package_name="spectralmatch",
    output_file="spectralmatch_qgis/function_headers.json",
    exclude_functions: List[str] = None,
    exclude_modules: List[str] = ["spectralmatch.handlers"],
    exclude_internal_functions: bool = True
):
    exclude_functions = set(exclude_functions or [])
    exclude_modules = set(exclude_modules or [])
    output = []

    def walk_module(module, prefix):
        if any(prefix.startswith(excl) for excl in exclude_modules):
            return

        for name in dir(module):
            if not _is_public(name, exclude_functions, exclude_internal_functions):
                continue
            try:
                obj = getattr(module, name)
            except Exception:
                continue
            if inspect.ismodule(obj) and obj.__package__ and obj.__package__.startswith(package_name):
                walk_module(obj, f"{prefix}.{name}")
            elif isinstance(obj, FunctionType) and obj.__module__ == module.__name__:
                if "." in obj.__qualname__:
                    continue
                _append_function_header(output, f"{prefix}.{name}", obj)
            elif inspect.isclass(obj) and obj.__module__ == module.__name__:
                for method_name, method in _iter_public_class_functions(
                    obj,
                    module.__name__,
                    exclude_functions,
                    exclude_internal_functions,
                ):
                    _append_function_header(output, f"{prefix}.{name}.{method_name}", method)

    pkg = importlib.import_module(package_name)
    walk_module(pkg, package_name)
    if not output: raise RuntimeError(f"No function headers found in package '{package_name}'. Check exclusions, installation, or package contents.")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output_file


def generate_requirements_txt(
    input_toml_path="pyproject.toml",
    output_txt_path="spectralmatch_qgis/requirements.txt",
):
    with open(input_toml_path, "rb") as f:
        pyproject = tomllib.load(f)

    project = pyproject["project"]
    deps = list(project.get("dependencies", []))

    with open(output_txt_path, "w") as f:
        for dep in deps:
            f.write(dep + "\n")


def copy_spectralmatch_package(
    source_dir="spectralmatch",
    target_dir="spectralmatch_qgis/spectralmatch",
):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(
        source_dir,
        target_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )

if __name__ == "__main__":
    copy_spectralmatch_package()
    generate_function_headers()
    generate_requirements_txt()
