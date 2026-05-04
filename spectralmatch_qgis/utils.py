import os
import platform
import sys
from pathlib import Path
import importlib
from qgis.core import Qgis, QgsMessageLog
import json
import ast
from qgis.core import (
    QgsProcessingParameterFile,
    QgsProcessingParameterString,
    QgsApplication
)


def log(message):
    QgsMessageLog.logMessage(message, "spectralmatch", level=Qgis.MessageLevel.Info)

def get_interpreter():
    return find_python_interpreter()


def find_python_interpreter():
    if (Path(sys.prefix) / "conda-meta").exists():  # Conda
        log("Attempt Conda install at 'python' shortcut")
        return "python"

    if platform.system() == "Windows":  # Windows
        base_path = Path(sys.prefix)
        for file in ["python.exe", "python3.exe"]:
            path = base_path / file
            if path.is_file():
                log(f"Attempt Windows install at {str(path)}")
                return str(path)
        path = sys.executable
        log(f"Attempt Windows install at {str(path)}")
        return path

    if platform.system() == "Darwin":  # Mac
        base_paths = [
            Path(sys.prefix),
            Path(sys.prefix) / "bin",
            Path(sys.executable).parent,
        ]
        for base_path in base_paths:
            for file in ["python", "python3"]:
                path = base_path / file
                if path.is_file():
                    log(f"Attempt MacOS install at {str(path)}")
                    return str(path)
        path = sys.executable
        log(f"Attempt MacOS install at {str(path)}")
        return path

    else:  # Fallback attempt
        path = sys.executable
        log(f"Attempt fallback install at {str(path)}")
        return path

def get_python_dependency_folder():
    python_dependencies = os.path.join(
        QgsApplication.qgisSettingsDirPath().replace("/", os.path.sep),
        "python",
        "dependencies",
    )
    log(f"Python dependencies folder: {python_dependencies}")
    return python_dependencies

def _add_folder_select_param(algorithm_instance, name: str, display_name: str):
    algorithm_instance.addParameter(
        QgsProcessingParameterFile(
            name,
            algorithm_instance.tr(display_name),
            behavior=QgsProcessingParameterFile.Folder
        )
    )


def _add_string_param(algorithm_instance, name: str, display_name: str, default):
    default_value = "None" if default in (None, 'None') else default.strip("'\"")
    algorithm_instance.addParameter(
        QgsProcessingParameterString(
            name,
            algorithm_instance.tr(display_name),
            defaultValue=default_value
        )
    )


def load_function_headers(target_function: str = None):
    json_path = os.path.join(os.path.dirname(__file__), "function_headers.json")
    with open(json_path, "r", encoding="utf-8") as f:
        all_funcs = json.load(f)
    if target_function:
        for entry in all_funcs:
            if entry["function"] == target_function:
                return entry
        raise ValueError(f"Function {target_function} not found in function_headers.json")
    return all_funcs


def normalize_cli_value(value: str) -> str:
    """Converts QGIS string parameter to safe CLI string representation."""
    stripped = value.strip()

    # Handle empty or null
    if stripped.lower() in {"none", "", "null"}:
        return "None"

    try:
        parsed = ast.literal_eval(stripped)
        return repr(parsed)
    except (ValueError, SyntaxError):
        return stripped


def normalize_python_value(value: str):
    stripped = value.strip()

    if stripped.lower() in {"none", "", "null"}:
        return None

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped


def resolve_function(full_function_path: str):
    parts = full_function_path.split(".")
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"Could not resolve function path: {full_function_path}")
