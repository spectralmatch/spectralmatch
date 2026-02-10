# Installation Methods

---

## Installation as a QGIS Plugin
### 1. Get QGIS
[Download](https://qgis.org/download/) and install QGIS.
> This plugin requires Python ≥ 3.10 and ≤ 3.12. QGIS ships with different versions of Python, to check, in the QGIS menu, go to QGIS>About gis. If your version of Python is not supported, you can update your QGIS (if available) or install it containerized with conda: `conda create --name qgis_env python=3.12`, `conda activate qgis_env`, `conda install -c conda-forge qgis`, then `qgis` to start the program.

### 2. Install spactalmatch QGIS plugin
- Go to Plugins → Manage and Install Plugins…
- Find spectralmatch in the list, install, and enable it
- Find the plugin in the Processing Toolbox

### 3. Install spectralmatch Python library
The plugin will attempt to automatically install all Python dependencies that it requires in the QGIS Python interpreter. It uses [QPIP](https://github.com/opengisch/qpip), in addition to custom installation scripts, to do this. If it is unable to, the user must manually locate the QGIS python interpreter and install the spectralmatch python library and all of its dependencies.

---

## Installation via pip as a Python Library and CLI

### 1. System requirements
Before installing, ensure you have the following system-level prerequisites:

- Python ≥ 3.10 and ≤ 3.12
- PROJ ≥ 9.3
- GDAL ≥ 3.10.2
- pip

An easy way to install these dependancies is to use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions):
```bash
conda create -n spectralmatch python=3.12 "gdal>=3.10.2" "proj>=9.3" -c conda-forge
conda activate spectralmatch
```

### 2. Install spectralmatch

You can automatically install the library via [PyPI](https://pypi.org/). (this method installs only the core code as a library):

```bash
pip install spectralmatch
```

---

## Installation via pixi as a Python Library and CLI

Installing via pixi can be easier as it handles the system level dependencies:

```bash
pixi init myproject
cd myproject
pixi add "python>=3.9,<3.13" "gdal>=3.10.2"
pixi add spectralmatch --pypi
```

## Installation from Source

### 1. Clone the Repository
```bash
git clone https://github.com/spectralmatch/spectralmatch.git
cd spectralmatch
```

> Assuming you have Make installed, you can then run `make install-setup` to automatically complete the remaining setup steps.

### 2. System requirements
Before installing, ensure you have the following system-level prerequisites:

- Python ≥ 3.10 and ≤ 3.12
- PROJ ≥ 9.3
- GDAL = 3.10.2

An easy way to install these dependancies is to use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions):
```bash
conda create -n spectralmatch python=3.12 "gdal>=3.10.2" "proj>=9.3" -c conda-forge
conda activate spectralmatch
```

### 3. Install Dependancies
The `pyproject.toml` defines **core** dependancies to run the library and optional **dev**, and **docs** dependancies.

```bash
pip install . # Normal dependencies
pip install -e ".[dev]"   # Developer dependencies
pip install -e ".[docs]"  # Documentation dependencies
pip install -e ".[qgis-build]" # Build qgis plugin
```
