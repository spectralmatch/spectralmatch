MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ENV_NAME = spectralmatch
BUILD_PLUGIN=python spectralmatch_qgis/build_plugin.py
QGISPLUGINNAME = spectralmatch_qgis
DASK_SCHEDULER_FILE ?= /tmp/spectralmatch-dask-1.json
DASK_LOCAL_DIRECTORY ?= /tmp/spectralmatch-dask-worker-1
DASK_NWORKERS ?= 4
DASK_NTHREADS ?= 1
DASK_MEMORY_LIMIT ?= 4GiB

.PHONY: start-dask

# Install
install:
	pip install $(MAKEFILE_DIR).

install-dev:
	pip install -e '$(MAKEFILE_DIR).[dev]'

install-docs:
	pip install -e '$(MAKEFILE_DIR).[docs]'

install-setup:
	bash -c "\
		conda create -y -n $(ENV_NAME) python>=3.10 gdal>=3.11 proj>=9.3 -c conda-forge && \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $(ENV_NAME) && \
		pip install . && \
		pip install -e '.[dev]' && \
		pip install -e '.[docs]' && \
		pre-commit install && \
		echo '✅ Setup complete. Environment \"$(ENV_NAME)\" is ready.' \
	"


# Local Dask cluster; example: make start-dask DASK_NWORKERS=8 DASK_MEMORY_LIMIT=2GiB
start-dask:
	@bash -c 'set -eu; \
		if ! command -v dask >/dev/null 2>&1; then \
			echo "Dask is not installed. Run: pip install -e '\''.[dask]'\''" >&2; exit 1; \
		fi; \
		scheduler_file="$(DASK_SCHEDULER_FILE)"; \
		local_directory="$(DASK_LOCAL_DIRECTORY)"; \
		rm -f "$$scheduler_file"; \
		mkdir -p "$$local_directory"; \
		cleanup() { \
			trap - EXIT INT TERM; \
			if [ -n "$${worker_pid:-}" ]; then kill "$$worker_pid" 2>/dev/null || true; fi; \
			if [ -n "$${scheduler_pid:-}" ]; then kill "$$scheduler_pid" 2>/dev/null || true; fi; \
			wait $${worker_pid:-} $${scheduler_pid:-} 2>/dev/null || true; \
			rm -f "$$scheduler_file"; \
		}; \
		trap cleanup EXIT INT TERM; \
		echo "Starting Dask scheduler (dashboard: http://localhost:8787/status)"; \
		dask scheduler --scheduler-file "$$scheduler_file" & scheduler_pid=$$!; \
		attempt=0; \
		while [ ! -s "$$scheduler_file" ]; do \
			if ! kill -0 "$$scheduler_pid" 2>/dev/null; then \
				echo "Dask scheduler failed to start" >&2; exit 1; \
			fi; \
			attempt=$$((attempt + 1)); \
			if [ "$$attempt" -ge 100 ]; then echo "Timed out waiting for Dask scheduler" >&2; exit 1; fi; \
			sleep 0.1; \
		done; \
		echo "Starting $(DASK_NWORKERS) workers with $(DASK_NTHREADS) thread(s) and $(DASK_MEMORY_LIMIT) memory each"; \
		echo "Connect with dask_scheduler=(\"file\", \"$$scheduler_file\")"; \
		dask worker --scheduler-file "$$scheduler_file" \
			--nworkers "$(DASK_NWORKERS)" \
			--nthreads "$(DASK_NTHREADS)" \
			--memory-limit "$(DASK_MEMORY_LIMIT)" \
			--local-directory "$$local_directory" & worker_pid=$$!; \
		wait "$$worker_pid"'


# Docs
docs-serve:
	mkdir -p $(MAKEFILE_DIR)docs/images
	cp -r $(MAKEFILE_DIR)images/* $(MAKEFILE_DIR)docs/images/
	mkdocs serve -a localhost:8001

docs-build:
	mkdir -p $(MAKEFILE_DIR)docs/images
	cp -r $(MAKEFILE_DIR)images/* $(MAKEFILE_DIR)docs/images/
	mkdocs build

docs-deploy:
	mkdir -p $(MAKEFILE_DIR)docs/images
	cp -r $(MAKEFILE_DIR)images/* $(MAKEFILE_DIR)docs/images/
	mkdocs gh-deploy


# Versions
tag:
	@if [ -z "$(version)" ]; then \
		echo "Usage: make tag version=1.2.3"; \
		exit 1; \
	fi
	git tag -a v$(version) -m "Version $(version)"
	git push origin v$(version)

version:
	@if [ -z "$(version)" ]; then \
		echo "Usage: make version version=1.2.3"; \
		exit 1; \
	fi
	@echo "Updating versions to $(version)..."
	sed -i.bak "s/^version = .*/version = \"$(version)\"/" pyproject.toml && rm pyproject.toml.bak
	sed -i.bak "s/^version=.*/version=$(version)/" spectralmatch_qgis/metadata.txt && rm spectralmatch_qgis/metadata.txt.bak
	git add pyproject.toml spectralmatch_qgis/metadata.txt
	git commit -m "Version $(version) released"
	git push origin HEAD
	$(MAKE) tag version=$(version)


# Code formatting
format:
	black $(MAKEFILE_DIR).

check-format:
	black --check $(MAKEFILE_DIR).

lint:
	ruff check $(MAKEFILE_DIR).


# Testing
test:
	pytest

test-file:
	pytest $(path)

# Cleanup
clean:
	rm -rf $(MAKEFILE_DIR)build \
	       $(MAKEFILE_DIR)dist \
	       $(MAKEFILE_DIR)*.egg-info \
	       $(MAKEFILE_DIR)__pycache__ \
	       $(MAKEFILE_DIR).pytest_cache \
	       $(MAKEFILE_DIR)site \
	       $(MAKEFILE_DIR)spectralmatch_qgis/help/build \
	       $(MAKEFILE_DIR)spectralmatch_qgis/spectralmatch \
	       $(MAKEFILE_DIR)spectralmatch_qgis/function_headers.json \
		   $(MAKEFILE_DIR)spectralmatch_qgis.zip \
		   $(MAKEFILE_DIR)docs/images \
		   $(MAKEFILE_DIR)spectralmatch_qgis/requirements.txt \
		   $(MAKEFILE_DIR)/spectralmatch_qgis/*.whl
	find $(MAKEFILE_DIR)docs/examples/data_landsat -mindepth 1 ! -path "*/Input*" -exec rm -rf {} +
	find $(MAKEFILE_DIR)docs/examples/data_worldview -mindepth 1 ! -path "*/Input*" -exec rm -rf {} +
	rm -rf $(MAKEFILE_DIR)docs/examples/data_worldview/PipelineOutput \
	       $(MAKEFILE_DIR)docs/examples/data_worldview/PipelineTemp

# Python
python-build:
	@echo "Building Python wheel..."
	python -m build --wheel

qgis-install-local-spectralmatch: python-build
	@if [ -z "$(interpreter)" ]; then \
		echo "Usage: make qgis-install-local-spectralmatch interpreter=/path/to/qgis/python"; \
		exit 1; \
	fi
	@echo "Installing local spectralmatch wheel into $(interpreter)..."
	"$(interpreter)" -m pip install --force-reinstall --no-deps dist/*.whl

# QGIS
qgis-build:
	PYTHONPATH=. $(BUILD_PLUGIN)
	@echo "Removing __pycache__..."
	find spectralmatch_qgis -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Creating plugin zip..."
	zip -r spectralmatch_qgis.zip spectralmatch_qgis/ \
	  -x "*.DS_Store" "*__MACOSX*"

qgis-deploy:
	python spectralmatch_qgis/plugin_upload.py spectralmatch_qgis.zip \
		--username your_username --password your_password
