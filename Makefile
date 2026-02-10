MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ENV_NAME = spectralmatch
BUILD_PLUGIN=python spectralmatch_qgis/build_plugin.py
QGISPLUGINNAME = spectralmatch_qgis

# Install
install:
	pip install $(MAKEFILE_DIR).

install-dev:
	pip install -e '$(MAKEFILE_DIR).[dev]'

install-docs:
	pip install -e '$(MAKEFILE_DIR).[docs]'

install-setup:
	bash -c "\
		conda create -y -n $(ENV_NAME) python>=3.10 gdal>=3.6 proj>=9.3 -c conda-forge && \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $(ENV_NAME) && \
		pip install . && \
		pip install -e '.[dev]' && \
		pip install -e '.[docs]' && \
		pre-commit install && \
		echo '✅ Setup complete. Environment \"$(ENV_NAME)\" is ready.' \
	"


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
	       $(MAKEFILE_DIR)spectralmatch_qgis/function_headers.json \
		   $(MAKEFILE_DIR)spectralmatch_qgis.zip \
		   $(MAKEFILE_DIR)docs/images \
		   $(MAKEFILE_DIR)spectralmatch_qgis/requirements.txt \
		   $(MAKEFILE_DIR)/spectralmatch_qgis/*.whl
	find $(MAKEFILE_DIR)docs/examples/data_landsat -mindepth 1 ! -path "*/Input*" -exec rm -rf {} +
	find $(MAKEFILE_DIR)docs/examples/data_worldview -mindepth 1 ! -path "*/Input*" -exec rm -rf {} +

# Python
python-build:
	@echo "Building Python wheel..."
	python -m build --wheel

# QGIS
qgis-build: python-build
	@cp dist/*.whl spectralmatch_qgis
	PYTHONPATH=. $(BUILD_PLUGIN)
	@echo "Removing __pycache__..."
	rm -rf spectralmatch_qgis/__pycache__ spectralmatch_qgis/test/__pycache__
	@echo "Creating plugin zip..."
	zip -r spectralmatch_qgis.zip spectralmatch_qgis/ \
	  -x "*.DS_Store" "*__MACOSX*"

qgis-deploy:
	python spectralmatch_qgis/plugin_upload.py spectralmatch_qgis.zip \
		--username your_username --password your_password