PYTHON ?= python
REPO_ROOT := $(abspath ../..)
SPECTRALMATCH ?= PYTHONPATH="$(REPO_ROOT)" $(PYTHON) -m spectralmatch

WORLDVIEW_DIR := $(CURDIR)/data_worldview
WORLDVIEW_INPUT := $(WORLDVIEW_DIR)/Input
WORLDVIEW_OUTPUT := $(WORLDVIEW_DIR)/PipelineOutput/MergedImage.tif
WORLDVIEW_TEMP := $(WORLDVIEW_DIR)/PipelineTemp

.PHONY: worldview-pipeline clean-worldview-pipeline

worldview-pipeline:
	mkdir -p "$(dir $(WORLDVIEW_OUTPUT))"
	$(SPECTRALMATCH) pipeline "$(WORLDVIEW_INPUT)" "$(WORLDVIEW_OUTPUT)" \
		--shared_temp_dir "$(WORLDVIEW_TEMP)" \
		--delete_temp_dir False \
		--shared_debug_logs True \
		--shared_window_size 1024 \
		--shared_image_threads 3 \
		--shared_io_threads 3 \
		--shared_tile_threads 3

clean-worldview-pipeline:
	rm -rf "$(WORLDVIEW_DIR)/PipelineOutput" "$(WORLDVIEW_TEMP)"
