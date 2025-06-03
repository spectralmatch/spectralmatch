# %% Setup
import os

from spectralmatch import *

# Important: If this does not automatically find the correct CWD, manually copy the path to the /data_worldview folder
working_directory = '/Users/kanoalindiwe/Downloads/Projects/spectralmatch/docs/examples/data_worldview'
print(working_directory)

input_folder = os.path.join(working_directory, "Input")
global_folder = os.path.join(working_directory, "GlobalMatch")
local_folder = os.path.join(working_directory, "LocalMatch")
aligned_folder = os.path.join(working_directory, "Aligned")
clipped_folder = os.path.join(working_directory, "Clipped")
stats_folder = os.path.join(working_directory, "Stats")
new_global_folder = os.path.join(working_directory, "GlobalMatch_New")

window_size = 128
num_workers = 5

# %% (OPTIONAL) Global match with increased performance using image and window process or threading with windowed calculations

# global_regression(
#     (input_folder, "*.tif"),
#     (global_folder, "$_Global.tif"),
#     debug_logs=True,
#     window_size=window_size,
#     image_parallel_workers=("process", num_workers),
#     window_parallel_workers=("process", num_workers),
#     )
#
# # %% Local matching
# local_block_adjustment(
#     (global_folder, "*.tif"),
#     (local_folder, "$_Local.tif"),
#     number_of_blocks=100,
#     debug_logs=True,
#     window_size=window_size,
#     image_parallel_workers=("process", num_workers),
#     window_parallel_workers=("process", num_workers),
#     )


# %% Align

# align_rasters(
#     (input_folder, "*.tif"),
#     (aligned_folder, "$_Aligned.tif"),
#     tap=True,
#     resolution='highest',
#     debug_logs=True,
#     window_size=window_size,
#     image_parallel_workers=("process", num_workers),
#     window_parallel_workers=("process", num_workers),
#     )


# mask_rasters(
#     (aligned_folder, "*.tif"),
#     (clipped_folder, "$_Clipped.tif"),
#     vector_mask=("include", os.path.join(working_directory, "ImageMasks.gpkg"), "image"),
#     debug_logs=True,
#     window_size=window_size,
#     image_parallel_workers=("process", num_workers),
#     window_parallel_workers=("process", num_workers),
#     )


output_merged_image_path = os.path.join(working_directory, "MergedImage.tif")

merge_rasters(
    (clipped_folder, "*.tif"),
    output_merged_image_path,
    window_size=window_size,
    debug_logs=True,
    image_parallel_workers=("process", num_workers),
)
