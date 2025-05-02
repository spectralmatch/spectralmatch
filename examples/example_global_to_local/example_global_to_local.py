import os

from spectralmatch import global_regression, local_block_adjustment
from spectralmatch import merge_rasters

# -------------------- Parameters
working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_data")
# This script is setup to perform matching on all tif files from a folder within the working directory called "input" e.g. working_directory/input/*.tif.

vector_mask_path = working_directory + "/Input/Masks.gpkg"

input_folder = os.path.join(working_directory, "Input")
global_folder = os.path.join(working_directory, "Output/GlobalMatch")
local_folder = os.path.join(working_directory, "Output/LocalMatch")

# -------------------- Global histogram matching
input_image_paths_array = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")]

matched_global_images_paths = global_regression(
    input_image_paths_array,
    global_folder,
    custom_mean_factor = 3, # Defualt 1; 3 often works better to 'move' the spectral mean of images closer together
    custom_std_factor = 1,
    # vector_mask_path=vector_mask_path,
    debug_mode=False,
    tile_width_and_height_tuple=(512, 512),
    parallel=True,
    custom_nodata_value=-9999,
    )

merge_rasters(
    matched_global_images_paths, # Rasters are layered with the last ones on top
    os.path.join(working_directory, "Output/GlobalMatch/MatchedGlobalImages.tif"),
    tile_width_and_height_tuple=(512, 512),
    )

# -------------------- Local histogram matching
global_image_paths_array = [os.path.join(f"{global_folder}/Images", f) for f in os.listdir(f"{global_folder}/Images") if f.lower().endswith(".tif")]

matched_local_images_paths = local_block_adjustment(
    global_image_paths_array,
    local_folder,
    target_blocks_per_image=100,
    projection="EPSG:6635",
    debug_mode=False,
    tile_width_and_height_tuple=(512, 512),
    parallel=True,
    custom_nodata_value=-9999,
    )

merge_rasters(
    matched_local_images_paths, # Rasters are layered with the last ones on top
    os.path.join(working_directory, "Output/LocalMatch/MatchedLocalImages.tif"),
    tile_width_and_height_tuple=(512, 512),
    )

print("Done with global and local histogram matching")