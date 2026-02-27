# %% Lirrn normalization mosaic


# %% Setup
import os
from spectralmatch import *

# Important: If this does not automatically find the correct CWD, manually copy the path to the /data_worldview folder
working_directory = os.path.join(os.getcwd(), "data_worldview")
print(working_directory)

input_folder = os.path.join(working_directory, "Input")
lirrn_folder = os.path.join(working_directory, "LirrnMatch")


window_size = 128
image_threads = "cpu"
io_threads = "cpu"
tile_threads = "cpu"
debug_mode = True

# %% Lirrn normalization
#
lirrn_normalization(
    input_images=input_folder,
    output_images=lirrn_folder,
    window_size=window_size,
    image_threads=image_threads,
    io_threads=io_threads,
    tile_threads = tile_threads,
    debug_logs=debug_mode,
    )