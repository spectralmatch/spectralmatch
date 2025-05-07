import os
import numpy as np
import cv2 as cv

from spectralmatch.seamline import load_images_and_masks, compute_graphcut_seams, apply_seams_to_images, apply_seams_to_tif

working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_data")
input_folder = os.path.join(working_directory, "Output", "LocalMatch", "Images")
input_image_paths_array = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".tif")]
output_paths = [os.path.splitext(p)[0] + "_seam.tif" for p in input_image_paths_array]

images, masks, corners = load_images_and_masks(input_image_paths_array)
seam_masks = compute_graphcut_seams(images, masks, corners)
# seamed_images = apply_seams_to_images(images, seam_masks)
apply_seams_to_tif(input_image_paths_array, seam_masks, output_paths)
