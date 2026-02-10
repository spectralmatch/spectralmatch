import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from osgeo import gdal


def compare_image_spectral_profiles_pairs(
    image_groups_dict: dict,
    output_figure_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    line_width: float = 1,
    estimate_stats: bool = True,
):
    """
    Plots paired spectral profiles for before-and-after image comparisons.

    Args:
        image_groups_dict (dict): Mapping of labels to image path pairs (before, after):
            {'Image A': [
                '/image/before/a.tif',
                'image/after/a.tif'
            ],
            'Image B': [
                '/image/before/b.tif',
                '/image/after/b.tif'
            ]}
        output_figure_path (str): Path to save the resulting comparison figure.
        title (str): Title of the plot.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        line_width (float, optional): Width of the spectral profiles lines. Default is 1.
        estimate_stats (bool, optional): Whether to estimate band statistics. Default is True.

    Outputs:
        Saves a spectral comparison plot showing pre- and post-processing profiles.
    """

    os.makedirs(os.path.dirname(output_figure_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = itertools.cycle(plt.cm.tab10.colors)

    for label, group in image_groups_dict.items():
        if len(group) == 2:
            image_path1, image_path2 = group
            color = next(colors)

            for i, image_path in enumerate([image_path1, image_path2]):
                ds = gdal.Open(image_path, gdal.GA_ReadOnly)
                if ds is None:
                    continue

                num_bands = ds.RasterCount
                mean_spectral = np.zeros(num_bands, dtype=float)

                for b in range(1, num_bands + 1):
                    band = ds.GetRasterBand(b)

                    # (min, max, mean, std)
                    stats = band.GetStatistics(False, estimate_stats)

                    # stats[2] is the mean
                    mean_spectral[b - 1] = stats[2]

                ds = None

                bands = np.arange(1, num_bands + 1)
                linestyle = "dashed" if i == 0 else "solid"

                plt.plot(
                    bands,
                    mean_spectral,
                    linestyle=linestyle,
                    color=color,
                    linewidth=line_width,
                    label=f"{label} - {'Before' if i == 0 else 'After'}",
                )

                ds = None

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
    )
    plt.tight_layout()
    plt.grid(True)
    plt.xticks(np.arange(1, num_bands + 1, 1))
    plt.savefig(output_figure_path, dpi=300)
    plt.close()
    print(f"Saved: {os.path.splitext(os.path.basename(output_figure_path))[0]}")


def compare_spatial_spectral_difference_band_average(
    input_images: list,
    output_figure_path: str,
    title: str,
    diff_label: str,
    subtitle: str,
    scale: tuple = None,
):
    """
    Computes and visualizes the mean per-pixel spectral difference between two coregistered, equal-size images.

    Args:
        input_images (list): List of two image file paths [before, after].
        output_figure_path (str): Path to save the resulting difference image (PNG).
        title (str): Title for the plot.
        diff_label (str): Label for the colorbar.
        subtitle (str): Subtitle text shown below the image.
        scale (tuple, optional): Tuple (vmin, vmax) to fix the color scale. Centered at 0.

    Raises:
        ValueError: If the input list doesn't contain exactly two image paths, or shapes mismatch.
    """
    if len(input_images) != 2:
        raise ValueError("input_images must be a list of exactly two image paths.")

    path1, path2 = input_images

    ds1 = gdal.Open(path1, gdal.GA_ReadOnly)
    ds2 = gdal.Open(path2, gdal.GA_ReadOnly)

    if ds1 is None or ds2 is None:
        raise RuntimeError("Failed to open one or both input images.")

    bands1, rows1, cols1 = ds1.RasterCount, ds1.RasterYSize, ds1.RasterXSize
    bands2, rows2, cols2 = ds2.RasterCount, ds2.RasterYSize, ds2.RasterXSize

    if (bands1, rows1, cols1) != (bands2, rows2, cols2):
        raise ValueError("Images must have the same dimensions.")

    # Read all bands
    img1 = np.stack(
        [ds1.GetRasterBand(i + 1).ReadAsArray() for i in range(bands1)]
    ).astype("float32")

    img2 = np.stack(
        [ds2.GetRasterBand(i + 1).ReadAsArray() for i in range(bands2)]
    ).astype("float32")

    nodata = ds1.GetRasterBand(1).GetNoDataValue()

    diff = img2 - img1

    if nodata is not None:
        mask = np.full(diff.shape[1:], True)
        for b in range(diff.shape[0]):
            mask &= (img1[b] != nodata) & (img2[b] != nodata)
        diff[:, ~mask] = np.nan

    with np.errstate(invalid="ignore"):
        mean_diff = np.full(diff.shape[1:], np.nan)
        valid_mask = ~np.all(np.isnan(diff), axis=0)
        mean_diff[valid_mask] = np.nanmean(diff[:, valid_mask], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    vmin, vmax = scale if scale else (np.nanmin(mean_diff), np.nanmax(mean_diff))
    max_abs = max(abs(vmin), abs(vmax))
    im = ax.imshow(mean_diff, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(diff_label)

    ax.set_title(title, fontsize=14, pad=12)
    if subtitle:
        ax.text(0.5, -0.1, subtitle, fontsize=10, ha="center", transform=ax.transAxes)

    ax.axis("off")
    plt.savefig(output_figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    ds1 = None
    ds2 = None

    print(f"Saved: {os.path.splitext(os.path.basename(output_figure_path))[0]}")

def compare_before_after_all_images(
    input_images_1: list,
    input_images_2: list,
    output_figure_path: str,
    title: str,
    ylabel_1: str,
    ylabel_2: str,
    image_names: list = None,
):
    """
    Creates a two-row image grid to compare before-and-after raster pairs with consistent per-row contrast stretching. Each column shows a pair of aligned images with transparent nodata. Supports 1- and 3-band rasters.

    Args:
        input_images_1 (list): List of file paths to the "before" images (top row).
        input_images_2 (list): List of file paths to the "after" images (bottom row).
        output_figure_path (str): Destination path to save the output PNG figure.
        title (str): Title of the entire figure.
        ylabel_1 (str): Y-axis label for the top row.
        ylabel_2 (str): Y-axis label for the bottom row.
        image_names (list, optional): List of image names to use as column titles. Must match the number of image pairs.

    Raises:
        AssertionError: If input lists have mismatched lengths or if `image_names` does not match image count.

    Output:
        Saves a PNG file with the comparison figure.
    """
    def read_as_3band(ds):
        count = ds.RasterCount
        if count >= 3:
            arr = np.stack(
                [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(3)]
            )
        else:
            single = ds.GetRasterBand(1).ReadAsArray()
            arr = np.repeat(single[np.newaxis, ...], 3, axis=0)
        return arr.astype("float32")

    def compute_row_stretch(paths):
        all_valid = [[] for _ in range(3)]
        for path in paths:
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"Failed to open {path}")

            nodata = ds.GetRasterBand(1).GetNoDataValue()
            img = read_as_3band(ds)

            mask = np.full(img.shape[1:], False)
            if nodata is not None:
                for b in range(img.shape[0]):
                    mask |= img[b] == nodata

            for b in range(img.shape[0]):
                all_valid[b].append(img[b][~mask])

            ds = None

        return [
            np.percentile(np.concatenate(valid), (2, 98))
            if valid else (0, 1)
            for valid in all_valid
        ]

    assert len(input_images_1) == len(input_images_2)
    if image_names:
        assert len(image_names) == len(input_images_1)

    os.makedirs(os.path.dirname(output_figure_path), exist_ok=True)

    num_images = len(input_images_1)
    fig = plt.figure(figsize=(5 * num_images, 10))
    gs = gridspec.GridSpec(2, num_images + 1, width_ratios=[0.05] + [1] * num_images)

    stretch_1 = compute_row_stretch(input_images_1)
    stretch_2 = compute_row_stretch(input_images_2)

    for col_idx, (path1, path2) in enumerate(zip(input_images_1, input_images_2)):
        for row_idx, (path, stretch) in enumerate(
            [(path1, stretch_1), (path2, stretch_2)]
        ):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])

            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"Failed to open {path}")

            nodata = ds.GetRasterBand(1).GetNoDataValue()
            img = read_as_3band(ds)

            mask = np.full(img.shape[1:], False)
            if nodata is not None:
                for b in range(img.shape[0]):
                    mask |= img[b] == nodata

            for b in range(img.shape[0]):
                vmin, vmax = stretch[b]
                img[b] = np.clip((img[b] - vmin) / (vmax - vmin), 0, 1)

            img = img.transpose(1, 2, 0)
            alpha = (~mask).astype("float32")
            rgba = np.dstack((img, alpha))

            ax.imshow(rgba)

            if row_idx == 0 and image_names:
                ax.set_title(image_names[col_idx])

            ax.axis("off")
            ds = None

    for i, label in enumerate([ylabel_1, ylabel_2]):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_ylabel(label, fontsize=12, rotation=90, labelpad=10, va="center")
        ax.tick_params(left=False, labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_figure_path, dpi=300)
    plt.close()

    print(f"Saved: {os.path.splitext(os.path.basename(output_figure_path))[0]}")