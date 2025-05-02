"""
Benchmark full pipeline  (global_match + local_match)

• Synthetic 8-band rasters, 2 overlapping tiles
• Tile size = 1024 × 1024 for both steps
• Serial vs parallel (32 workers)
"""

import os, shutil, tempfile, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin

from spectralmatch.process import global_match, local_match


# ───────────────────── synthetic raster helper ─────────────────────────────
def make_fake_rasters(out_dir, n_images, width, height, nodata=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=width,
        height=height,
        count=8,
        dtype="uint16",
        nodata=nodata,
        crs="EPSG:3857",
        transform=from_origin(0, 0, 1, 1),
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="LZW",
    )
    rng = np.random.default_rng(seed=42)
    paths = []
    for i in range(n_images):
        p = out_dir / f"fake_{i+1}_{width}px.tif"
        with rasterio.open(p, "w", **profile) as dst:
            for b in range(1, 9):
                data = rng.integers(1, 1000, size=(height, width), dtype="uint16")
                data[0, 0] = nodata
                dst.write(data, indexes=b)
        paths.append(str(p))
    return paths


SIZES = [2_048, 4_096, 6_144, 8_192, 10_240, 12_288]
NUM_IMAGES = 2
TILE_SIZE = (1024, 1024)
MAX_WORKERS = 32

WORK_DIR = Path(__file__).parent / "bench_output"
WORK_DIR.mkdir(exist_ok=True)

SERIAL, PARALLEL = [], []

for sz in SIZES:
    print(f"\n=== {sz} × {sz} px  ({NUM_IMAGES} images) ===")
    tmp = Path(tempfile.mkdtemp(prefix=f"fake_{sz}px_", dir=WORK_DIR))
    imgs = make_fake_rasters(tmp, NUM_IMAGES, sz, sz)

    t0 = time.time()
    g_dir = tmp / "serial_g"
    l_dir = tmp / "serial_l"

    global_match(
        imgs,
        g_dir,
        custom_mean_factor=3,
        custom_std_factor=1,
        tile_width_and_height_tuple=TILE_SIZE,
        parallel=False,
        debug_mode=False,
    )
    glob_imgs = sorted((g_dir / "Images").glob("*.tif"))

    local_match(
        [str(p) for p in glob_imgs],
        l_dir,
        target_blocks_per_image=100,
        tile_width_and_height_tuple=TILE_SIZE,
        custom_nodata_value=-9999,
        parallel=False,
        debug_mode=False,
    )
    SERIAL.append(time.time() - t0)
    print(f"serial   : {SERIAL[-1]:.1f} s")

    t0 = time.time()
    g_dir = tmp / "parallel_g"
    l_dir = tmp / "parallel_l"

    global_match(
        imgs,
        g_dir,
        custom_mean_factor=3,
        custom_std_factor=1,
        tile_width_and_height_tuple=TILE_SIZE,
        parallel=True,
        max_workers=MAX_WORKERS,
        debug_mode=False,
    )
    glob_imgs = sorted((g_dir / "Images").glob("*.tif"))

    local_match(
        [str(p) for p in glob_imgs],
        l_dir,
        target_blocks_per_image=100,
        tile_width_and_height_tuple=TILE_SIZE,
        custom_nodata_value=-9999,
        parallel=True,
        max_workers=MAX_WORKERS,
        debug_mode=False,
    )
    PARALLEL.append(time.time() - t0)
    print(f"parallel : {PARALLEL[-1]:.1f} s")

    shutil.rmtree(tmp, ignore_errors=True)

plt.figure(figsize=(8, 5))
plt.plot(SIZES, SERIAL, "-o", label="serial")
plt.plot(SIZES, PARALLEL, "-o", label=f"parallel ({MAX_WORKERS} workers)")
plt.xlabel("Raster width = height (pixels)")
plt.ylabel("Total runtime: global + local (seconds)")
plt.title("Pipeline runtime vs. image size (8-band, 2 images)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
