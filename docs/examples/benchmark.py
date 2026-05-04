import os, time, tempfile, glob
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr
from spectralmatch import Match


def make_two_rasters(size, out_dir):
    """Create two square rasters that overlap by 50% in X."""
    def _create(path, x0):
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(path, size, size, NUM_BANDS, gdal.GDT_UInt16,
                        options=[
                            "TILED=YES",
                            f"BLOCKXSIZE={BLOCK}", f"BLOCKYSIZE={BLOCK}",
                            "COMPRESS=LZW", "BIGTIFF=YES"
                        ])
        srs = osr.SpatialReference(); srs.ImportFromEPSG(EPSG)
        ds.SetProjection(srs.ExportToWkt())
        ds.SetGeoTransform((x0, PIX, 0, 0, 0, -PIX))
        rng = np.random.default_rng(42)
        for b in range(1, NUM_BANDS+1):
            ds.GetRasterBand(b).WriteArray(rng.integers(1, 1000, size=(size, size), dtype=np.uint16))
        ds.FlushCache(); ds = None

    a = os.path.join(out_dir, f"A_{size}.tif")
    b = os.path.join(out_dir, f"B_{size}.tif")
    make_input = lambda p, xoff: _create(p, xoff)
    make_input(a, 0.0)
    make_input(b, (size * PIX) / 2.0)  # 50% overlap shift
    return [a, b]

def run_one(size, base):
    """Create inputs, run global+local, return (t_global, t_local)."""
    inp = os.path.join(base, "Input"); os.makedirs(inp, exist_ok=True)
    gdir = os.path.join(base, "GlobalMatch"); os.makedirs(gdir, exist_ok=True)
    ldir = os.path.join(base, "LocalMatch"); os.makedirs(ldir, exist_ok=True)

    make_two_rasters(size, inp)

    t0 = time.perf_counter()
    Match.global_regression(
        input_images=inp,
        output_images=gdir,
        debug_logs=True,
        custom_nodata_value=9999,
        window_size=BLOCK,
        image_threads=IMAGE_THREADS,
        io_threads=IO_THREADS,
        tile_threads=TILE_THREADS,
        custom_std_factor=3,
        save_adjustments=os.path.join(gdir, "GlobalAdjustments.json"),
    )
    t1 = time.perf_counter()

    ref_map = os.path.join(ldir, "ReferenceBlockMap", "ReferenceBlockMap.tif"); os.makedirs(ref_map, exist_ok=True)
    local_maps_tpl = os.path.join(ldir, "LocalBlockMap", "$_LocalBlockMap.tif"); os.makedirs(local_maps_tpl, exist_ok=True)

    t2 = time.perf_counter()
    Match.local_block_adjustment(
        input_images=gdir,
        output_images=ldir,
        debug_logs=True,
        custom_nodata_value=9999,
        window_size=BLOCK,
        image_threads=IMAGE_THREADS,
        io_threads=IO_THREADS,
        tile_threads=TILE_THREADS,
        number_of_blocks=50,
        # save_block_maps=(ref_map, local_maps_tpl),
    )
    t3 = time.perf_counter()

    return (t1 - t0, t3 - t2)

def main():
    with tempfile.TemporaryDirectory(prefix="spectralmatch_") as tmp:
        t_g, t_l = [], []
        for sz in SIZES:
            # fresh subfolders each size
            for d in ["Input", "GlobalMatch", "LocalMatch"]:
                p = os.path.join(tmp, d)
                if os.path.isdir(p):
                    for f in glob.glob(os.path.join(p, "**", "*"), recursive=True):
                        try: os.remove(f)
                        except: pass
                os.makedirs(p, exist_ok=True)
            tg, tl = run_one(sz, tmp)
            print(f"{sz:6d}: global={tg:7.2f}s  local={tl:7.2f}s")
            t_g.append(tg); t_l.append(tl)

        # Plot
        plt.figure()
        plt.plot(SIZES, t_g, marker="o", label="Global")
        plt.plot(SIZES, t_l, marker="s", label="Local")
        plt.xlabel("Raster size (pixels)"); plt.ylabel("Time (s)")
        plt.title("spectralmatch timings vs size"); plt.legend(); plt.grid(True, linestyle=":")
        plt.tight_layout(); plt.savefig(OUTPUT_PATH, dpi=150)
        print("Plot:", OUTPUT_PATH)

if __name__ == "__main__":
    SIZES = [2048, 4096,]# 6144, 8192, 10240, 12288]
    NUM_BANDS = 8
    EPSG = 3857
    PIX = 1.0

    BLOCK = 512
    IMAGE_THREADS = 2
    IO_THREADS = 2
    TILE_THREADS = 2

    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_results.png")

    main()
