"""Read-only fixed-reference validity checks for joint coregistration."""

import os
import tempfile

import numpy as np
from osgeo import gdal


def fully_covered_images(infos, fixed_names):
    """Moving images whose valid pixel centres all fall on valid reference data.

    Coverage is the union of selected-band GDAL masks (including NoData, explicit
    masks and alpha). Work in bounded chunks without polygonizing full rasters.
    """
    covered = set()
    with tempfile.TemporaryDirectory(prefix="spectralmatch_fixed_coverage_") as folder:
        masks = [[gdal.Translate(
            os.path.join(folder, f"mask_{index}_{band}.vrt"), infos[name].path,
            format="VRT", bandList=[f"mask,{band}"],
        ) for band in infos[name].matching_bands]
            for index, name in enumerate(sorted(fixed_names))]
        for name, info in infos.items():
            if name in fixed_names:
                continue
            source = gdal.Open(info.path, gdal.GA_ReadOnly)
            source_masks = [source.GetRasterBand(band).GetMaskBand() for band in info.matching_bands]
            north_up = (info.transform[2] == info.transform[4] == 0
                        and info.transform[1] > 0 and info.transform[5] < 0)
            warped = [[gdal.Warp(
                "", mask, format="VRT", dstSRS=info.projection,
                outputBounds=info.bounds, width=info.width, height=info.height,
                resampleAlg="near", srcNodata=0, dstNodata=0,
            ) for mask in group] for group in masks] if north_up else []
            complete = True
            for y in range(0, info.height, 512):
                for x in range(0, info.width, 512):
                    width, height = min(512, info.width-x), min(512, info.height-y)
                    valid = np.ones((height,width), dtype=bool)
                    for mask in source_masks:
                        valid &= mask.ReadAsArray(x,y,width,height) > 0
                    if not valid.any():
                        continue
                    union = np.zeros(valid.shape, dtype=bool)
                    for index, group in enumerate(masks):
                        reference_valid = np.ones(valid.shape, dtype=bool)
                        for band_index, mask in enumerate(group):
                            if north_up:
                                values = warped[index][band_index].ReadAsArray(x, y, width, height)
                            else:
                                # Reproject into the exact pixel grid for rotated inputs.
                                tile = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
                                gt = info.transform
                                origin = gdal.ApplyGeoTransform(gt, x, y)
                                tile.SetGeoTransform((origin[0], gt[1], gt[2], origin[1], gt[4], gt[5]))
                                tile.SetProjection(info.projection)
                                tile.GetRasterBand(1).SetNoDataValue(0)
                                tile.GetRasterBand(1).Fill(0)
                                gdal.ReprojectImage(mask, tile, mask.GetProjection(), info.projection,
                                                    gdal.GRA_NearestNeighbour)
                                values = tile.ReadAsArray()
                                tile = None
                            reference_valid &= values > 0
                        union |= reference_valid
                        if union[valid].all():
                            break
                    if np.any(valid & ~union):
                        complete = False
                        break
                if not complete:
                    break
            if complete:
                covered.add(name)
            source_masks.clear()
            source = None
            warped.clear()
        masks.clear()
    return covered


def valid_fixed_ties(ties, infos, fixed_names):
    """Check fixed endpoints, including selected points loaded from a tie cache."""
    filtered = {}
    for pair, points in ties.items():
        keep = np.ones(len(points), dtype=bool)
        for side, name in enumerate(pair):
            if name not in fixed_names or not len(points):
                continue
            source = gdal.Open(infos[name].path, gdal.GA_ReadOnly)
            pixels = np.floor(points[:, side*2:side*2+2] + .5).astype(np.int64)
            inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < source.RasterXSize)
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] < source.RasterYSize))
            valid = inside.copy()
            groups = {}
            for index in np.flatnonzero(inside):
                col, row = pixels[index]
                groups.setdefault((int(col)//512, int(row)//512), []).append(index)
            for band_index in infos[name].matching_bands:
                band = source.GetRasterBand(band_index)
                mask = band.GetMaskBand()
                for (bx, by), indices in groups.items():
                    x, y = bx*512, by*512
                    width, height = min(512,source.RasterXSize-x), min(512,source.RasterYSize-y)
                    values = mask.ReadAsArray(x,y,width,height)
                    samples = band.ReadAsArray(x,y,width,height)
                    columns, rows = pixels[indices].T
                    sample_values = samples[rows-y, columns-x]
                    valid[indices] &= (values[rows-y, columns-x] > 0) & np.isfinite(sample_values)
                    nodata = band.GetNoDataValue()
                    if nodata is not None:
                        valid[indices] &= sample_values != nodata
            keep &= valid
            mask = None
            source = None
        filtered[pair] = points[keep]
    return filtered
