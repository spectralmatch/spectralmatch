"""Footprint scheduling must save reads while preserving pixel/grid semantics."""
import importlib
import numpy as np
from osgeo import gdal,osr
from shapely.geometry import box
from spectralmatch.footprint_reads import FootprintWindows,occupied_rectangles


def test_rectangle_schedule_preserves_holes_and_original_grid():
    geometry=box(0,0,12,10).difference(box(3,3,9,7))
    index=FootprintWindows([geometry.wkb],(0,1,0,10,0,-1))
    result=np.zeros((10,12),bool)
    for x,y,w,h in occupied_rectangles(index,0,0,12,10):result[y:y+h,x:x+w]=True
    expected=np.array([[index.intersects((x,y,1,1)) for x in range(12)] for y in range(10)])
    np.testing.assert_array_equal(result,expected)
    assert not result.all()


def test_shared_mask_skips_empty_reads_without_changing_index(tmp_path,monkeypatch):
    joint=importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')
    spread=importlib.import_module('spectralmatch.joint_coregistration.spread_matching')
    path=str(tmp_path/'sparse.tif');s=osr.SpatialReference();s.ImportFromEPSG(3857)
    ds=gdal.GetDriverByName('GTiff').Create(path,2048,2048,1,gdal.GDT_Int16,options=['TILED=YES'])
    ds.SetGeoTransform((0,1,0,2048,0,-1));ds.SetProjection(s.ExportToWkt());band=ds.GetRasterBand(1)
    band.SetNoDataValue(-9999);band.Fill(-9999);band.WriteArray(np.ones((150,150),dtype='int16'),30,30);band=None;ds=None
    geometry=box(30,2048-180,180,2048-30)
    source=joint._read_image_info('a',path)
    from dataclasses import replace
    hinted=replace(source,footprint=geometry.wkb)
    original=joint._build_overlap_valid_mask;reads=[]
    def read(*args,**kwargs):reads.append(args[2]);return original(*args,**kwargs)
    monkeypatch.setattr(joint,'_build_overlap_valid_mask',read)
    ds=gdal.Open(path);results=[];counts=[]
    for label,info in [('full',source),('hint',hinted)]:
        folder=tmp_path/label;folder.mkdir();reads.clear()
        mask,area,index=spread._shared_mask(ds,ds,info,info,str(folder))
        results.append((np.array(mask),area,index.prefix.copy()));counts.append(len(reads));mask._mmap.close()
    for a,b in zip(*results):np.testing.assert_array_equal(a,b)
    assert counts == [8,1]


def test_lazy_pair_validity_matches_materialized_mask(tmp_path):
    from .test_pif_batched import sources
    from spectralmatch.pif import pif
    paths,w,h,gt,crs=sources(tmp_path)
    eager=pif._build_valid_mask_raster(*paths,w,h,gt,crs,str(tmp_path))
    lazy=pif._build_valid_mask_raster(*paths,w,h,gt,crs,str(tmp_path),materialize=False)
    assert lazy.endswith('.vrt')
    np.testing.assert_array_equal(gdal.Open(eager).ReadAsArray(),gdal.Open(lazy).ReadAsArray())


def test_sparse_corrected_pair_preserves_every_pixel_and_alpha(tmp_path):
    from .test_footprint_reads import raster
    from spectralmatch.joint_coregistration.joint_coregistration import _coregister_overlap
    from spectralmatch.utils_compression import compression_context
    a,_,_=raster(tmp_path,'a');b,_,_=raster(tmp_path,'b')
    points=np.array([[30,30,30,30],[100,30,100,30],[30,100,30,100],[100,100,100,100]])
    arrays=[]
    with compression_context(False):
        for sparse in [False,True]:
            path=str(tmp_path/f'corrected_{sparse}.tif')
            _coregister_overlap(a,b,path,tie_pairs=points,sparse_output=sparse)
            ds=gdal.Open(path)
            arrays.append(ds.ReadAsArray());ds=None
    np.testing.assert_array_equal(*arrays)
