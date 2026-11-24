import importlib
from pathlib import Path
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal, osr
from shapely.geometry import Polygon, box

from spectralmatch.footprint_reads import load_footprints, footprint_pairs, sparse_footprint_vrt


def raster(tmp_path, name='scene', transform=(0, 1, 0, 256, 0, -1)):
    path = str(tmp_path / (name + '.tif'))
    ds = gdal.GetDriverByName('GTiff').Create(path, 256, 256, 2, gdal.GDT_Float32,
          options=['TILED=YES', 'BLOCKXSIZE=32', 'BLOCKYSIZE=32', 'COMPRESS=DEFLATE'])
    ds.SetGeoTransform(transform)
    sr = osr.SpatialReference(); sr.ImportFromEPSG(3857); ds.SetProjection(sr.ExportToWkt())
    values = np.full((256, 256), -9999, dtype=np.float32)
    # Separate islands, a hole, and a boundary that cuts through TIFF tiles.
    values[17:113, 21:117] = np.random.default_rng(9).normal(30, 4, (96, 96))
    values[45:89, 45:89] = -9999
    values[201:222, 197:231] = 17
    for n in (1, 2):
        band = ds.GetRasterBand(n); band.SetNoDataValue(-9999)
        band.WriteArray(np.where(values == -9999, -9999, values*n))
    ds = None
    from spectralmatch.seamline.footprints import _footprint_from_image
    geom, crs = _footprint_from_image(path, 1, True)
    return path, geom, crs


def vectors(tmp_path, path, geometry, crs):
    output = str(tmp_path/'footprints.gpkg')
    gpd.GeoDataFrame({'image': [Path(path).stem]}, geometry=[geometry], crs=crs).to_file(output, layer='footprints')
    return output


@pytest.mark.parametrize('transform', [(0,1,0,256,0,-1), (10,1,.2,256,.1,-1)])
def test_sparse_reads_preserve_values_masks_stats_and_grid(tmp_path, transform):
    path, geom, _ = raster(tmp_path, transform=transform)
    view = sparse_footprint_vrt(path, geom.wkb, tmp_path/'sparse.vrt')
    source, sparse = gdal.Open(path), gdal.Open(view)
    assert source.GetGeoTransform() == sparse.GetGeoTransform()
    for b in (1,2):
        left, right = source.GetRasterBand(b), sparse.GetRasterBand(b)
        np.testing.assert_array_equal(left.ReadAsArray(), right.ReadAsArray())
        np.testing.assert_array_equal(left.GetMaskBand().ReadAsArray(), right.GetMaskBand().ReadAsArray())
        for approximate in (False, True):
            np.testing.assert_allclose(left.ComputeStatistics(approximate), right.ComputeStatistics(approximate), rtol=1e-12)
    root = ET.parse(view).getroot()
    sources = root.find('VRTRasterBand').findall('SimpleSource')
    if not sources:
        sources = root.find('VRTRasterBand').findall('ComplexSource')
    area = sum(int(s.find('SrcRect').get('xSize')) * int(s.find('SrcRect').get('ySize')) for s in sources)
    assert area < 256*256/2  # Most input blocks are never referenced.


def test_loading_crs_suffix_ids_missing_and_fixed_reference(tmp_path):
    path, geom, crs = raster(tmp_path)
    output = vectors(tmp_path, path, geom, crs)
    # Test actual reprojection as well as suffix reuse.
    frame = gpd.read_file(output).to_crs('EPSG:4326')
    frame.to_file(output, layer='footprints')
    current = str(tmp_path/'scene_global_local.tif')
    gdal.Translate(current, path)
    result = load_footprints(output, [current, 'lidar.tif'], optional_paths=['lidar.tif'])
    from shapely import from_wkb
    assert from_wkb(result[current]).hausdorff_distance(geom) < 1e-6
    assert 'lidar.tif' not in result
    with pytest.raises(ValueError, match='No input_footprints'):
        load_footprints(output, ['missing.tif'])
    assert load_footprints(None, ['missing.tif']) == {}


def test_pair_pruning_retains_unknown_reference(tmp_path):
    paths={'a':'a.tif','b':'b.tif','lidar':'lidar.tif'}
    footprints={'a.tif':box(0,0,10,10).wkb, 'b.tif':box(20,20,30,30).wkb}
    assert footprint_pairs((('a','b'),('a','lidar')), paths, footprints) == (('a','lidar'),)


def test_joint_shared_valid_area_and_centers_unchanged(tmp_path):
    module=importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')
    from spectralmatch.joint_coregistration.spread_matching import _shared_mask
    from spectralmatch.overlap_sampling import spread_centers
    path, geom, _ = raster(tmp_path)
    results=[]
    for label, footprint in [('old',None),('new',geom.wkb)]:
        folder=tmp_path/label; folder.mkdir()
        info=module._read_image_info('scene',path,footprint=footprint)
        a,b=module._build_pair_overlap_vrts(info,info,str(folder))
        mask, area, index=_shared_mask(gdal.Open(a),gdal.Open(b),info,info,str(folder))
        import itertools
        centers=list(itertools.islice(spread_centers(mask,index=index),10))
        results.append((mask.copy(),area,centers))
        mask._mmap.close()
    np.testing.assert_array_equal(results[0][0],results[1][0])
    assert results[0][1:] == results[1][1:]


def test_entire_overlap_and_local_statistics_unchanged(tmp_path):
    from spectralmatch.match.global_regression import _overlap_stats_process_image
    from spectralmatch.match.local_block_adjustment import _calculate_block_process_image
    path, geom, _ = raster(tmp_path)
    args=(False,1,2,path,path,'a','b',(0,0,256,256),(0,0,256,256),False,None,-9999,False)
    old=_overlap_stats_process_image(*args)
    new=_overlap_stats_process_image(*args, footprints=(geom.wkb,geom.wkb))
    assert old == new
    args=('scene',path,(0,0,256,256),8,8,2,False,-9999,'float32',False,1)
    _, old=_calculate_block_process_image(*args)
    _, new=_calculate_block_process_image(*args,footprint=geom.wkb)
    np.testing.assert_array_equal(old,new)


@pytest.mark.parametrize('identifier', ['scene_coreg', 'folder/scene_coreg.tif', r'folder\scene_coreg.TIF'])
def test_seamlines_match_aligned_footprints_after_radiometry(tmp_path, identifier):
    from spectralmatch.seamline.footprints import _match_footprints
    frame=gpd.GeoDataFrame({'image':[identifier]},geometry=[box(0,0,1,1)],crs='EPSG:3857')
    result=_match_footprints(frame,'image',[str(tmp_path/'scene_coreg_global_local.tif')])
    assert result.image.tolist() == ['scene_coreg_global_local']
    assert result.geometry.iloc[0].equals(frame.geometry.iloc[0])


def test_sparse_approximate_statistics_with_overviews(tmp_path):
    path, geom, _ = raster(tmp_path)
    enlarged = str(tmp_path/'large.tif')
    gdal.Translate(enlarged, path, width=2048, height=2048,
                   creationOptions=['TILED=YES','BLOCKXSIZE=128','BLOCKYSIZE=128'])
    ds=gdal.Open(enlarged,gdal.GA_Update)
    ds.BuildOverviews('AVERAGE',[2,4,8]); ds=None
    view=sparse_footprint_vrt(enlarged,geom.wkb,tmp_path/'large.vrt')
    a,b=gdal.Open(enlarged),gdal.Open(view)
    for band in (1,2):
        np.testing.assert_allclose(a.GetRasterBand(band).ComputeStatistics(True),
                                   b.GetRasterBand(band).ComputeStatistics(True),rtol=1e-10)


def test_public_global_local_outputs_identical(tmp_path):
    from spectralmatch import global_regression, local_block_adjustment
    first,geom,crs=raster(tmp_path,'scene')
    second,_,_=raster(tmp_path,'other')
    ds=gdal.Open(second,gdal.GA_Update)
    for b in (1,2):
        band=ds.GetRasterBand(b); data=band.ReadAsArray()
        band.WriteArray(np.where(data==-9999,-9999,data*1.12+3))
    ds=None
    fp=str(tmp_path/'both.gpkg')
    gpd.GeoDataFrame({'image':['scene','other']},geometry=[geom,geom],crs=crs).to_file(fp,layer='footprints')
    for function,kwargs in [(global_regression,dict(pif_method='entire',estimate_stats=False)),
                            (local_block_adjustment,dict(number_of_blocks=(8,8),correction_method='offset'))]:
        results=[]
        for label,footprints in [('old',None),('new',fp)]:
            out=[str(tmp_path/f'{function.__name__}_{label}_{i}.tif') for i in (0,1)]
            function([first,second],out,input_footprints=footprints,**kwargs)
            results.append([gdal.Open(path).ReadAsArray() for path in out])
        for old,new in zip(*results):
            np.testing.assert_array_equal(old,new)


def test_pif_full_overlap_statistics_with_footprints(tmp_path):
    from spectralmatch.pif.pif import Pif
    joint=importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')
    a,geom,crs=raster(tmp_path,'a')
    b,_,_=raster(tmp_path,'b')
    fp=str(tmp_path/'pif.gpkg')
    gpd.GeoDataFrame({'image':['a','b']},geometry=[geom,geom],crs=crs).to_file(fp,layer='footprints')
    points=np.asarray([[x,y,x,y] for x,y in [(30,30),(100,30),(30,100),(100,100)]],dtype=float)
    ties=str(tmp_path/'ties.json');joint._save_ties(ties,{('a','b'):points})
    old=Pif.flood_from_match_points([a,b],load_ties=ties,tie_sampling_method='grid',
                                   estimate_stats=False,region_radius=3,_return_statistics=True)
    new=Pif.flood_from_match_points([a,b],load_ties=ties,tie_sampling_method='grid',
                                   estimate_stats=False,region_radius=3,_return_statistics=True,input_footprints=fp)
    np.testing.assert_array_equal(old[0],new[0])
    assert old[1:] == new[1:]


def test_resized_vrt_sources_keep_their_original_grid(tmp_path):
    path, geom, _=raster(tmp_path)
    resized=str(tmp_path/'resized.vrt')
    gdal.Translate(resized,path,format='VRT',width=128,height=128)
    view=sparse_footprint_vrt(resized,geom.wkb,tmp_path/'sparse_resized.vrt')
    a,b=gdal.Open(resized),gdal.Open(view)
    np.testing.assert_array_equal(a.ReadAsArray(),b.ReadAsArray())
    np.testing.assert_array_equal(a.GetRasterBand(1).GetMaskBand().ReadAsArray(),
                                  b.GetRasterBand(1).GetMaskBand().ReadAsArray())


def test_joint_alignment_outputs_unchanged_with_footprints(tmp_path):
    from spectralmatch import joint_coregistration
    module=importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')
    a,geom,crs=raster(tmp_path,'a')
    b,_,_=raster(tmp_path,'b')
    fp=str(tmp_path/'joint.gpkg')
    gpd.GeoDataFrame({'image':['a','b']},geometry=[geom,geom],crs=crs).to_file(fp,layer='footprints')
    points=np.asarray([[x,y,x,y] for x,y in [(30,30),(100,30),(30,100),(100,100)]],dtype=float)
    ties=str(tmp_path/'joint_ties.json');module._save_ties(ties,{('a','b'):points})
    results=[]
    for label,footprints in [('old',None),('new',fp)]:
        outputs=[str(tmp_path/f'{label}_{name}.tif') for name in ('a','b')]
        joint_coregistration([a,b],outputs,input_footprints=footprints,tie_load_path=ties,
                             tie_sampling_method='grid',global_model='translation',local_model='none')
        results.append([(gdal.Open(path).GetGeoTransform(),gdal.Open(path).ReadAsArray()) for path in outputs])
    for before,after in zip(*results):
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1],after[1])


def test_footprint_ids_with_dots_and_ambiguous_case():
    from spectralmatch.footprint_reads import matching_identifier
    assert matching_identifier('scene.v1_global.tif', ['scene.v1']) == 'scene.v1'
    assert matching_identifier('scene_long_global.tif', ['scene','scene_long']) == 'scene_long'
    with pytest.raises(ValueError, match='Ambiguous'):
        matching_identifier('scene.tif',['scene','SCENE'])
