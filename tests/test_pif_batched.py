"""Compare optimized reads against the original GDAL expression pipeline."""
import numpy as np
import pytest
from osgeo import gdal, osr
from spectralmatch.pif import pif
from spectralmatch.pif.window_statistics import difference_statistics


def sources(tmp_path, size=(769,513)):
    w,h=size; rng=np.random.default_rng(5); crs=osr.SpatialReference();crs.ImportFromEPSG(32605)
    base=rng.uniform(0,3000,(3,h,w)).astype(np.float32)
    # Nonuniform difference makes approximate-versus-exact sampling observable.
    delta=rng.normal(20,50,base.shape).astype(np.float32)+np.arange(w)[None,None,:]*.08
    paths=[]
    for label,values in [('ref',base),('sen',base+delta)]:
        values[:,30:45,40:70]=-9999
        values[:,55:60,80:85]=0  # valid zeros must not disappear
        path=str(tmp_path/f'{label}.tif')
        ds=gdal.GetDriverByName('GTiff').Create(path,w,h,3,gdal.GDT_Float32,options=['TILED=YES'])
        ds.SetGeoTransform((0,1,0,h,0,-1));ds.SetProjection(crs.ExportToWkt())
        for b in range(3):
            ds.GetRasterBand(b+1).SetNoDataValue(-9999);ds.GetRasterBand(b+1).WriteArray(values[b])
        ds=None;paths.append(path)
    return paths,w,h,(0,1,0,h,0,-1),crs.ExportToWkt()


@pytest.mark.parametrize('estimate',[False,True])
@pytest.mark.parametrize('save',[False,True])
@pytest.mark.parametrize('overviews',[False,True])
def test_fused_masks_scores_and_statistics_match_legacy(tmp_path,estimate,save,overviews):
    paths,w,h,gt,crs=sources(tmp_path)
    if overviews:
        for path in paths:
            with gdal.Open(path,gdal.GA_Update) as ds:ds.BuildOverviews('NEAREST',[2,4,8])
    valid=pif._build_valid_mask_raster(*paths,w,h,gt,crs,str(tmp_path))
    oldscore=str(tmp_path/'old_score.tif')
    oldmask=pif._build_inz_stable_mask_raster(ref_vrt=paths[0],sensed_vrt=paths[1],valid_mask_path=valid,
        width=w,height=h,gt=gt,projection=crs,num_bands=3,red_band_index=1,nir_band_index=3,
        vegetation_threshold=.2,inz_threshold=.9,tmpdir=str(tmp_path),estimate_stats=estimate,save_inz_path=oldscore)
    seeds=pif._build_seed_mask_raster(np.array([[57,82],[150,160],[155,165],[450,720]]),w,h,gt,crs,20,str(tmp_path))
    expected=pif._combine_masks_raster(oldmask,seeds,w,h,gt,crs,str(tmp_path))
    ds=gdal.Open(expected);expected_values=ds.ReadAsArray();ds=None
    expected_stats=[pif._masked_band_stats(path,b,expected,estimate_stats=estimate)
                    for path in paths for b in range(1,4)]
    saved=str(tmp_path/'new_score.tif') if save else None
    actual,count=pif._seed_pif_mask(*paths,valid,seeds,w,h,gt,crs,3,1,3,.2,.9,str(tmp_path),estimate,saved)
    ds=gdal.Open(actual);actual_values=ds.ReadAsArray();ds=None
    np.testing.assert_array_equal(actual_values,expected_values)
    assert count==np.count_nonzero(expected_values)>0
    if save:
        a,b=gdal.Open(oldscore),gdal.Open(saved)
        np.testing.assert_allclose(a.ReadAsArray(),b.ReadAsArray(),rtol=1e-6,atol=1e-6);a=b=None
    actual_stats=pif._selected_pair_statistics(*paths,actual,3,count,estimate)
    for a,b in zip(actual_stats,expected_stats):
        assert a['mean']==pytest.approx(b['mean'],rel=1e-10,abs=1e-9)
        assert a['std']==pytest.approx(b['std'],rel=1e-10,abs=1e-9)
        assert a['size']==b['size']


def test_sparse_approximate_sampling_retries_exact_population(tmp_path):
    paths,w,h,gt,crs=sources(tmp_path)
    mask=str(tmp_path/'mask.tif');ds=gdal.GetDriverByName('GTiff').Create(mask,w,h,1,gdal.GDT_Byte)
    ds.GetRasterBand(1).WriteArray(np.ones((2,2),np.uint8),400,200);ds=None
    expected=[pif._masked_band_stats(path,b,mask,estimate_stats=True) for path in paths for b in range(1,4)]
    actual=pif._selected_pair_statistics(*paths,mask,3,4,True)
    for a,b in zip(actual,expected):
        assert a['mean']==pytest.approx(b['mean'],rel=1e-10)
        assert a['std']==pytest.approx(b['std'],rel=1e-10)


def test_batched_statistics_read_errors_are_not_treated_as_empty_samples(tmp_path,monkeypatch):
    from spectralmatch.pif import window_statistics
    paths,w,h,gt,crs=sources(tmp_path)
    mask=pif._build_valid_mask_raster(*paths,w,h,gt,crs,str(tmp_path))
    def fail(*args,**kwargs):raise RuntimeError('Failed to read source raster block.')
    monkeypatch.setattr(window_statistics,'read_bands',fail)
    with pytest.raises(RuntimeError,match='Failed to read source raster block'):
        pif._selected_pair_statistics(*paths,mask,3,1,True)


def test_selected_valid_zeros_are_counted(tmp_path):
    paths,w,h,gt,crs=sources(tmp_path)
    path=str(tmp_path/'zero_mask.tif')
    with gdal.GetDriverByName('GTiff').Create(path,w,h,1,gdal.GDT_Byte) as ds:
        ds.GetRasterBand(1).WriteArray(np.ones((5,5),np.uint8),80,55)
    actual=pif._selected_pair_statistics(*paths,path,3,25,False)
    assert all(row=={'mean':0.,'std':0.,'size':25} for row in actual)


def test_mask_count_and_capped_row_major_selection(tmp_path):
    paths,w,h,gt,crs=sources(tmp_path)
    valid=pif._build_valid_mask_raster(*paths,w,h,gt,crs,str(tmp_path))
    seeds=pif._build_seed_mask_raster(np.array([[120,150],[125,155],[450,720]]),w,h,gt,crs,15,str(tmp_path))
    path,count=pif._seed_pif_mask(*paths,valid,seeds,w,h,gt,crs,3,None,None,.2,10,str(tmp_path),False)
    ds=gdal.Open(path);values=ds.ReadAsArray();ds=None
    assert count==np.count_nonzero(values)
    selected=pif._sample_mask_raster(path,53,w,h,gt,crs,str(tmp_path),count=count)
    ds=gdal.Open(selected);actual=ds.ReadAsArray();ds=None
    expected=np.flatnonzero(values)[np.linspace(0,count-1,53).astype(int)]
    np.testing.assert_array_equal(np.flatnonzero(actual),expected)
    assert not (tmp_path/'inz_score.tif').exists()
    assert not (tmp_path/'stable_mask.tif').exists()
