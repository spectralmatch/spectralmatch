import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box
from spectralmatch import create_footprints, postprocess_footprints
from spectralmatch.seamline.footprints import _coverage_polygons
from spectralmatch.seamline.weighted_seamline import _rank_gap_polygons


@pytest.mark.parametrize('gap', [False, True])
def test_removed_owner_leaves_area_for_next_image(gap):
    small=box(900,900,1100,1100)
    frame=gpd.GeoDataFrame({'image':['small','large'],'quality':[1000,0]},geometry=[small,box(0,0,2000,2000)],crs=3857)
    result=_coverage_polygons(frame,'image',outline_path=None,outline_layer=None,
        image_rank_function='{var:quality}'+(' + 200 * {stat:gap_coverage_fraction}' if gap else ''),
        image_rank_descending=True,extrude_distance=0,do_postprocess_footprints=True)
    assert result.image.tolist()==['large']
    assert result.geometry.iloc[0].covers(small)
    if gap: assert result.weighted_score.iloc[0]==pytest.approx(200)


def test_gap_ranking_uses_cleaned_ownership():
    frame=gpd.GeoDataFrame({'image':['a','b','c'],'quality':[1000,0,0]},
        geometry=[box(0,0,1000,1000),box(0,0,1000,1000),box(2000,0,2100,100)],crs=3857)
    rows=_rank_gap_polygons(frame,'image','{var:quality} + 200 * {stat:gap_coverage_fraction}',True)
    assert next(rows)['image']=='a'
    assert rows.send(Polygon())['image']=='b'
    assert rows.send(Polygon())['image']=='c'


@pytest.mark.parametrize('params',[{}, {'simplify_smoothing_radius':0, 'simplify_tolerance':0,'filter_area_rank':None,'input_layer':'source','output_layer':'clean'}])
def test_public_forwarding_matches_standalone_defaults(tmp_path,params):
    geometry=box(0,0,2000,2000).union(box(2000,900,2500,1100))
    raw=tmp_path/'raw.gpkg'; output=tmp_path/'coverage.gpkg'; expected=tmp_path/'expected.gpkg'
    gpd.GeoDataFrame({'image':['a'],'image_path':['/images/a.tif'],'quality':[5]},geometry=[geometry],crs=3857).to_file(raw,layer=params.get('input_layer','footprints'))
    postprocess_footprints(str(raw),str(expected),**params)
    create_footprints([str(raw)],str(output),footprint_source='vector_bounds',vector_layer=params.get('input_layer','footprints'),
        do_postprocess_footprints=True,postprocess_footprints_params=params)
    actual=gpd.read_file(output); target=gpd.read_file(expected)
    assert actual.geometry.iloc[0].symmetric_difference(target.geometry.iloc[0]).area < 1e-6
    assert actual.quality.iloc[0]==5


@pytest.mark.parametrize('params',[{'unknown_cleanup_option':1},{'input_polygons':'override.gpkg'}])
def test_bad_forwarded_parameters_rejected_before_work(tmp_path,params):
    with pytest.raises(TypeError):
        create_footprints(['/missing.tif'],str(tmp_path/'out.gpkg'),do_postprocess_footprints=True,postprocess_footprints_params=params)
    assert not (tmp_path/'out.gpkg').exists()


def test_unchanged_gap_intersections_are_reused(monkeypatch):
    import importlib
    module=importlib.import_module('spectralmatch.seamline.weighted_seamline')
    original=module._gap_intersection_areas
    calls=[]
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module,'_gap_intersection_areas',counted)
    frame=gpd.GeoDataFrame({'image':['a','b','c'],'quality':[3,2,1]},
        geometry=[box(0,0,1000,1000)]*3,crs=3857)
    rows=module._rank_gap_polygons(frame,'image','{var:quality} + 200 * {stat:gap_coverage_fraction}',True)
    assert next(rows)['image']=='a'
    assert rows.send(Polygon())['image']=='b'
    assert rows.send(Polygon())['image']=='c'
    assert len(calls)==1
