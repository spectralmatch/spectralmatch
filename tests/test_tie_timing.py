import importlib
from types import SimpleNamespace
import numpy as np
import pytest

spread=importlib.import_module('spectralmatch.joint_coregistration.spread_matching')
joint=importlib.import_module('spectralmatch.joint_coregistration.joint_coregistration')


def test_tie_selection_does_not_depend_on_clock_speed(tmp_path,monkeypatch):
    info=SimpleNamespace(name='a');other=SimpleNamespace(name='b')
    dataset=SimpleNamespace(GetGeoTransform=lambda:(0,1,0,0,0,1))
    monkeypatch.setattr(joint,'_build_pair_overlap_vrts',lambda *a:('reference','sensed'))
    monkeypatch.setattr(spread.gdal,'Open',lambda *a:dataset)
    monkeypatch.setattr(spread,'spread_centers',lambda *a,**k:iter((0,i) for i in range(100)))
    monkeypatch.setattr(joint,'_extract_grid_pixel_matches',lambda *a,**k:(np.ones((1,4)),np.zeros((1,2))))
    monkeypatch.setattr(joint,'_overlap_to_original_pixels',lambda points,*a:points)
    monkeypatch.setattr(spread,'_valid_endpoints',lambda points,*a:points)
    results=[];checks=[]
    def accepted(candidates,*a):
        checks.append(len(candidates))
        return np.full((len(candidates),4),len(candidates),dtype=float)
    monkeypatch.setattr(spread,'_accepted',accepted)
    for speed in [0,2]:
        mask=np.memmap(tmp_path/f'{speed}.bin',mode='w+',dtype=bool,shape=(1,100));mask[:]=True
        monkeypatch.setattr(spread,'_shared_mask',lambda *a:(mask,9_500_000,None))
        clock=iter(i*speed for i in range(1000))
        monkeypatch.setattr(spread,'monotonic',lambda:next(clock),raising=False)
        checks.clear()
        result=spread.extract_spread_ties(info,other,'orb',None,None,'128px',100,1,'500crs','1000crs',None,3,str(tmp_path))
        assert checks==[9,18,27,36,45]
        results.append(result)
    np.testing.assert_array_equal(*results)
    assert results[0].shape==(38,4)
