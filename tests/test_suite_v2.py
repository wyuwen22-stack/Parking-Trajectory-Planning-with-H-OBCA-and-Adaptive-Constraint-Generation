import inspect,json
from safe_parking.suite import specs,key,config
from safe_parking.stage import worker

def test_manifest128_no_regression():
    ss=specs();assert len(ss)==128 and len({key(s) for s in ss})==128 and not any(s.is_regression for s in ss)
    assert {f:sum(s.family==f for s in ss) for f in ['standard','diagonal','irregular','tpcap']}==dict(standard=60,diagonal=36,irregular=12,tpcap=20)

def test_budget_and_variants_frozen():
    c=config();assert all(c[k] is None for k in ['solver_max_cpu_seconds_per_round','case_hard_timeout_seconds','ha_search_seconds','ha_hard_timeout_seconds'])
    assert c['experiment_variants']==['node','dense','adaptive'] and c['dense_reference_fractions']==[.25,.5,.75]

def test_node_has_no_interior_seed():
    src=inspect.getsource(worker);assert "variant=='node'" in src

def test_all128_geometry_matches_frozen_catalog():
    import numpy as np
    from safe_parking.stage import ROOT
    from safe_parking.geometry import area
    from eval_parking.scenarios import scene_for_spec
    catalog=json.loads((ROOT/'manifest128.json').read_text())
    mapped={c['id']:c for c in catalog}
    for s in specs():
        k=key(s);c=mapped[k.replace('_project_4.7x2','')];scene=scene_for_spec(s)
        assert np.max(abs(scene.start-np.array(c['start'])))<1e-7
        assert np.max(abs(scene.goal-np.array(c['goal'])))<1e-7
        assert abs(sum(abs(area(p)) for p in scene.obs)-sum(abs(area(p)) for p in c['obstacles']))<1e-6
