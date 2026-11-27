import json
import subprocess
import numpy as np
import pytest
from safe_parking.stage import config
from safe_parking.solver import ipopt_options, terminal_bands, solve_round
from safe_parking.outcome import outcome
from safe_parking.validator import validate
from safe_parking.motion import rollout
from safe_parking.geometry import Geometry
from traj_py.scenario import Scenario, Vehicle
from traj_py.warm_start import make_warm_start


def scene(goal):
    return Scenario(Vehicle(2.7,np.array([.5,.5,.5,.5]),3.95),
        np.array([-10,10,-10,10]),np.zeros(3),np.asarray(goal,float),(),(),(),np.zeros((1,1),bool),5.)


@pytest.mark.parametrize('position,heading,expected',[(.05,.05,True),(.05+1e-10,0,False),(0,.05+1e-10,False)])
def test_terminal_acceptance_is_not_expanded(position,heading,expected):
    s=scene([position,0,heading]);X=np.zeros((4,2));U=np.zeros((2,1))
    v=validate(X,U,.4,s,Geometry((),s.veh.ego),config())
    assert v['terminal_pass']==expected


@pytest.mark.parametrize('end_speed,expected',[(0,True),(5e-7,True),(2e-6,False),(.05,False)])
def test_terminal_speed_uses_global_rollout(end_speed,expected):
    c=config();U=np.array([[0.],[end_speed/.4]]);X=rollout(np.zeros(4),U,.4)
    s=scene(X[:3,-1]);v=validate(X,U,.4,s,Geometry((),s.veh.ego),c)
    assert v['checks']['terminal_speed']==expected
    assert v['accepted']==expected


def test_terminal_insets_cover_declared_rollout_guards():
    c=config();bands=terminal_bands(c)
    assert bands[0]+c['global_rollout_position_drift_max_m']+c['dynamics_residual_max']<c['goal_xy_component_tol']
    assert bands[2]+c['global_rollout_heading_drift_max_rad']+c['dynamics_residual_max']<c['goal_heading_tol']


def test_zero_terminal_speed_in_real_solver():
    s=scene([.8,0,0]);path=np.column_stack((np.linspace(0,.8,20),np.zeros((20,2))))
    p=make_warm_start(path,s,parking=True,horizon=8).problem;p.x_initial[3,[0,-1]]=0
    r,_=solve_round(p,set(),config())
    assert abs(r.X[3,-1])<1e-8
    v=validate(r.X,r.U,.4,s,Geometry((),s.veh.ego),config())
    assert v['accepted'] and abs(v['terminal_speed'])<=config()['goal_speed_tol']


def test_limits_are_omitted_not_huge_numbers():
    c=config();assert 'max_cpu_time' not in ipopt_options(c) and 'max_wall_time' not in ipopt_options(c)
    assert ipopt_options(c)['max_iter']==1500
    assert c['adaptive_max_rounds']==6 and c['adaptive_max_new_samples_per_round']==24
    c['solver_max_cpu_seconds_per_round']=2
    assert ipopt_options(c)['max_cpu_time']==2


def test_process_timeout_none_reaches_subprocess(monkeypatch,tmp_path):
    from safe_parking.suite import child
    seen=[]
    def fake(*args,**kwargs):
        seen.append(kwargs['timeout'])
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(subprocess,'run',fake)
    c=config()
    for key in ['ha_hard_timeout_seconds','case_hard_timeout_seconds']:
        assert child(['pack'],tmp_path,c[key]) is None
    assert seen==[None,None]


def test_fresh_ha_receives_no_time_limit(monkeypatch,tmp_path):
    from safe_parking import suite
    from traj_py.hybrid_astar import SearchStopped
    def stopped(s,settings):
        assert settings.max_time is None and settings.max_expansions==200000
        raise SearchStopped('EXPANSION_LIMIT',200000)
    monkeypatch.setattr(suite,'plan_hybrid_astar',stopped)
    suite.pack('perpendicular_tight_x-8_y7.5','fresh',tmp_path)
    r=json.loads((tmp_path/'ha.json').read_text())
    assert r['stop_reason']=='EXPANSION_LIMIT' and r['expanded']==200000


def test_stop_reason_does_not_hide_terminal_failure():
    v={'accepted':False,'curve':{'state':'PASS'},'failure_items':['goal_heading']}
    r=outcome('Solve_Succeeded','NO_NEW_SAMPLE',v)
    assert not r['accepted'] and r['validation_status']=='PASS'
    assert r['failure_items']==['goal_heading'] and r['stop_reason']=='NO_NEW_SAMPLE'
    v['accepted']=True;v['failure_items']=[]
    r=outcome('Maximum_Iterations_Exceeded','ITERATION_LIMIT',v)
    assert not r['accepted'] and r['validation_status']=='PASS'


def test_actual_ha_expansion_cap():
    from dataclasses import replace
    from eval_parking.scenarios import RECTANGLE_PLANNER_SETTINGS,build_scene
    from traj_py.hybrid_astar import plan_hybrid_astar,SearchStopped
    s=build_scene('perpendicular','wide',np.array([-8,8.5,0]))
    with pytest.raises(SearchStopped) as e:
        plan_hybrid_astar(s,replace(RECTANGLE_PLANNER_SETTINGS,max_time=None,max_expansions=0))
    assert e.value.reason=='EXPANSION_LIMIT'


def test_worker_preserves_native_solver_exception(monkeypatch,tmp_path):
    from safe_parking import suite,stage
    from safe_parking.solver import SolveFailure
    suite.pack('perpendicular_wide_x0_y8.5','seeded',tmp_path)
    def fail(*args,**kwargs):
        raise SolveFailure('native error',{'return_status':'Infeasible_Problem_Detected','iter_count':17})
    monkeypatch.setattr(stage,'solve_round',fail)
    out=tmp_path/'adaptive'
    stage.worker(str(tmp_path/'input.npz'),out,'adaptive')
    r=json.loads((out/'result.json').read_text())
    assert r['solver_status']=='Infeasible_Problem_Detected'
    assert r['stop_reason']=='SOLVER_REPORTED_INFEASIBLE'
    assert r['validation_status']=='NOT_RUN' and not r['accepted']
    assert r['solver_stats']['iter_count']==17 and r['error']=='native error'
