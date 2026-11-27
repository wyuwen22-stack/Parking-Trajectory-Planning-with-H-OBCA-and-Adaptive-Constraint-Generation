import json,math
from dataclasses import replace
from pathlib import Path
import casadi as ca
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from safe_parking.motion import arc,arc_symbolic,travel,motion_bound,rollout
from safe_parking.geometry import Geometry,body,decompose,area,convex_signed_distance
from safe_parking.validator import check_curve,validate
from safe_parking.stage import ROOT,config,load

@pytest.mark.parametrize('v,a,d,T',[(1,0,0,.4),(1,.4,.3,.4),(-1,-.2,.4,.4),(0,1,0,.4),(.1,-1,.4,.4),(1,0,1e-12,.4),(-1,.2,-.3,1.)])
def test_arc_against_independent_adaptive_ode(v,a,d,T):
    x=np.array([1.,2.,.7,v]);u=[d,a]
    def f(t,q):return [q[3]*math.cos(q[2]),q[3]*math.sin(q[2]),q[3]/2.7*math.tan(d),a]
    y=solve_ivp(f,[0,T],x,method='DOP853',rtol=1e-12,atol=1e-13).y[:,-1]
    assert np.max(np.abs(arc(x,u,T)-y))<2e-11
    q=arc(arc(x,u,.37*T),u,.63*T);assert np.max(np.abs(q-y))<2e-11

@pytest.mark.parametrize('delta',[0.,1e-12,.2,-.4])
def test_casadi_arc_value_and_jacobian_at_zero(delta):
    x=ca.MX.sym('x',4);u=ca.MX.sym('u',2);y=arc_symbolic(x,u,.4,2.7)
    f=ca.Function('arc_test',[x,u],[y,ca.jacobian(y,u)])
    a,b=f([0,0,.2,1],[delta,.3]);assert np.isfinite(np.array(b)).all()
    assert np.max(np.abs(np.array(a).ravel()-arc([0,0,.2,1],[delta,.3],.4)))<1e-12

def test_zero_speed_crossing_absolute_motion_not_net_displacement():
    assert travel(.1,-1,0,.2)==pytest.approx(.01)
    assert travel(.1,-1,0,.1)==pytest.approx(.005)
    assert np.linalg.norm(arc([0,0,0,.1],[0,-1],.2)[:2])<1e-15

def test_bound_contains_every_corner_with_reversal():
    ego=[3.7,1,1,1];x=[0,0,.5,.1];u=[.4,-1];T=.4;mid=.2;b=motion_bound(x,u,0,mid,T,ego)
    at=body(arc(x,u,mid),ego)
    for t in np.linspace(0,T,101):assert np.max(np.linalg.norm(body(arc(x,u,t),ego)-at,axis=1))<=b+1e-12

def box(x0,y0,x1,y1):return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
@pytest.mark.parametrize('offset,expected',[(2.,1.),(1.,0.),(.9,-.1),(1+1e-9,1e-9)])
def test_distance_touch_overlap_near_parallel(offset,expected):
    assert convex_signed_distance(box(0,0,1,1),box(offset,0,offset+1,1))==pytest.approx(expected,abs=2e-14)

def test_translation_invariance():
    a,b=box(0,0,1,1),box(2,.2,3,1)
    assert convex_signed_distance(a+1e6,b+1e6)==pytest.approx(convex_signed_distance(a,b),abs=1e-9)

def test_concave_decomposition_and_empty_notch():
    l=np.array([[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]])
    pieces=decompose(l);assert len(pieces)>1;assert sum(area(p) for p in pieces)==pytest.approx(area(l))
    assert min(convex_signed_distance(p,box(1.5,1.5,2,2)) for p in pieces)>.4
    assert min(convex_signed_distance(p,box(.2,1.2,.8,2)) for p in pieces)<0

def test_three_states_and_zero_contact_never_pass():
    X=np.array([[0,0],[0,0],[0,0],[0,0]],float);U=np.zeros((2,1));ego=[.5,.5,.5,.5]
    assert check_curve(X,U,.4,Geometry([box(3,3,4,4)],ego))['state']=='PASS'
    assert check_curve(X,U,.4,Geometry([box(.3,.3,.7,.7)],ego))['state']=='COLLISION'
    assert check_curve(X,U,.4,Geometry([box(3,3,4,4)],ego),max_queries=0)['state']=='UNKNOWN'
    assert check_curve(X,U,.4,Geometry([box(.5,-1,1,1)],ego),max_depth=2)['state']!='PASS'

def test_residual_and_global_drift_cannot_be_ignored():
    z,s=load('h_obca_parallel_wide_x-4_y7.5.npz');v=validate(z['X'],z['U'],.4,s,Geometry(s.obs,s.veh.ego),config())
    assert not v['dynamics_pass'];assert not v['accepted'];assert v['global_position_drift_m']>1e-4

def test_manifest128_and_separate_regression():
    catalog=json.loads((ROOT/'inputs/scene_catalog_129.json').read_text());assert sum(not c.get('regression',False) for c in catalog)==128
    assert sum(bool(c.get('regression',False)) for c in catalog)==1

def test_legacy_solve_obca_source_unchanged():
    import hashlib
    hashes=json.loads((ROOT/'readonly_inputs_sha256.json').read_text());old='/workspace/traj_py/python_scripts/traj_py/obca.py'
    assert hashlib.sha256((ROOT/'python_scripts/traj_py/obca.py').read_bytes()).hexdigest()==hashes[old]

def test_ha_no_precheck_hard_rejection():
    import inspect
    from traj_py.hybrid_astar import plan_hybrid_astar
    assert 'goal_component_escapes(' not in inspect.getsource(plan_hybrid_astar)

def test_safe_once_objective_and_endpoints_small_solve():
    from traj_py.scenario import Vehicle,Scenario
    from traj_py.warm_start import make_warm_start
    from safe_parking.solver import solve_round
    from eval_parking.metrics import parking_cost
    scene=Scenario(Vehicle(2.7,np.array([3.7,1,1,1]),5),np.array([-10,10,-10,10]),np.array([0.,0.,0.]),np.array([.8,0.,0.]),(),(),(),np.zeros((1,1),bool),5.)
    path=np.column_stack((np.linspace(0,.8,20),np.zeros((20,2))));p=make_warm_start(path,scene,parking=True,horizon=8).problem;p.parking_objective='once';p.x_initial[3,0]=0;p.x_initial[3,-1]=0
    result,structure=solve_round(p,{(k,.5) for k in range(8)},config())
    assert result.status=='Solve_Succeeded'
    assert result.objective==pytest.approx(parking_cost(result.X,result.U,p.x_initial,'once'),abs=1e-9)
    assert np.max(np.abs(result.X-rollout(np.r_[scene.start,0],result.U,.4)))<1e-6

@pytest.mark.parametrize('mode',['adaptive_seed','dense'])
def test_dense_and_adaptive_share_arc_and_validator(mode):
    from traj_py.scenario import Vehicle,Scenario,_poly2hrep
    from traj_py.warm_start import make_warm_start
    from safe_parking.solver import solve_round
    obs=box(-5,3,5,4);closed=np.vstack((obs,obs[0]));A,b=_poly2hrep(closed)
    s=Scenario(Vehicle(2.7,np.array([3.7,1,1,1]),5),np.array([-10,10,-10,10]),np.array([0.,0.,0.]),np.array([.8,0.,0.]),(closed,),(A,),(b,),np.zeros((1,1),bool),5.)
    path=np.column_stack((np.linspace(0,.8,20),np.zeros((20,2))));p=make_warm_start(path,s,parking=True,horizon=8).problem;p.parking_objective='once';p.x_initial[3,[0,-1]]=0
    fs=config()['dense_reference_fractions'] if mode=='dense' else [.5]
    sol,counts=solve_round(p,{(k,f) for k in range(8) for f in fs},config())
    v=validate(sol.X,sol.U,.4,s,Geometry(s.obs,s.veh.ego),config())
    assert sol.status=='Solve_Succeeded' and v['accepted']
    assert counts['interior_pose_constraints']==8*len(fs)

def test_random_motion_bound_with_fixed_seed():
    rng=np.random.default_rng(20261120);ego=[3.7,1,1,1]
    for _ in range(30):
        x=np.r_[rng.normal(size=3),rng.uniform(-3,3)];u=[rng.uniform(-.495,.495),rng.uniform(-1,1)]
        t0,tm,t1=0.,.2,.4;b=motion_bound(x,u,t0,tm,t1,ego);ref=body(arc(x,u,tm),ego)
        for t in np.linspace(0,.4,21):assert np.max(np.linalg.norm(body(arc(x,u,t),ego)-ref,axis=1))<=b+1e-11

def test_nearly_parallel_rotated_edges():
    a=box(0,0,1,1);b=box(2,0,3,1);angle=1e-9;c,s=np.cos(angle),np.sin(angle);b=b@np.array([[c,s],[-s,c]])
    assert convex_signed_distance(a,b)>0.99999999

def test_whole_trajectory_accepts_0049_node_clearance():
    from traj_py.scenario import Vehicle,Scenario,_poly2hrep
    o=box(3.749,-2,5,2);closed=np.vstack((o,o[0]));A,b=_poly2hrep(closed)
    s=Scenario(Vehicle(2.7,np.array([3.7,1,1,1]),5),np.array([-10,10,-10,10]),np.zeros(3),np.zeros(3),(closed,),(A,),(b,),np.zeros((1,1),bool),5.)
    X=np.zeros((4,2));U=np.zeros((2,1));r=validate(X,U,.4,s,Geometry(s.obs,s.veh.ego),config())
    assert r['curve']['state']=='PASS' and not r['node_pass'] and r['accepted']

def test_both_ha_rectangle_entrypoints_use_same_strict_005():
    from traj_py.scenario import Vehicle,Scenario,_poly2hrep
    from traj_py.hybrid_astar import rectangle_collision_free,make_rectangle_checker
    for clearance,expected in [(.0495,False),(.0505,True)]:
        o=box(3.7+clearance,-2,5,2);closed=np.vstack((o,o[0]));A,b=_poly2hrep(closed)
        s=Scenario(Vehicle(2.7,np.array([3.7,1,1,1]),5),np.array([-10,10,-10,10]),np.zeros(3),np.zeros(3),(closed,),(A,),(b,),np.zeros((1,1),bool),5.)
        q=np.zeros((1,3))
        assert rectangle_collision_free(q,s,min_clearance=.05)==expected
        assert make_rectangle_checker(s,min_clearance=.05)(q)==expected
