"""Three-state interval checks, independent of OBCA's dual formulation."""
import math
import numpy as np
from .motion import arc,motion_bound,rollout,speed_roots


def check_curve(X,U,T,g,L=2.7,model='constant_control_arc',max_depth=12,max_queries=16000):
    X=np.asarray(X);n=X.shape[1]-1;start_queries=g.queries
    r=math.hypot(max(g.ego[0],g.ego[2]),max(g.ego[1],g.ego[3]))
    cached={};bad=[];unresolved=[];lowers=[];samples=[]
    def pose(k,t):
        if model=='constant_control_arc':return arc(X[:,k],U[:,k],t,L)
        if model=='linear_pose':
            a=t/T;d=math.atan2(math.sin(X[2,k+1]-X[2,k]),math.cos(X[2,k+1]-X[2,k]));q=(1-a)*X[:,k]+a*X[:,k+1];q[2]=X[2,k]+a*d;return q
        raise ValueError('unknown model')
    def dist(k,t):
        key=(k,float(t))
        if key not in cached:
            cached[key]=g.clearance(pose(k,t));samples.append((k,t/T,cached[key]))
        return cached[key]
    def leaf(k,a,b,depth):
        mid=(a+b)/2
        if g.queries-start_queries>=max_queries:
            unresolved.append((k,mid/T,float('-inf'),'query_budget'));return
        d=dist(k,mid)
        if d < -g.epsilon:
            bad.append((k,mid/T,d,'strict_interior_overlap'));return
        if model=='constant_control_arc':B=motion_bound(X[:,k],U[:,k],a,mid,b,g.ego,L)
        else:
            dp=np.linalg.norm(X[:2,k+1]-X[:2,k]);dth=abs(math.atan2(math.sin(X[2,k+1]-X[2,k]),math.cos(X[2,k+1]-X[2,k])))
            B=(dp+r*dth)*(b-a)/(2*T)
        lower=d-B-g.epsilon
        if lower>0:lowers.append(lower);return
        if depth>=max_depth:
            unresolved.append((k,mid/T,lower,'depth_or_contact'));return
        leaf(k,a,mid,depth+1);leaf(k,mid,b,depth+1)
    node_min=float('inf')
    for k in range(n):
        for t in (0.,T):
            if g.queries-start_queries>=max_queries:
                unresolved.append((k,t/T,float('-inf'),'query_budget'));continue
            d=dist(k,t);node_min=min(node_min,d)
            if d < -g.epsilon:bad.append((k,t/T,d,'endpoint_overlap'))
        cuts=[0.]+(speed_roots(X[3,k],U[1,k],T) if model=='constant_control_arc' else [])+[T]
        for a,b in zip(cuts,cuts[1:]):leaf(k,a,b,0)
    state='COLLISION' if bad else 'UNKNOWN' if unresolved else 'PASS'
    return {'state':state,'model':model,'certificate_label':'conditional_numerical_not_formal_fp','node_min':node_min,
        'checked_min':min((s[2] for s in samples),default=None),'certified_lower_bound':min(lowers) if state=='PASS' and lowers else None,
        'numerical_allowance_m':g.epsilon,'queries':g.queries-start_queries,'witnesses':bad,'unresolved':unresolved,
        'suggested_samples':sorted(bad+unresolved,key=lambda z:(z[3] not in ['strict_interior_overlap','endpoint_overlap'],z[2]))}


def validate(X,U,T,scene,g,config):
    X=np.asarray(X);U=np.asarray(U);L=scene.veh.L
    Q=rollout(np.r_[scene.start,0.],U,T,L)
    pred=np.column_stack([arc(X[:,k],U[:,k],T,L) for k in range(U.shape[1])]);res=float(np.max(np.abs(pred-X[:,1:])))
    drift=float(np.max(np.linalg.norm(Q[:2]-X[:2],axis=0)));heading_drift=float(np.max(np.abs(Q[2]-X[2])))
    start=float(np.max(np.abs(X[:,0]-np.r_[scene.start,0.])))
    goal=np.abs(Q[:3,-1]-scene.goal);goal[2]=abs(math.atan2(math.sin(Q[2,-1]-scene.goal[2]),math.cos(Q[2,-1]-scene.goal[2])))
    terminal=bool(np.max(goal[:2])<=config['goal_xy_component_tol'] and goal[2]<=config['goal_heading_tol'] and abs(Q[3,-1])<=config['goal_speed_tol'])
    limits={'speed':float(np.max(np.abs(X[3]))),'accel':float(np.max(np.abs(U[1]))),'steering':float(np.max(np.abs(U[0]))),
        'steering_delta':float(np.max(np.abs(np.diff(U[0])))) if U.shape[1]>1 else 0.}
    tol=config['control_numeric_tol']
    controls=limits['speed']<=config['v_max']+tol and limits['accel']<=config['a_max']+tol and limits['steering']<=math.atan(L/scene.veh.Rmin)+tol and limits['steering_delta']<=config['ddelta_max']*T+tol
    b=scene.XYbounds
    bounds=bool(np.min(X[0])>=b[0]+1-tol and np.max(X[0])<=b[1]-1+tol and np.min(X[1])>=b[2]-tol and np.max(X[1])<=b[3]-1+tol)
    dynamic=bool(start<=config['start_tol'] and res<=config['dynamics_residual_max'] and drift<=config['global_rollout_position_drift_max_m'] and heading_drift<=config['global_rollout_heading_drift_max_rad'] and controls and bounds)
    saved_min=min(g.clearance(q) for q in X[:3].T)
    result=check_curve(Q,U,T,g,L,max_depth=config['certificate_max_depth'],max_queries=config['certificate_max_queries'])
    node=bool(saved_min>=.05 and result['node_min']>=.05)
    checks={'start':start<=config['start_tol'],'dynamics_residual':res<=config['dynamics_residual_max'],
        'rollout_position':drift<=config['global_rollout_position_drift_max_m'],'rollout_heading':heading_drift<=config['global_rollout_heading_drift_max_rad'],
        'controls':controls,'reference_bounds':bounds,'goal_position':np.max(goal[:2])<=config['goal_xy_component_tol'],
        'goal_heading':goal[2]<=config['goal_heading_tol'],'terminal_speed':abs(Q[3,-1])<=config['goal_speed_tol'],
        'whole_trajectory':result['state']=='PASS'}
    return {'accepted':bool(dynamic and terminal and result['state']=='PASS'),'node_pass':node,'dynamics_pass':dynamic,'terminal_pass':terminal,
        'acceptance_version':config['acceptance_version'],'checks':checks,'failure_items':[k for k,v in checks.items() if not v],
        'saved_node_min':saved_min,'global_node_min':result['node_min'],'endpoint_residual_max':res,'global_position_drift_m':drift,'global_heading_drift_rad':heading_drift,
        'start_residual_max':start,'terminal_component_errors':goal.tolist(),'terminal_speed':float(Q[3,-1]),'control_extrema':limits,'reference_bounds_pass':bounds,'curve':result}
