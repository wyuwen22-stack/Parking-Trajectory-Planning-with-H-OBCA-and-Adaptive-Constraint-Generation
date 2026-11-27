"""Explicit new arc mode. Historical traj_py.obca.solve_obca is unmodified."""
import casadi as ca
import numpy as np
from traj_py.obca import ObcaResult
from .motion import arc_symbolic


class SolveFailure(RuntimeError):
    def __init__(self, message, stats):
        super().__init__(message)
        self.stats = stats


def ipopt_options(config):
    options = {'print_level':0,'max_iter':1500,'tol':1e-8,'constr_viol_tol':1e-8,
               'acceptable_tol':1e-6,'bound_relax_factor':0}
    if config['solver_max_cpu_seconds_per_round'] is not None:
        options['max_cpu_time'] = config['solver_max_cpu_seconds_per_round']
    return options


def terminal_bands(config):
    xy = config['goal_xy_component_tol'] - config['terminal_xy_inset_m']
    heading = config['goal_heading_tol'] - config['terminal_heading_inset_rad']
    if not (0 < xy <= config['goal_xy_component_tol'] and 0 < heading <= config['goal_heading_tol']):
        raise ValueError('Terminal inset must leave a positive interval')
    return (xy, xy, heading)


def solve_round(p, samples, config, previous=None):
    n=p.ref.shape[0]-1;o=ca.Opti();X=o.variable(4,n+1);U=o.variable(2,n)
    # Frozen historical once objective; no obstacle-distance term.
    J=0.5*ca.sumsqr(U[0,:])+0.1*ca.sumsqr(U[1,:])
    if n>1:J+=0.2*ca.sumsqr(U[0,1:]-U[0,:-1])+2*ca.sumsqr(U[1,1:]-U[1,:-1])+ca.sumsqr(X[3,1:n]-X[3,:n-1])
    J+=0.3*ca.sumsqr(X[:2,:]-p.x_initial[:2,:]);o.minimize(J)
    for k in range(n):o.subject_to(X[:,k+1]==arc_symbolic(X[:,k],U[:,k],p.Ts,p.L))
    o.subject_to(X[:,0]==np.r_[p.start,0.])
    for q,band in enumerate(terminal_bands(config)):o.subject_to(o.bounded(p.goal[q]-band,X[q,n],p.goal[q]+band))
    o.subject_to(X[3,n]==config['terminal_speed_target'])
    o.subject_to(o.bounded(-p.v_max,X[3,:],p.v_max));o.subject_to(o.bounded(-p.delta_max,U[0,:],p.delta_max));o.subject_to(o.bounded(-p.a_max,U[1,:],p.a_max))
    if n>1:o.subject_to(o.bounded(-p.ddelta_max*p.Ts,U[0,1:]-U[0,:-1],p.ddelta_max*p.Ts))
    o.subject_to(o.bounded(p.bounds[0]+1,X[0,:],p.bounds[1]-1));o.subject_to(o.bounded(p.bounds[2],X[1,:],p.bounds[3]-1))
    poses=[X[:3,k] for k in range(n+1)]+[arc_symbolic(X[:,k],U[:,k],f*p.Ts,p.L)[:3] for k,f in sorted(samples)]
    for A,b in zip(p.Aob,p.bob):
        lm=o.variable(A.shape[0],len(poses));mu=o.variable(p.G.shape[0],len(poses));o.subject_to(ca.vec(lm)>=0);o.subject_to(ca.vec(mu)>=0)
        for j,q in enumerate(poses):
            th=q[2];R=ca.vertcat(ca.horzcat(ca.cos(th),-ca.sin(th)),ca.horzcat(ca.sin(th),ca.cos(th)))
            l,u=lm[:,j],mu[:,j]
            target=(.05 if j<=n else config['interior_constraint_clearance_m'])+config['node_constraint_buffer_m']
            o.subject_to(-ca.DM(p.g).T@u+(A@q[:2]-b).T@l>=target)
            o.subject_to(p.G.T@u+R.T@(A.T@l)==0);o.subject_to(ca.sumsqr(A.T@l)<=1)
        o.set_initial(lm,.1);o.set_initial(mu,.1)
    o.set_initial(X,p.x_initial if previous is None else previous.X);o.set_initial(U,p.u_initial if previous is None else previous.U)
    o.solver('ipopt',{'print_time':False,'ipopt':ipopt_options(config)})
    try:
        sol=o.solve_limited()
    except RuntimeError as exc:
        raise SolveFailure(str(exc),o.stats()) from exc
    stats=o.stats()
    return ObcaResult(np.asarray(sol.value(X)),np.asarray(sol.value(U)),float(sol.value(J)),int(stats['iter_count']),stats['return_status']),{'configuration_nodes':n+1,'interior_pose_constraints':len(samples),'obstacle_pairs':len(poses)*len(p.Aob),'decision_variables':int(o.nx),'constraints':int(o.ng)}
