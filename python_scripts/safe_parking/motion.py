"""Constant-control bicycle flow; no IO, no solver dependencies."""
import math
import numpy as np

def sinc(z):
    z=float(z)
    return 1-z*z/6+z**4/120-z**6/5040 if abs(z)<1e-4 else math.sin(z)/z

def arc(x,u,t,L=2.7):
    x=np.asarray(x,dtype=float);delta,a=map(float,u)
    s=x[3]*t+0.5*a*t*t;k=math.tan(delta)/L;h=0.5*k*s
    return np.array([x[0]+s*sinc(h)*math.cos(x[2]+h),x[1]+s*sinc(h)*math.sin(x[2]+h),x[2]+2*h,x[3]+a*t])

def arc_symbolic(x,u,t,L):
    import casadi as ca
    s=x[3]*t+0.5*u[1]*t*t;h=0.5*ca.tan(u[0])/L*s
    # Inactive division must also remain finite for AD at exactly zero curvature.
    den=ca.if_else(ca.fabs(h)<1e-4,1,h)
    sh=ca.if_else(ca.fabs(h)<1e-4,1-h*h/6+h**4/120-h**6/5040,ca.sin(h)/den)
    return ca.vertcat(x[0]+s*sh*ca.cos(x[2]+h),x[1]+s*sh*ca.sin(x[2]+h),x[2]+2*h,x[3]+u[1]*t)

def speed_roots(v,a,T):
    return [-v/a] if a!=0 and 0 < -v/a < T else []

def travel(v,a,t0,t1):
    points=[t0]+[t for t in speed_roots(v,a,t1) if t0<t<t1]+[t1]
    return sum(abs(v*(b-c)+0.5*a*(b*b-c*c)) for c,b in zip(points,points[1:]))

def motion_bound(x,u,t0,tm,t1,ego,L=2.7):
    r=math.hypot(max(ego[0],ego[2]),max(ego[1],ego[3]))
    return (1+r*abs(math.tan(u[0])/L))*max(travel(x[3],u[1],t0,tm),travel(x[3],u[1],tm,t1))

def rollout(x0,U,T,L=2.7):
    out=[np.asarray(x0,dtype=float)]
    for u in np.asarray(U).T:out.append(arc(out[-1],u,T,L))
    return np.asarray(out).T
