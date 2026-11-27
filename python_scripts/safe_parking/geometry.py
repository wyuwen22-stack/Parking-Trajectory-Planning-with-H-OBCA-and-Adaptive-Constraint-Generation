"""Independent float64 Euclidean/SAT validator, strict convex decomposition.

No solver duals or inherited collision flags are used. Numeric guards are
engineering allowances, not interval-arithmetic error proofs.
"""
import numpy as np

def vertices(poly):
    p=np.asarray(poly,dtype=float)
    if len(p)>1 and np.linalg.norm(p[0]-p[-1])<1e-10:p=p[:-1]
    if len(p)<3 or not np.isfinite(p).all():raise ValueError('invalid polygon')
    return p

def area(p):
    q=vertices(p);q=q-q[0]
    return float(np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))/2)

def cross(a,b):return a[0]*b[1]-a[1]*b[0]

def decompose(poly):
    p=vertices(poly).copy()
    if area(p)<0:p=p[::-1]
    if all(cross(p[(i+1)%len(p)]-p[i],p[(i+2)%len(p)]-p[(i+1)%len(p)])>=-1e-12 for i in range(len(p))):return [p]
    original=area(p);out=[]
    while len(p)>3:
        found=False
        for i in range(len(p)):
            a,b,c=p[i-1],p[i],p[(i+1)%len(p)]
            if cross(b-a,c-b)<=1e-12:continue
            other=np.delete(p,[(i-1)%len(p),i,(i+1)%len(p)],axis=0)
            inside=lambda q:cross(b-a,q-a)>=-1e-12 and cross(c-b,q-b)>=-1e-12 and cross(a-c,q-c)>=-1e-12
            if any(inside(q) for q in other):continue
            out.append(np.array([a,b,c]));p=np.delete(p,i,axis=0);found=True;break
        if not found:raise ValueError('non-simple or degenerate polygon; refusing unsafe decomposition')
    out.append(p)
    if abs(sum(area(q) for q in out)-original)>1e-8*max(1,original):raise ValueError('decomposition area mismatch')
    return out

def body(q,ego):
    front,left,rear,right=ego;v=np.array([[front,left],[front,-right],[-rear,-right],[-rear,left]])
    c,s=np.cos(q[2]),np.sin(q[2]);return v@np.array([[c,s],[-s,c]])+np.asarray(q[:2])

def convex_signed_distance(a,b):
    a,b=vertices(a),vertices(b);origin=a[0].copy();a=a-origin;b=b-origin
    axes=[]
    for p in (a,b):
        e=np.roll(p,-1,axis=0)-p;n=np.column_stack((-e[:,1],e[:,0]));length=np.linalg.norm(n,axis=1)
        if np.any(length<1e-13):raise ValueError('zero polygon edge')
        axes.extend(n/length[:,None])
    axes=np.array(axes);pa=a@axes.T;pb=b@axes.T
    gaps=np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0))
    if gaps.max()<=0:return float(gaps.max())
    best=float('inf')
    for src,dst in ((a,b),(b,a)):
        e=np.roll(dst,-1,axis=0)-dst;v=src[:,None,:]-dst[None,:,:]
        t=np.clip(np.sum(v*e[None,:,:],axis=2)/np.sum(e*e,axis=1)[None,:],0,1)
        d=np.linalg.norm(v-t[:,:,None]*e[None,:,:],axis=2);best=min(best,float(d.min()))
    return best

class Geometry:
    def __init__(self,obstacles,ego):
        self.ego=np.asarray(ego);self.pieces=[p for o in obstacles for p in decompose(o)]
        self.queries=0
        self.scale=max([1.0]+[float(np.max(np.abs(p))) for p in self.pieces])
        self.epsilon=max(1e-7,64*np.finfo(float).eps*self.scale)
    def clearance(self,q):
        self.queries+=1;a=body(q,self.ego)
        return min((convex_signed_distance(a,b) for b in self.pieces),default=float('inf'))
