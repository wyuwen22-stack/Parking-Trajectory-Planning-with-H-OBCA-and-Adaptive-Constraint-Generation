"""Local resumable stage1 runner; only writes inside this independent workspace."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,hashlib,json,math,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from .geometry import Geometry,decompose
from .motion import rollout
from .validator import check_curve,validate
from .solver import solve_round
from .outcome import outcome, solver_stop
from traj_py.scenario import Vehicle,Scenario,_poly2hrep
from traj_py.warm_start import make_warm_start
from parking_config import read_config
ROOT=Path(__file__).resolve().parents[2];INPUT=ROOT/'inputs'

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)

def config():return read_config()
def code_hash():
    ps=sorted(p for p in (ROOT/'python_scripts').rglob('*.py') if not p.name.startswith('view_'))+[ROOT/'contract.json',ROOT/'manifest128.json']
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in ps)).hexdigest()

def sources():return list(csv.DictReader((INPUT/'collision_segments.csv').open(encoding='utf-8-sig')))
def load(file):
    z=np.load(INPUT/'paths'/file);obs=[z[k] for k in z.files if k.startswith('obs_')];pieces=[np.vstack((p,p[0])) for o in obs for p in decompose(o)];hs=[_poly2hrep(p) for p in pieces]
    # Metadata-free archives are the historical 5 m model; never relabel them.
    L=float(z['L']) if 'L' in z else 2.7;Rmin=float(z['Rmin']) if 'Rmin' in z else 5.
    scene=Scenario(Vehicle(L,z['ego'].copy(),Rmin),z['bounds'].copy(),z['start'].copy(),z['goal'].copy(),tuple(pieces),tuple(a for a,b in hs),tuple(b for a,b in hs),np.zeros((1,1),bool),5.)
    return z,scene

def prepare():
    allc=json.loads((INPUT/'scene_catalog_129.json').read_text());formal=[c for c in allc if not c.get('regression',False)];reg=[c for c in allc if c.get('regression',False)]
    assert len(formal)==128 and len(reg)==1
    save(ROOT/'manifest128.json',formal);save(ROOT/'regression_manifest.json',reg);save(ROOT/'overlap18_manifest.json',sources())

def replay():
    out=[]
    for r in sources():
        z,scene=load(r['file']);g=Geometry(scene.obs,scene.veh.ego)
        lin=check_curve(z['X'],z['U'],.4,g,model='linear_pose')
        loc=check_curve(z['X'],z['U'],.4,g,model='constant_control_arc')
        out.append({'file':r['file'],'ha_old_double005':r['ha_strict_pass']=='True','historical_linear':lin,'local_arc_reset_each_node':loc})
        print('REPLAY',r['file'],lin['state'],loc['state'],flush=True)
    assert len(out)==18 and all(r['historical_linear']['state']=='COLLISION' and r['local_arc_reset_each_node']['state']=='COLLISION' for r in out)
    save(ROOT/'tests_output/replay18.json',out)

def worker(file,out,variant):
    cfg=config();out=Path(out);out.mkdir(parents=True,exist_ok=True);z,scene=load(file)
    if not (np.isclose(scene.veh.Rmin,cfg['vehicle']['Rmin']) and np.isclose(scene.veh.L,cfg['vehicle']['L'])):
        raise ValueError('Input vehicle differs from contract.json; prepare a new input, do not reuse legacy 5 m packs.')
    path=z['ha_states'];n=z['X'].shape[1]-1
    p=make_warm_start(path,scene,parking=True,collision=True,horizon=n).problem;p.parking_objective='once';p.dmin=.05
    p.x_initial[3,0]=0.;p.x_initial[3,-1]=0.
    p.x_initial[3]=np.clip(p.x_initial[3],-p.v_max,p.v_max);p.u_initial[0]=np.clip(p.u_initial[0],-p.delta_max,p.delta_max);p.u_initial[1]=np.clip(p.u_initial[1],-p.a_max,p.a_max)
    samples={(k,float(f)) for k in range(n) for f in (cfg['dense_reference_fractions'] if variant=='dense' else [] if variant=='node' else [.5])}
    introduced={q:1 for q in samples}
    history=[];previous=None;t0=time.monotonic();final=None
    for turn in range(cfg['adaptive_max_rounds'] if variant=='adaptive' else 1):
        t=time.monotonic()
        try:
            result,structure=solve_round(p,samples,cfg,previous)
        except Exception as e:
            stats=getattr(e,'stats',{});raw=stats.get('return_status')
            final=outcome(raw,solver_stop(raw) if raw else 'SOLVER_EXCEPTION',None,error=str(e),solver_stats=stats);break
        solve_seconds=time.monotonic()-t
        np.savez_compressed(out/f'round{turn+1:02}.npz',X=result.X,U=result.U,Ts=p.Ts,ha_states=path,start=scene.start,goal=scene.goal,ego=scene.veh.ego,bounds=scene.XYbounds,L=scene.veh.L,Rmin=scene.veh.Rmin,delta_max=p.delta_max,constraint_samples=np.asarray(sorted(samples),dtype=float).reshape(-1,2),constraint_added_round=np.asarray([introduced[q] for q in sorted(samples)],dtype=int),**{f'obs_{i}':o for i,o in enumerate(scene.obs)})
        t=time.monotonic();g=Geometry(scene.obs,scene.veh.ego);ver=validate(result.X,result.U,p.Ts,scene,g,cfg);validation_seconds=time.monotonic()-t
        # Linear-pose finite samples retain the historical metric label, never the certificate label.
        from traj_py.geometry import densify_poses
        linear_poses=densify_poses(result.X[:3].T,scene.veh.ego)
        linear_min=min(g.clearance(q) for q in linear_poses)
        h={'round':turn+1,'status':result.status,'iterations':result.iterations,'objective':result.objective,'solve_seconds':solve_seconds,'validation_seconds':validation_seconds,'distance_queries':g.queries,'structure':structure,'samples':sorted(samples),'validation':ver,
           'historical_linear_sampled_swept_min':linear_min,'linear_sampled_main':bool(ver['dynamics_pass'] and ver['terminal_pass'] and ver['saved_node_min']>=.05 and linear_min>0)}
        h['added_samples']=[list(q) for q in sorted(samples) if introduced[q]==turn+1]
        history.append(h);save(out/f'round{turn+1:02}.json',h);save(out/'progress.json',{'file':file,'history':history,'code_sha256':code_hash()})
        print('ROUND',turn+1,result.status,'cert',ver['curve']['state'],'accept',ver['accepted'],'samples',len(samples),'seconds',round(solve_seconds+validation_seconds,2),flush=True)
        if result.status not in ['Solve_Succeeded','Solved_To_Acceptable_Level']:
            final=outcome(result.status,solver_stop(result.status),ver);break
        if ver['accepted']:
            final=outcome(result.status,'ACCEPTED',ver);break
        if variant!='adaptive' or turn+1>=cfg['adaptive_max_rounds']:
            final=outcome(result.status,'METHOD_COMPLETED' if variant!='adaptive' else 'REFINEMENT_LIMIT',ver);break
        suggestions=[]
        for k,f,d,reason in ver['curve']['suggested_samples']:
            q=(int(k),round(float(f),12))
            if 0<q[1]<1 and q not in samples and q not in suggestions:suggestions.append(q)
            if len(suggestions)>=cfg['adaptive_max_new_samples_per_round']:break
        if not suggestions:
            final=outcome(result.status,'NO_NEW_SAMPLE',ver);break
        introduced.update({q:turn+2 for q in suggestions});samples.update(suggestions);previous=result
    if final is None:
        ver=history[-1]['validation'] if history else None
        final=outcome(history[-1]['status'] if history else None,'REFINEMENT_LIMIT',ver)
    final.update({'file':str(file),'variant':variant,'accepted':final['status']=='PASS_CONDITIONAL_NUMERICAL','history':history,'elapsed_seconds':time.monotonic()-t0,'warmstart':'shared HA geometry; historical N/Ts; clipped velocity/control initial guess, initial/final v=0; no feasible-initialization guarantee','N':n,'code_sha256':code_hash(),'regression':str(file).endswith('_dissertation.npz'),'vehicle':{'L':scene.veh.L,'Rmin':scene.veh.Rmin,'ego':scene.veh.ego.tolist(),'delta_max_rad':p.delta_max},'motion_model':'constant_control_arc','acceptance_version':cfg['acceptance_version']})
    save(out/'result.json',final)

def batch(runname,variant,include_regression=False,only=None):
    folder=ROOT/'runs'/runname;folder.mkdir(parents=True,exist_ok=True);cfg=config();stamp=code_hash()
    p=folder/'run_fingerprint.json'
    if p.exists():assert json.loads(p.read_text())['code_sha256']==stamp,'code changed; use a new run name'
    else:save(p,{'code_sha256':stamp,'variant':variant,'config':cfg})
    files=[r['file'] for r in sources()]
    if include_regression:files.append('h_obca_perpendicular_tight_dissertation.npz')
    if only:files=[f for f in files if only in f]
    env=dict(os.environ,PYTHONPATH=str(ROOT/'python_scripts'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for i,file in enumerate(files):
        out=folder/file.removesuffix('.npz');out.mkdir(exist_ok=True)
        if (out/'result.json').exists():print('RESUME_SKIP',file,flush=True);continue
        print(f'CASE {i+1}/{len(files)} {file}',flush=True);t=time.monotonic()
        cmd=[sys.executable,'-m','safe_parking.stage','worker','--file',file,'--out',str(out),'--variant',variant]
        with (out/'worker.log').open('w') as log:
            try:
                proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=cfg['case_hard_timeout_seconds'])
                if proc.returncode or not (out/'result.json').exists():save(out/'result.json',{'file':file,'status':'WORKER_ERROR','returncode':proc.returncode,'accepted':False,'regression':file.endswith('_dissertation.npz'),'code_sha256':stamp,'elapsed_seconds':time.monotonic()-t})
            except subprocess.TimeoutExpired:
                save(out/'result.json',{'file':file,'status':'HARD_TIMEOUT','accepted':False,'regression':file.endswith('_dissertation.npz'),'code_sha256':stamp,'elapsed_seconds':time.monotonic()-t,'partial_progress':'progress.json and round*.npz retained if available'})
        r=json.loads((out/'result.json').read_text());print('RESULT',file,r['status'],round(time.monotonic()-t,2),flush=True)
    summarize(folder)

def summarize(folder):
    result=[];old={r['file']:r for r in sources()}
    for p in sorted(Path(folder).glob('*/result.json')):
        r=json.loads(p.read_text());h=r.get('history',[]);v=r.get('validation',{});v=v or {};c=v.get('curve',{});o=old.get(r['file'],{})
        result.append({'file':r['file'],'regression':r.get('regression',False),'HA_old_double005':o.get('ha_strict_pass',''), 'old_node_min':min(float(o.get('node_k_clearance_m','nan')),float(o.get('node_next_clearance_m','nan'))),'old_linear_penetration_m':o.get('linear_penetration_m',''),'status':r['status'],'accepted':r['accepted'],'rounds':len(h),'elapsed_seconds':r['elapsed_seconds'],'new_node_min':v.get('saved_node_min'),'new_global_node_min':v.get('global_node_min'),'new_arc_state':c.get('state'),'new_arc_conditional_lower_bound':c.get('certified_lower_bound'),'endpoint_residual':v.get('endpoint_residual_max'),'global_position_drift':v.get('global_position_drift_m'),'dynamics_pass':v.get('dynamics_pass'),'terminal_pass':v.get('terminal_pass'),'new_linear_sampled_main':h[-1].get('linear_sampled_main') if h else None,'new_linear_sampled_min':h[-1].get('historical_linear_sampled_swept_min') if h else None})
    save(Path(folder)/'summary.json',result)
    if result:
        with (Path(folder)/'before_after.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    print('SUMMARY',len(result),'formal18',sum(not r['regression'] for r in result),'accepted',sum(r['accepted'] for r in result if not r['regression']),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','replay','run','worker','summarize']);a.add_argument('--run',default='arc_adaptive_v1');a.add_argument('--variant',choices=['adaptive','dense','node'],default='adaptive');a.add_argument('--file');a.add_argument('--out');a.add_argument('--include-regression',action='store_true');a.add_argument('--only');args=a.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='replay':replay()
    elif args.action=='worker':worker(args.file,args.out,args.variant)
    elif args.action=='summarize':summarize(ROOT/'runs'/args.run)
    else:batch(args.run,args.variant,args.include_regression,args.only)
