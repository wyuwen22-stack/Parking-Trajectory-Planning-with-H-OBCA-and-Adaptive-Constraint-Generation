"""Version-frozen128 experiments: cached backend and fresh HA pipelines separate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,concurrent.futures,csv,json,math,subprocess,sys,time,hashlib
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from .stage import ROOT,INPUT,save,config,code_hash
from .geometry import Geometry
from .validator import check_curve
from traj_py.geometry import densify_poses
from traj_py.hybrid_astar import plan_hybrid_astar,SearchStopped
from .outcome import outcome
from traj_py.warm_start import choose_parking_horizon,make_warm_start
from eval_parking.scenarios import (CaseSpec,iter_suite_specs,iter_diagonal_specs,iter_irregular_specs,iter_tpcap_specs,dissertation_regression_specs,scene_for_spec,RECTANGLE_PLANNER_SETTINGS)
BASE=Path(os.environ.get('PARKING_SAFE_OUTPUT',str(ROOT/'experiments'))).resolve()

def specs():return iter_suite_specs(('h_obca',))+iter_diagonal_specs(('h_obca',))+iter_irregular_specs(('h_obca',))+iter_tpcap_specs(('h_obca',))
def key(spec):return spec.case_id.removeprefix('h_obca_')
def get_spec(identifier):return next(s for s in specs()+dissertation_regression_specs(('h_obca',)) if key(s)==identifier)
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    ss=specs();assert len(ss)==128 and len({key(s) for s in ss})==128 and not any(s.is_regression for s in ss)
    freeze={'code_sha256':code_hash(),'configuration':config(),'experiments':list(config()['experiment_variants']),'seeded_source':'frozen R395 HA paths; failed searches stay WS; shared by all variants; no fresh replacement','fresh_source':'one strict new HA search per case, shared by variants; failure stays WS; no cached fallback','case_ids':[key(s) for s in ss]}
    path=BASE/'freeze.json'
    if path.exists():assert json.loads(path.read_text())==freeze,'frozen code/config changed'
    else:save(path,freeze)
    save(BASE/'specs128.json',[dict(asdict(s),start=s.start.tolist()) for s in ss])

def pack(identifier,source,out):
    s=get_spec(identifier);scene=scene_for_spec(s);cfg=config();t=time.monotonic()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'scene.npz',start=scene.start,goal=scene.goal,bounds=scene.XYbounds,ego=scene.veh.ego,L=scene.veh.L,Rmin=scene.veh.Rmin,**{f'obs_{i}':o for i,o in enumerate(scene.obs)})
    if source=='cached':
        raise ValueError('Legacy cached paths used 5 m. Use seeded for the frozen 3.95 m HA experiment.')
    elif source=='seeded':
        seed=INPUT/'ha_r395'/'cases'/identifier
        old=json.loads((seed/'result.json').read_text(encoding='utf-8'))
        if old['search_status']!='PATH':
            save(out/'ha.json',{'status':'WS_'+old['search_status'],'source':'seeded_r395','seed_record':old,'elapsed_seconds':0,'stop_reason':'CACHED_'+old['search_status'],'source_time_limit_s':90.0});return
        p=seed/'path.npz'
        if file_sha(p)!=old['path_sha256']:raise ValueError('Frozen HA path hash mismatch')
        with np.load(p,allow_pickle=False) as z:
            if not (np.isclose(z['Rmin'],scene.veh.Rmin) and np.isclose(z['L'],scene.veh.L) and np.allclose(z['ego'],scene.veh.ego)):
                raise ValueError('Frozen HA vehicle differs from current contract')
            path=z['states'].copy()
        meta={'source_time_limit_s':90.0,'source':'seeded_r395','path_sha256':file_sha(p),'expanded':old['expanded'],'seed_returned_path_5cm_pass':old['accepted'],'seed_record':old}
    else:
        try:
            r=plan_hybrid_astar(scene,replace(RECTANGLE_PLANNER_SETTINGS,max_time=cfg['ha_search_seconds']))
            path=r.states;meta={'source_time_limit_s':cfg['ha_search_seconds'],'source':'fresh_strict005_HA','expanded':r.expanded,'cost':r.cost}
        except SearchStopped as e:
            save(out/'ha.json',{'status':'WS_SEARCH','stop_reason':e.reason,'expanded':e.expanded,'source':'fresh','error':str(e),'elapsed_seconds':time.monotonic()-t});return
        except Exception as e:
            save(out/'ha.json',{'stop_reason':'INPUT_REJECTED' if isinstance(e,ValueError) else 'SEARCH_EXCEPTION','exception_type':type(e).__name__,'status':'WS_SEARCH','source':'fresh','error':str(e),'elapsed_seconds':time.monotonic()-t});return
    g=Geometry(scene.obs,scene.veh.ego);node_min=min(g.clearance(q) for q in path);linear_min=min(g.clearance(q) for q in densify_poses(path,scene.veh.ego));err=np.abs(path[-1]-scene.goal);err[2]=abs(math.atan2(math.sin(path[-1,2]-scene.goal[2]),math.cos(path[-1,2]-scene.goal[2])))
    h=choose_parking_horizon(path);p=make_warm_start(path,scene,True,True,h).problem
    # A common immutable pack for all three variants; the X field sets horizon only.
    np.savez_compressed(out/'input.npz',ha_states=path,X=p.x_initial,U=p.u_initial,start=scene.start,goal=scene.goal,bounds=scene.XYbounds,ego=scene.veh.ego,L=scene.veh.L,Rmin=scene.veh.Rmin,**{f'obs_{i}':o for i,o in enumerate(scene.obs)})
    X=np.vstack((path.T,np.zeros(len(path))));curve=check_curve(X,np.zeros((2,len(path)-1)),.4,g,model='linear_pose')
    from .motion import arc
    initial_res=max(float(np.max(abs(arc(p.x_initial[:,i],p.u_initial[:,i],.4)-p.x_initial[:,i+1]))) for i in range(h))
    save(out/'ha.json',dict(meta,status='PATH',node_min=node_min,linear_sampled_min=linear_min,geometric_main=bool(node_min>=.05 and linear_min>0 and max(err)<=.05),geometric_conditional_state=curve['state'],geometric_conditional_lower=curve['certified_lower_bound'],horizon=h,Ts=.4,initial_arc_residual_before_clip=initial_res,initial_accel_max=float(abs(p.u_initial[1]).max()),elapsed_seconds=time.monotonic()-t,prepared_input_sha256=file_sha(out/'input.npz'),terminal_error=err.tolist()))

def child(cmd,out,timeout):
    t=time.monotonic()
    with (out/'process.log').open('w') as log:
        try:
            p=subprocess.run([sys.executable,'-m','safe_parking.suite']+cmd,env=dict(os.environ,PYTHONPATH=str(ROOT/'python_scripts'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            return None if p.returncode==0 else outcome(None,'WORKER_ERROR',None,returncode=p.returncode,elapsed_seconds=time.monotonic()-t)
        except subprocess.TimeoutExpired:return outcome(None,'PROCESS_TIME_LIMIT',None,elapsed_seconds=time.monotonic()-t)

def case(identifier,source):
    out=BASE/source/identifier;out.mkdir(parents=True,exist_ok=True);cfg=config();stamp=code_hash()
    if not (out/'ha.json').exists():
        err=child(['pack','--id',identifier,'--source',source],out,cfg['ha_hard_timeout_seconds'])
        if err or not (out/'ha.json').exists():save(out/'ha.json',err or outcome(None,'MISSING_OUTPUT',None))
    ha=json.loads((out/'ha.json').read_text())
    for variant in cfg['experiment_variants']:
        v=out/variant;v.mkdir(exist_ok=True)
        if (v/'result.json').exists():continue
        if ha['status']!='PATH':save(v/'result.json',outcome(None,ha.get('stop_reason',ha['status']),None,ha_status=ha['status'],code_sha256=stamp,source=source,upstream_failure=True));continue
        err=child(['opt','--id',identifier,'--source',source,'--variant',variant],v,cfg['case_hard_timeout_seconds'])
        if err or not (v/'result.json').exists():save(v/'result.json',err or outcome(None,'MISSING_OUTPUT',None))
    print('CASE_DONE',source,identifier,ha['status'],[json.loads((out/v/'result.json').read_text())['status'] for v in cfg['experiment_variants']],flush=True)
    return identifier

def summarize(source):
    rows=[];ss=specs()
    for s in ss:
        identifier=key(s);d=BASE/source/identifier
        if not (d/'ha.json').exists():continue
        ha=json.loads((d/'ha.json').read_text())
        for variant in config()['experiment_variants']:
            p=d/variant/'result.json'
            if not p.exists():continue
            r=json.loads(p.read_text());v=r.get('validation') or {};c=v.get('curve',{});h=r.get('history',[])
            ok=bool(h) and h[-1]['status'] in ['Solve_Succeeded','Solved_To_Acceptable_Level'];gate=ok and v.get('dynamics_pass',False) and v.get('terminal_pass',False)
            linear=h[-1].get('historical_linear_sampled_swept_min') if h else None
            rows.append({'id':identifier,'family':s.family,'source':source,'variant':variant,'status':r['status'],'solver_status':r.get('solver_status'),'stop_reason':r.get('stop_reason'),'validation_status':r.get('validation_status'),'failure_items':r.get('failure_items',[]),'terminal_speed':v.get('terminal_speed'),'acceptance_version':config()['acceptance_version'],'conditional_arc_accept':r.get('accepted',False),'arc_geometry_state':c.get('state'),'node_pass':bool(gate and v.get('node_pass',False)),'linear_sampled_main':bool(gate and v.get('saved_node_min',-1)>=.05 and linear is not None and linear>0),'linear_double005':bool(gate and v.get('saved_node_min',-1)>=.05 and linear is not None and linear>=.05),'arc_checked_min':c.get('checked_min'),'conditional_lower':c.get('certified_lower_bound'),'node_min':v.get('saved_node_min'),'linear_sampled_min':linear,'dynamics_pass':v.get('dynamics_pass'),'terminal_pass':v.get('terminal_pass'),'ha_geometric_main':ha.get('geometric_main',False),'ha_status':ha['status'],'ha_source':ha.get('source'),'input_sha256':ha.get('prepared_input_sha256'),'code_sha256':code_hash(),'rounds':len(h),'time_seconds':r.get('elapsed_seconds')})
    count={}
    for variant in config()['experiment_variants']:
        rr=[r for r in rows if r['variant']==variant];count[variant]={'completed':len(rr),'formal_denominator':128,'node':sum(r['node_pass'] for r in rr),'linear_sampled_main':sum(r['linear_sampled_main'] for r in rr),'linear_double005':sum(r['linear_double005'] for r in rr),'conditional_arc_accept':sum(r['conditional_arc_accept'] for r in rr),'states':{state:sum(r['status']==state for r in rr) for state in sorted({r['status'] for r in rr})},'min_conditional_lower':min((r['conditional_lower'] for r in rr if r['conditional_arc_accept']),default=None)}
    save(BASE/source/'counts.json',count);save(BASE/source/'results.json',rows)
    if rows:
        target=BASE/source/'results.csv';tmp=target.with_suffix('.csv.tmp')
        with tmp.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        tmp.replace(target)
    return count

def run(source,workers):
    prepare();ids=[key(s) for s in specs()]
    print('START',source,'workers',workers,'pid',os.getpid(),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(case,i,source):i for i in ids}
        for f in concurrent.futures.as_completed(tasks):
            try:f.result()
            except Exception as e:print('CASE_EXCEPTION',tasks[f],repr(e),flush=True)
            summarize(source)
    print('FINISH',source,json.dumps(summarize(source)),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','pack','opt','run','summary']);a.add_argument('--source',choices=['fresh','seeded'],default='seeded');a.add_argument('--workers',type=int,default=3);a.add_argument('--id');a.add_argument('--variant',choices=['node','dense','adaptive']);z=a.parse_args()
    if z.action=='prepare':prepare()
    elif z.action=='pack':pack(z.id,z.source,BASE/z.source/z.id)
    elif z.action=='opt':
        from .stage import worker
        worker(str(BASE/z.source/z.id/'input.npz'),BASE/z.source/z.id/z.variant,z.variant)
    elif z.action=='summary':print(json.dumps(summarize(z.source),ensure_ascii=False))
    else:run(z.source,z.workers)
