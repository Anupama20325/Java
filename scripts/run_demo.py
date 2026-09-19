#!/usr/bin/env python3
"""End-to-end engineering demo. Synthetic sequences are used ONLY for software-path tests."""
import sys,csv,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.adapters import auto_adapter
from src.preprocessing.sequence import normalize,windows
from src.features.behavioural import extract_features
from src.personalization.baseline import PersonalBaseline
from src.visualization.plots import generate_basic
from src.reporting.dashboard import dashboard

def synthetic_sequence():
 # Explicit test fixture, never an experimental participant or research result.
 return [[[j*.1+t*.01, j*.05, (t%9)*.01] for j in range(17)] for t in range(180)]
def write_csv(path,rows,fields):
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def run(root='', demo=True):
 out=Path('outputs'); fig=out/'figures'; res=out/'results'; rep=out/'reports'; mod=out/'models'
 for p in (fig,res,rep,mod):p.mkdir(parents=True,exist_ok=True)
 adapter=auto_adapter(root); found=adapter.discover(); source='local open-schema dataset' if found else 'no local dataset; synthetic sequence used only for software smoke-test'
 samples=[]
 if found:
  for item in found[:10]: samples.append(adapter.load_sequence(item['participant_id'],item['session_id']))
 else:samples=[{'sequence':synthetic_sequence(),'participant_id':'SYNTHETIC_TEST','session_id':'test-only','label':None}]
 rows=[]; activities=[]; all_devs=[]
 for s in samples:
  x,params=normalize(s['sequence']); ws=windows(x,150,75); fs=[]; summary={}
  for i,w in enumerate(ws):
   f,summary=extract_features(w); fs.append(f); activities.append(summary['activity']); rows.append({'participant_id':s['participant_id'],'session_id':s['session_id'],'window':i,'feature_vector':json.dumps(f),'activity':summary['activity'],'label':s['label'] or ''})
  devs,note=PersonalBaseline().within_session_proxy(fs); all_devs.extend(d['baseline_deviation_score'] for d in devs)
  dashboard(rep/f"{s['participant_id']}.html",s['participant_id'],summary,devs[-1]['baseline_deviation_score'])
 write_csv(res/'feature_summary.csv',rows,['participant_id','session_id','window','feature_vector','activity','label'])
 write_csv(res/'baseline_deviation.csv',[{'window':i,'baseline_deviation_score':v,'note':'Within-session baseline proxy; not longitudinal validation.'} for i,v in enumerate(all_devs)],['window','baseline_deviation_score','note'])
 no_real=not found; status='not_run_no_labelled_real_dataset' if no_real else 'requires >=2 labelled participants; run full pipeline with data'
 write_csv(res/'ablation_results.csv',[{'model':m,'status':status,'note':'No synthetic metrics are reported.'} for m in ['A statistical motion','B temporal LSTM','C motion+posture','D temporal+baseline','E graph-temporal experimental']],['model','status','note'])
 write_csv(res/'model_comparison.csv',[{'model':m,'status':status} for m in ['LogisticRegression','RandomForest','MLP','LSTM','GraphTemporal']],['model','status'])
 write_csv(res/'participant_predictions.csv',[{'participant_id':s['participant_id'],'model_estimated_screening_result':'Not trained; no labelled real dataset' if no_real else 'Not evaluated in demo'} for s in samples],['participant_id','model_estimated_screening_result'])
 metrics={'dataset_detected':source,'synthetic_test_only':no_real,'clinical_disclaimer':'Not a diagnostic system; no clinical or longitudinal validation claimed.','metrics':None if no_real else 'Run full training with adequate labelled participants.'}
 (res/'metrics.json').write_text(json.dumps(metrics,indent=2)); generate_basic(fig,activities,all_devs)
 print('Dataset detected:',source); print('Demo completed. Outputs:',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',default='');a=p.parse_args();run(a.dataset_root)
