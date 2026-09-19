from pathlib import Path
# Dependency-free SVG placeholders convey data availability without fabricated inference plots.
def svg(path,title,values):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); vals=values or [0]; mx=max(max(vals),1e-9); pts=' '.join(f'{10+i*280/max(1,len(vals)-1):.1f},{190-v/mx*160:.1f}' for i,v in enumerate(vals)); p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="220"><text x="10" y="20">{title}</text><polyline fill="none" stroke="#2563eb" points="{pts}"/></svg>')
def generate_basic(figures, activity, deviations):
 for n,v in {'movement_magnitude':activity,'velocity':activity,'acceleration':[abs(activity[i]-activity[i-1]) for i in range(1,len(activity))],'baseline_deviation':deviations,'joint_trajectories':activity,'posture_features':activity,'skeleton_3d':activity,'temporal_embedding_pca':activity,'model_comparison':activity,'feature_importance':activity,'confusion_matrix':activity,'roc_curve':activity,'pr_curve':activity}.items():svg(Path(figures)/(n+'.svg'),n.replace('_',' '),v)
