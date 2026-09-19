import math

def clean_sequence(x):
 """Linear interpolation per coordinate; entirely absent coordinates become zero."""
 y=[[[float(v) for v in p] for p in f] for f in x]
 if not y or not y[0]: raise ValueError('sequence must have frames and joints')
 for j in range(len(y[0])):
  for c in range(3):
   valid=[i for i,f in enumerate(y) if math.isfinite(f[j][c])]
   if not valid:
    for f in y:f[j][c]=0.; continue
   for i in range(len(y)):
    if not math.isfinite(y[i][j][c]):
     lo=max((k for k in valid if k<i),default=valid[0]); hi=min((k for k in valid if k>i),default=valid[-1]);
     y[i][j][c]=y[lo][j][c] if lo==hi else y[lo][j][c]+(y[hi][j][c]-y[lo][j][c])*(i-lo)/(hi-lo)
 return y
def normalize(x, reference_joint=0, translation=True, scale=True, rotation=False):
 y=clean_sequence(x); params={'reference_joint':reference_joint,'translation':translation,'scale':scale,'rotation':rotation}
 if translation:
  for f in y:
   r=f[reference_joint];
   for p in f:
    for c in range(3):p[c]-=r[c]
 if scale:
  ds=[math.sqrt(sum(v*v for v in p)) for f in y for p in f if any(p)]; s=sum(ds)/len(ds) if ds else 1.; s=s or 1.; params['scale_factor']=s
  for f in y:
   for p in f:
    for c in range(3):p[c]/=s
 return y,params
def resample(x, frames):
 if frames<1: raise ValueError('frames must be positive')
 if len(x)==frames:return x
 return [x[round(i*(len(x)-1)/max(frames-1,1))] for i in range(frames)]
def windows(x,length=150,stride=75):
 if len(x)<length:return [resample(x,length)]
 out=[x[i:i+length] for i in range(0,len(x)-length+1,stride)]
 return out or [resample(x,length)]
