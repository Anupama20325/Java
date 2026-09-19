"""Optional PyTorch encoders; import is deferred so the data pipeline stays dependency-light."""
def select_device():
 try:
  import torch
  return 'cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu')
 except ImportError:return 'cpu'
def standard_edges(joints=17):
 return [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)] if joints==17 else [(i,i+1) for i in range(joints-1)]
def _torch():
 import torch
 from torch import nn
 return torch,nn
class TemporalEncoder:
 def __new__(cls,joints,spatial_hidden=64,temporal_hidden=64,embedding_dim=128,dropout=.2,task='classification'):
  torch,nn=_torch()
  class Net(nn.Module):
   def __init__(self):
    super().__init__(); self.spatial=nn.Sequential(nn.Linear(joints*3,spatial_hidden),nn.ReLU(),nn.Dropout(dropout)); self.rnn=nn.GRU(spatial_hidden,temporal_hidden,batch_first=True,bidirectional=True); self.embed=nn.Linear(temporal_hidden*2,embedding_dim); self.head=nn.Linear(embedding_dim,1)
   def forward(self,x):
    z=self.spatial(x.flatten(2)); z,_=self.rnn(z); e=self.embed(z[:,-1]); return self.head(e).squeeze(-1),e
  return Net()
class GraphTemporalEncoder:
 """Experimental graph-temporal baseline using adjacency aggregation then TemporalEncoder."""
 def __new__(cls,joints,edges=None,**kwargs):
  torch,nn=_torch(); base=TemporalEncoder(joints,**kwargs); edges=edges or standard_edges(joints)
  class Net(nn.Module):
   def __init__(self):
    super().__init__(); self.base=base; a=torch.eye(joints)
    for i,j in edges:a[i,j]=a[j,i]=1
    self.register_buffer('adj',a/a.sum(1,keepdim=True))
   def forward(self,x): return self.base(torch.einsum('ij,btjc->btic',self.adj,x))
  return Net()
class FaceEncoder: pass
class HeadEncoder: pass
class BodyEncoder: pass
class AudioEncoder: pass
class TextEncoder: pass
class ConcatenationFusion:
 def __call__(self, embeddings): return [v for x in embeddings for v in x]
class GatedFusion(ConcatenationFusion): pass
