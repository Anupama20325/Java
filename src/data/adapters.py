"""Dataset-agnostic adapters; standardized samples use (T, J, 3) coordinates."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import csv, json

REQUIRED_COLUMNS={"participant_id","session_id","timestamp","joint_or_landmark_id","x","y","z"}
class BaseDatasetAdapter(ABC):
 def __init__(self, root=""): self.root=Path(root or ".")
 @abstractmethod
 def discover(self): ...
 @abstractmethod
 def load_sequence(self, participant_id, session_id): ...
 def load_participant(self, participant_id): return [s for s in self.discover() if s['participant_id']==participant_id]
 def load_session(self, participant_id, session_id): return self.load_sequence(participant_id,session_id)
 def load_labels(self): return {x['participant_id']:x.get('label') for x in self.discover()}
 def get_metadata(self): return {'adapter':type(self).__name__,'root':str(self.root),'samples':len(self.discover())}
class Generic3DSequenceAdapter(BaseDatasetAdapter):
 """Reads CSV rows in the documented common schema. modality and clinical_label are optional."""
 def _files(self): return list(self.root.rglob('*.csv')) if self.root.exists() else []
 def discover(self):
  out=[]
  for f in self._files():
   with f.open(newline='') as h:
    rows=list(csv.DictReader(h))
   if rows and REQUIRED_COLUMNS <= set(rows[0]):
    key=(rows[0]['participant_id'],rows[0]['session_id']); out.append({'participant_id':key[0],'session_id':key[1],'path':str(f),'label':rows[0].get('clinical_label') or None})
  return out
 def load_sequence(self,pid,sid):
  entry=next(x for x in self.discover() if x['participant_id']==pid and x['session_id']==sid)
  with open(entry['path'], newline='') as h: rows=list(csv.DictReader(h))
  frames=sorted({r['timestamp'] for r in rows},key=float); joints=sorted({r['joint_or_landmark_id'] for r in rows})
  arr=[[[float('nan')]*3 for _ in joints] for _ in frames]; fi={x:i for i,x in enumerate(frames)}; ji={x:i for i,x in enumerate(joints)}
  for r in rows: arr[fi[r['timestamp']]][ji[r['joint_or_landmark_id']]]=[float(r[k]) for k in ('x','y','z')]
  return {'sequence':arr,'timestamps':frames,'joints':joints,'label':entry['label'],'participant_id':pid,'session_id':sid}
class EmotionalBodyMovementAdapter(Generic3DSequenceAdapter):
 """CSV adapter alias for open Emotional Body Movement exports; labels mean tendency, not diagnosis."""
class DAICWOZAdapter(BaseDatasetAdapter):
 """Future controlled-access adapter: map face/head/AU/audio/text/PHQ-8 to common schema; no data bundled."""
 def discover(self): return []
 def load_sequence(self,*_): raise FileNotFoundError('DAIC-WOZ is not included or required.')
class RGBDAdapter:
 """Future camera boundary. Hardware/depth extraction is intentionally not implemented."""
 def _future(self): raise NotImplementedError('Provide RGB-D camera data in a future integration.')
 load_rgb=load_depth=extract_face_3d=extract_body_3d=extract_head_pose=_future

def auto_adapter(root):
 a=Generic3DSequenceAdapter(root)
 return a if a.discover() else a
