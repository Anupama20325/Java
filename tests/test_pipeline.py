import math, unittest, tempfile, csv
from pathlib import Path
from src.data.adapters import Generic3DSequenceAdapter
from src.preprocessing.sequence import clean_sequence,normalize,windows
from src.features.behavioural import extract_features
from src.personalization.baseline import PersonalBaseline
from src.evaluation.splits import assert_no_leakage
class PipelineTests(unittest.TestCase):
 def setUp(self):self.x=[[[float(t),float(j),0.] for j in range(3)] for t in range(8)]
 def test_data_loading(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'data.csv'; p.write_text('participant_id,session_id,timestamp,joint_or_landmark_id,x,y,z\np,one,0,j,1,2,3\n'); a=Generic3DSequenceAdapter(d); self.assertEqual(len(a.discover()),1); self.assertEqual(a.load_sequence('p','one')['sequence'][0][0],[1.0,2.0,3.0])
 def test_normalization_and_nan(self):
  self.x[2][1][1]=float('nan'); x,p=normalize(self.x);self.assertTrue(all(math.isfinite(v) for f in x for q in f for v in q));self.assertEqual(len(x),8)
 def test_features_and_windows(self):
  w=windows(self.x,5,2);f,s=extract_features(w[0]);self.assertGreater(len(f),10);self.assertIn('activity',s)
 def test_baseline(self):
  d,n=PersonalBaseline().within_session_proxy([[1.,2.],[2.,3.]]);self.assertIn('not longitudinal',n);self.assertEqual(len(d),2)
 def test_leakage(self):
  with self.assertRaises(ValueError):assert_no_leakage([{'participant_id':'a'}],[{'participant_id':'a'}])
 def test_model_interface(self):
  try:
   from src.models.temporal import TemporalEncoder
   import torch
  except ImportError:return
  model=TemporalEncoder(3); y,e=model(torch.zeros(2,5,3,3));self.assertEqual(tuple(y.shape),(2,));self.assertEqual(tuple(e.shape),(2,128))
if __name__=='__main__':unittest.main()
