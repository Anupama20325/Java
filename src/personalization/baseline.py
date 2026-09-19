import math
class PersonalBaseline:
 def fit(self, embeddings):
  if not embeddings: raise ValueError('baseline needs embeddings')
  self.mean=[sum(x[i] for x in embeddings)/len(embeddings) for i in range(len(embeddings[0]))]; self.std=[max(1e-8,math.sqrt(sum((x[i]-self.mean[i])**2 for x in embeddings)/len(embeddings))) for i in range(len(self.mean))]; return self
 def deviation(self,x):
  euclidean=math.sqrt(sum((a-b)**2 for a,b in zip(x,self.mean))); dot=sum(a*b for a,b in zip(x,self.mean)); cosine=1-dot/(math.sqrt(sum(a*a for a in x))*math.sqrt(sum(b*b for b in self.mean)) or 1); standardized=math.sqrt(sum(((a-b)/s)**2 for a,b,s in zip(x,self.mean,self.std))); return {'euclidean':euclidean,'cosine':cosine,'standardized':standardized,'baseline_deviation_score':standardized}
 def within_session_proxy(self, embeddings):
  n=max(1,math.ceil(len(embeddings)*.2)); self.fit(embeddings[:n]); return [self.deviation(x) for x in embeddings], 'Within-session baseline proxy; not longitudinal validation.'
