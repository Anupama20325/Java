import math

def _mean(a): return sum(a)/len(a) if a else 0.
def _std(a): return math.sqrt(_mean([(v-_mean(a))**2 for v in a])) if a else 0.
def extract_features(x):
 flat=[[p[c] for f in x for p in f] for c in range(3)]; pos=sum(([ _mean(a),_std(a),max(a)-min(a)] for a in flat),[])
 centers=[[ _mean([p[c] for p in f]) for c in range(3)] for f in x]
 speeds=[math.sqrt(sum((centers[i][c]-centers[i-1][c])**2 for c in range(3))) for i in range(1,len(x))]; acc=[abs(speeds[i]-speeds[i-1]) for i in range(1,len(speeds))]
 spreads=[_mean([math.sqrt(sum((p[c]-centers[t][c])**2 for c in range(3))) for p in f]) for t,f in enumerate(x)]
 freq=sum(1 for i in range(1,len(speeds)) if speeds[i-1]*speeds[i]<0)/max(1,len(speeds))
 return pos+[_mean(speeds),_std(speeds),sum(v*v for v in speeds),_mean(acc),_std(acc),_mean(spreads),_std(spreads),freq], {'activity':_mean(speeds),'movement_variability':_std(speeds),'postural_variability':_std(spreads)}
