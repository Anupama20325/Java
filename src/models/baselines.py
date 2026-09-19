"""Classical baselines; sklearn is optional and only used with labelled real data."""
def build_baselines(task='classification'):
 try:
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression, Ridge
  from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
  from sklearn.neural_network import MLPClassifier, MLPRegressor
 except ImportError: return {}
 if task=='regression': return {'ridge':make_pipeline(StandardScaler(),Ridge()),'random_forest':RandomForestRegressor(n_estimators=100,random_state=42),'mlp':make_pipeline(StandardScaler(),MLPRegressor(random_state=42,max_iter=300))}
 return {'logistic_regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=500)),'random_forest':RandomForestClassifier(n_estimators=100,random_state=42),'mlp':make_pipeline(StandardScaler(),MLPClassifier(random_state=42,max_iter=300))}
