def classification_metrics(y_true,y_score,threshold=.5):
 """Metrics require sklearn; returns explicit unavailable status rather than invented metrics."""
 try:
  from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
 except ImportError:return {'status':'unavailable: install scikit-learn'}
 y_pred=[int(x>=threshold) for x in y_score]
 return {'accuracy':accuracy_score(y_true,y_pred),'balanced_accuracy':balanced_accuracy_score(y_true,y_pred),'precision':precision_score(y_true,y_pred,zero_division=0),'recall':recall_score(y_true,y_pred,zero_division=0),'f1':f1_score(y_true,y_pred,zero_division=0),'roc_auc':roc_auc_score(y_true,y_score) if len(set(y_true))>1 else None,'pr_auc':average_precision_score(y_true,y_score) if len(set(y_true))>1 else None,'class_distribution':{str(x):y_true.count(x) for x in set(y_true)}}
def regression_metrics(y_true,y_pred):
 try:
  from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
 except ImportError:return {'status':'unavailable: install scikit-learn'}
 return {'mae':mean_absolute_error(y_true,y_pred),'rmse':mean_squared_error(y_true,y_pred)**.5,'r2':r2_score(y_true,y_pred)}
