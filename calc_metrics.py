import numpy as np
from sklearn.metrics import *

def calc_model_performance(y_true,y_pred):
    _perform = {
        'Accuracy':accuracy_score(y_true=y_true,y_pred=y_pred>0.5),
        'AUROC':roc_auc_score(y_true=y_true,y_score=y_pred),
        'Precision':precision_score(y_true=y_true,y_pred=y_pred>0.5), # binary
        'Recall':recall_score(y_true=y_true,y_pred=y_pred>0.5), # binary
        'F1 score':f1_score(y_true=y_true,y_pred=y_pred>0.5), # binary
        'MCC':matthews_corrcoef(y_true=y_true,y_pred=y_pred>0.5),
        
    }
    return _perform
