import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from data_generater import data_gene 
# 运行 xgboost安装包中的示例程序
from xgboost import XGBClassifier as xgb
# 加载LibSVM格式数据模块
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

def xg_model():
    param_dist = {'objective':'binary:logistic', 'n_estimators':2}
    X_train,X_test,y_train,y_test=data_gene()
    clf = xgb(**param_dist)
    clf.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test,y_test)],eval_metric='auc',verbose=True)
    evals_result = clf.evals_result()
    
if __name__=='__main__':
     
     xg_model()