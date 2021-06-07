from lightgbm import LGBMRanker
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.model_selection import train_test_split,cross_validate,cross_val_score,GridSearchCV,cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import NMF

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from surprise.prediction_algorithms.knns import KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline
import numpy as np
import pickle
import pandas as pd 
cat_feat_list_catboost=['city','shopId','currency']

def define_Hyper_pare():       
        cv_num=3
        num_class=8
        num_repeat=1
        lgbm_para=None
        lgbm_para ={
                'boosting_type': 'gbdt',
                'num_leaves': 128, 
                'max_depth': 8,
                # 'min_child_weight':1,
                'min_child_samples':110,
                'feature_fraction': 1.,
                'subsample': 1.,
                'subsample_freq':0,
                'reg_alpha': 0., 
                'reg_lambda': 0.,
                'learning_rate': 0.05, 
                'objective': 'multiclass',
                'n_estimators': 200,
                'subsample_for_bin':20000
                }
        nmf_para={
            'n_components':None
        }
            
        return cv_num,num_class,lgbm_para,nmf_para,num_repeat
cv_num,num_class,lgbm_para,nmf_para,num_repeat=define_Hyper_pare()

class TwoStageModel:
        def __init__(self):
                self.first_stage_models=self.define_first_stage_models(nmf_para)
                self.second_stage_model=LGBMRanker()
        
        def define_first_stage_models(self,nmf_para=None):
                models={}
                if nmf_para:
                        models['nmf']=NMF(**nmf_para)

                return models

        def first_stage_models_predict(self,X,model_name):
                saved_model_name='./checkpoints/%s_%s.sav'%(model_name)
                model = pickle.load(open(saved_model_name, 'rb'))
                prob= pd.DataFrame(model.transform(X.values),index=X.index)
                return prob

        def predict(self,X):
                prob_train_list=[]
                for model_name in self.first_stage_models.keys():
                        prob_for_second_stage_model=self.first_stage_models_predict(X,model_name)
                        prob_train_list.append(prob_for_second_stage_model)
                prob_train_mean=np.mean(prob_train_list)
                second_stage_model_name='./checkpoints/lgbm.sav'
                self.second_stage_model=pickle.load(open(second_stage_model_name, 'rb'))
                prob=pd.DataFrame(self.second_stage_model.predict(prob_train_mean),index=X.index)   
                return prob

        def train(self,train_X1,train_Y1,train_X2,train_Y2,test_X,test_Y):
                prob_train1_list=[]
                prob_for_second_stage_model_list=[]
                for model_name,model in self.first_stage_models.items():
                        train_prob1=self.train_first_stage_models(model_name,model,train_X1,train_Y1)
                        prob_train1_list.append(train_prob1)
                        prob_for_second_stage_model=self.first_stage_models_predict(train_X2,model_name)
                        prob_for_second_stage_model_list.append(prob_for_second_stage_model)

                prob_for_second_stage_model_mean=np.mean(prob_for_second_stage_model_list)
                self.train_second_stage_model(prob_for_second_stage_model_mean,train_Y2)
                prob_train2=self.predict(train_X2)
                prob_test=self.predict(test_X)
                return prob_train1_list,prob_train2,prob_test

        def train_first_stage_models(self,model_name,model,train_X,train_Y):
                print('\n----------->>fitting 1st model: ',model_name)
                #1.-------------------------------------------train the model-------------------------------------------
                saved_model_name='./checkpoints/%s.sav'%(model_name)

                model.fit(train_X.values)
                pickle.dump(model, open(saved_model_name, 'wb'))
                prob_train=model.transform(train_X)
                return prob_train
        
        def train_second_stage_model(self,train_X,train_Y):
                print('\n----------->>fitting 2nd model: ')
                #1.-------------------------------------------train the model-------------------------------------------
                saved_model_name='./checkpoints/lgbm.sav'

                model.fit(train_X.values,train_Y.values.ravel())
                pickle.dump(model, open(saved_model_name, 'wb'))
                prob_train=model.predict(train_X)
                return prob_train