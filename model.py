from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.model_selection import train_test_split,cross_validate,cross_val_score,GridSearchCV,cross_val_predict
from sklearn.multiclass import OneVsRestClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
import numpy as np
import joblib
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
        return cv_num,num_class,lgbm_para,num_repeat
cv_num,num_class,lgbm_para,num_repeat=define_Hyper_pare()

class StackingModel:
        def __init__(self,version):
                self.version=version
                self.base_models=self.make_base_models(lgbm_para,GAM_para,GAM_para_new,xgb_para,catboost_para,GLM_para,NN_para)
                self.meta_model=make_pipeline(
                        FunctionTransformer(self.meta_link_function),
                        LogisticRegression(multi_class='ovr',penalty="l2",max_iter=1000,solver='lbfgs')
                        )
                print('stacking model:',self.base_models.keys())

        def meta_link_function(self,x):
                x[x==0]+=1e-4
                return np.log(x)
        
        def make_base_models(self,lgbm_para=None,GAM_para=None,GAM_para_new=None,xgb_para=None,catboost_para=None,GLM_para=None,NN_para=None):
                models={}
                if lgbm_para:
                        lgbm_classifier=make_pipeline(StandardScaler(),LGBMClassifier(**lgbm_para))
                        for i in range(num_repeat):
                                models['lgbm%s'%i]=lgbm_classifier
                if xgb_para:
                        xgb_classifier = make_pipeline(StandardScaler(),xgb.XGBClassifier(**xgb_para))
                        models['xgb']=(xgb_classifier)
                if GLM_para:
                        lr_classifier = make_pipeline(StandardScaler(),LogisticRegression(**GLM_para))
                        models['glm']=(lr_classifier)
                if GAM_para:
                        GAM_classifier=make_pipeline(StandardScaler(),(myGAM(**GAM_para))) if self.version=='old' else make_pipeline(StandardScaler(),(myGAM(**GAM_para_new)))
                        models['gam']=(GAM_classifier)
                if catboost_para:
                        catboost_classifier=make_pipeline(StandardScaler(),CatBoostClassifier(**catboost_para))
                        models['cb']=(catboost_classifier)
                if NN_para:
                        NN_classifier='define in train_nn'
                        models['nn']=(NN_classifier)
                return models

        def predict(self,X,path='./checkpoints'):
                y_prob_cv_for_meta_model_list=None
                for model_name in self.base_models.keys():
                        y_prob_cv_for_meta_model=self.base_model_predict(X,model_name,path)
                        y_prob_cv_for_meta_model_list=np.concatenate((y_prob_cv_for_meta_model_list,y_prob_cv_for_meta_model),1) if (y_prob_cv_for_meta_model_list is not None) else y_prob_cv_for_meta_model
                meta_model_name=path+'/meta_model_%s.sav'%self.version
                self.meta_model=joblib.load(open(meta_model_name, 'rb'))
                prob=pd.DataFrame(self.meta_model.predict_proba(y_prob_cv_for_meta_model_list),index=X.index)   
                # prob=pd.DataFrame((y_prob_cv_for_meta_model_list),index=X.index)   
                return prob

        def base_model_predict(self,X,model_name,path='./checkpoints'):
                if model_name=='nn':
                        pipe=load_model(path+'/%s_%s.h5'%(model_name,self.version) )
                else:
                        saved_model_name=path+'/%s_%s.sav'%(model_name,self.version)
                        pipe = joblib.load(open(saved_model_name, 'rb'))
                prob= pd.DataFrame(pipe.predict_proba(X.values),index=X.index)
                return prob

        def train(self,train_X,train_Y,test_X,test_Y,gridsearch_flag=False):
                y_prob_cv_for_meta_model_list=None
                for model_name,model in self.base_models.items():
                        y_prob_cv_for_meta_model=self.train_base_model(model_name,model,train_X,train_Y,test_X,test_Y,gridsearch_flag)
                        y_prob_cv_for_meta_model_list=np.concatenate((y_prob_cv_for_meta_model_list,y_prob_cv_for_meta_model),1) if (y_prob_cv_for_meta_model_list is not None) else y_prob_cv_for_meta_model
                self.meta_model.fit(y_prob_cv_for_meta_model_list,train_Y.values.ravel())
                coef=self.meta_model[1].coef_
                print('%s meta model coef:\n%s,intercept:\n%s'%(self.version,coef,self.meta_model[1].intercept_))
                meta_model_name='./checkpoints/meta_model_%s.sav'%self.version
                joblib.dump(self.meta_model, open(meta_model_name, 'wb'))

        def train_nn_gam(self,model_name,model,train_X,train_Y,test_X,test_Y,gridsearch_flag=False):
                '''
                manually do cross validation prediction for nn and gam since they does not support sklearn
                '''
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_num,shuffle=True)
                y_prob_cv_for_meta_model_list=[]
                for train_index, test_index in kf.split(train_X.values, train_Y.values):   
                        train_X_cv= train_X.iloc[train_index,:]
                        train_Y_cv= train_Y.iloc[train_index]
                        test_X_cv= train_X.iloc[test_index,:]
                        test_Y_cv= train_Y.iloc[test_index]

                        if model_name=='nn':
                                model=tinyDNN(train_X.shape[1],num_class)
                                model.compile(
                                optimizer=tf.keras.optimizers.Adam(lr=NN_para['lr'], decay=1e-7),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                metrics=None
                                )#compile model
                                model.fit(x=train_X_cv.values, y=train_Y_cv.values,
                                epochs=NN_para['epochs'],
                                validation_data=(test_X_cv.values,test_Y_cv.values),
                                batch_size=NN_para['batch_size'],
                                shuffle=True,
                                verbose=0
                                )
                                y_prob_cv_for_meta_model_list.append(pd.DataFrame(model.predict(test_X_cv.values),index=test_X_cv.index)) 
                        else:
                                model.fit(train_X_cv.values,train_Y_cv.values.ravel())
                                y_prob_cv_for_meta_model_list.append(pd.DataFrame(model.predict_proba(test_X_cv.values),index=test_X_cv.index))  
                y_prob_cv_for_meta_model=pd.concat(y_prob_cv_for_meta_model_list).sort_index().values
                print(y_prob_cv_for_meta_model.mean(axis=0))
                cv_metrics=caculate_metrics_classification_prob(train_Y.values,y_prob_cv_for_meta_model,num_class)
                print('\nCV result:')
                for key, value in cv_metrics.items():
                        print(key,'_mean:',np.mean(value))
                if model_name=='nn':
                        #retrain and save the model
                        model=tinyDNN(train_X.shape[1],num_class)
                        model.compile(
                        optimizer=tf.keras.optimizers.Adam(lr=NN_para['lr'], decay=1e-7),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=None
                        )#compile model
                        model.fit(x=train_X, y=train_Y,
                        epochs=NN_para['epochs'],
                        validation_data=(test_X,test_Y),
                        batch_size=NN_para['batch_size'],
                        shuffle=True,
                        verbose=1
                        )
                        saved_model_name='./checkpoints/%s_%s.h5'%(model_name,self.version)
                        print('--------------saving to',saved_model_name)
                        model.save(saved_model_name)
                else:#gam
                        saved_model_name='./checkpoints/%s_%s.sav'%(model_name,self.version)
                        model.fit(train_X.values, train_Y.values.ravel())
                        joblib.dump(model, open(saved_model_name, 'wb'))
                return y_prob_cv_for_meta_model


        def train_base_model(self,model_name,model,train_X,train_Y,test_X,test_Y,gridsearch_flag=False):
                print('\n----------->>fitting model: ',model_name,self.version)
                #-------------------------------------------define pipeline.-------------------------------------------
                if model_name=='nn' or model_name=='gam':
                        return self.train_nn_gam(model_name,model,train_X,train_Y,test_X,test_Y,gridsearch_flag)
                # -------------------------------------train and cross validation loop-----------------------
                y_prob_cv_for_meta_model=cross_val_predict(model, train_X.values,train_Y.values.ravel(), cv=cv_num,method='predict_proba')
                cv_metrics=caculate_metrics_classification_prob(train_Y.values,y_prob_cv_for_meta_model,num_class)
                print((train_X.shape[1]))
                print('\nCV result:')
                for key, value in cv_metrics.items():
                        print(key,'_mean:',np.mean(value))
        
                # -------------------------------------gridsearch to tune the hyper-parameters-----------------------
                if gridsearch_flag:
                        param_grid = {
                        'lgbm_classifier__learning_rate':[0.1, 0.01, 0.001],
                        'lgbm_classifier__n_estimators':[100, 500]
                        }
                        param_grid_parsed=param_grid_parse(param_grid)
                        print(param_grid_parsed)
                        average_precision_list=[]
                        for param in param_grid_parsed:
                                model.set_params(**param)
                                y_prob_cv=cross_val_predict(model, train_X.values,train_Y.values.ravel(), cv=5,method='predict_proba')
                                cv_metrics=caculate_metrics_classification_prob(train_Y.values,y_prob_cv,num_class)
                                average_precision_list.append(cv_metrics['average_precision_score'])
                                best_average_precision_score=np.max(average_precision_list)
                                best_average_precision_param=param_grid_parsed[np.argmax(average_precision_list)]
                                print('CV result for current Hyper-params:\n',average_precision_list)
                                print('best average_precision_score: %s with param:\n'%(best_average_precision_score),best_average_precision_param)
                #3.-------------------------------------------retrain the model-------------------------------------------
                saved_model_name='./checkpoints/%s_%s.sav'%(model_name,self.version)

                model.fit(train_X.values, train_Y.values.ravel())
                joblib.dump(model, open(saved_model_name, 'wb'))

                return y_prob_cv_for_meta_model

class DoubleModel:
        '''
        old/new model
        '''
        def __init__(self):
                self.model_old = StackingModel(version='old')
                self.model_new = StackingModel(version='new')
        
        def train(self,train_X,train_Y,test_X,test_Y,retrain_flag=False,threshold=50):
                '''
                train different models for new/old shops
                '''
                train_X_new  = train_X.loc[train_X.all_amount_shop<threshold].drop(['all_amount_shop']+[i for i in train_X.columns.tolist() if i.startswith('shop_claim_rate_')],axis=1)
                train_X_old  = train_X.loc[train_X.all_amount_shop>=threshold]
                test_X_new  = test_X.loc[test_X.all_amount_shop<threshold].drop(['all_amount_shop']+[i for i in test_X.columns.tolist() if i.startswith('shop_claim_rate_')],axis=1)
                test_X_old  = test_X.loc[test_X.all_amount_shop>=threshold]
                train_Y_new  = train_Y.loc[train_X.all_amount_shop<threshold]
                train_Y_old  = train_Y.loc[train_X.all_amount_shop>=threshold]
                test_Y_new  = test_Y.loc[test_X.all_amount_shop<threshold]
                test_Y_old  = test_Y.loc[test_X.all_amount_shop>=threshold]

                train_X_new_expanded  = train_X.drop(['all_amount_shop']+[i for i in train_X.columns.tolist() if i.startswith('shop_claim_rate_')],axis=1)
                test_X_new_expanded  = test_X.drop(['all_amount_shop']+[i for i in test_X.columns.tolist() if i.startswith('shop_claim_rate_')],axis=1)
                train_Y_new_expanded  = train_Y
                test_Y_new_expanded = test_Y
                if retrain_flag:
                        self.model_old.train(train_X_old,train_Y_old,test_X_old,test_Y_old)
                        self.model_new.train(train_X_new_expanded,train_Y_new_expanded,test_X_new_expanded,test_Y_new_expanded)
                        
                # -------------------------------------------evaluate-------------------------------------------
                train_prob_new=self.model_new.predict(train_X_new)
                test_prob_new=self.model_new.predict(test_X_new)
                train_prob_old=self.model_old.predict(train_X_old)
                test_prob_old=self.model_old.predict(test_X_old)
                train_prob=pd.concat([train_prob_new,train_prob_old]).sort_index().values
                test_prob=pd.concat([test_prob_new,test_prob_old]).sort_index().values

                num_base_models=len(self.model_old.base_models)
                importance_old=np.mean(np.mean(np.abs(self.model_old.meta_model[1].coef_),axis=0).reshape((num_base_models,num_class)),axis=1)
                importance_new=np.mean(np.mean(np.abs(self.model_new.meta_model[1].coef_),axis=0).reshape((num_base_models,num_class)),axis=1)
                
                print('model importance:\n',((importance_old+importance_new)/2))
                
                metrics=caculate_metrics_classification_prob(test_Y.values,test_prob,num_class)
                print('\nval result:\n',metrics)
                return train_prob,test_prob

        def predict(self,X,threshold=50,path='./checkpoints'):
                X_new  = X.loc[X.all_amount_shop<threshold].drop(['all_amount_shop']+[i for i in X.columns.tolist() if i.startswith('shop_claim_rate_')],axis=1)
                X_old  = X.loc[X.all_amount_shop>=threshold]
                prob_new=self.model_new.predict(X_new,path)
                prob_old=self.model_old.predict(X_old,path)
                prob=pd.concat([prob_new,prob_old]).sort_index().values
                return prob
