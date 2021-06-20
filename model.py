from lightgbm import LGBMRanker
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.model_selection import train_test_split,cross_validate,cross_val_score,GridSearchCV,cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from lightfm import LightFM
from lightfm.evaluation import recall_at_k

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import pandas as pd 
import scipy.sparse as sp

from utils.__init__ import get_config
config=get_config()

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
        fm_para={
                # 'no_components':10,
                'no_components':200, 'loss':'warp', 'learning_rate':0.02, 'max_sampled':400, 'random_state':1, 'user_alpha':1e-05
        }
        fm_para_text={
                # 'no_components':10,
                    'no_components':200, 
                    'loss':'warp', 
                    'learning_rate':0.03, 
                    'max_sampled':400, 
                    'random_state':1,
                    'user_alpha':1e-05,
        }
        return cv_num,num_class,lgbm_para,fm_para,fm_para_text
cv_num,num_class,lgbm_para,fm_para,fm_para_text=define_Hyper_pare()

class TwoStageModel:
        def __init__(self):
                self.first_stage_models=self.define_first_stage_models(fm_para,fm_para_text)
                self.second_stage_model=LGBMRanker()
        
        def define_first_stage_models(self,fm_para=None,fm_para_text=None):
                models={}
                if fm_para:
                        models['lightfm']=LightFM(**fm_para)
                if fm_para_text:
                        models['lightfm_text']=LightFM(**fm_para)
                return models

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

        def first_stage_models_predict(self,X,model_name):
                saved_model_name='./checkpoints/%s_%s.sav'%(model_name)
                model = pickle.load(open(saved_model_name, 'rb'))
                prob= pd.DataFrame(model.transform(X.values),index=X.index)
                return prob
        

        def train(self,playlist,tracks,map_train,map_val1,map_val2,val1_pids,val2_pids,play_list_test):
                
                # self.train_first_stage_fm_models('lightfm',self.first_stage_models['lightfm'],map_train,map_val1,map_val2,val1_pids,val2_pids)
                # self.train_first_stage_fm_text_models('lightfm_text',self.first_stage_models['lightfm_text'],playlist,tracks,map_train,map_val1,map_val2,val1_pids,val2_pids)
                user_seen = set(zip(map_train.pid, map_train.tid))
                save_candidates(
                                val1_pids.values,
                                map_val1.pid.value_counts(),
                                'res/ii_candidate.csv',
                                map_val1,
                                user_seen
                                )
                save_candidates(
                                val2_pids.values,
                                map_val2.pid.value_counts(),
                                'res/iii_candidate.csv',
                                map_val2,
                                user_seen
                                )
                save_candidates(
                                play_list_test.pid.values,
                                play_list_test.set_index('pid').num_holdouts,
                                'res/test_candidate.csv',
                                None,
                                user_seen
                                )
        def train_first_stage_fm_models(self,model_name,model,map_train,map_val1,map_val2,val1_pids,val2_pids):
                print('\n----------->>fitting fm model: ',model_name)
                
                saved_model_name='./checkpoints/%s.sav'%(model_name)
                train_X_sparse = sp.coo_matrix(
                                                (np.ones(map_train.shape[0]), (map_train.pid, map_train.tid)),
                                                shape=(config['num_playlists'], config['num_tracks'])
                                                )        
                val_sparse = sp.coo_matrix(
                                (np.ones(map_val1.shape[0]), (map_val1.pid, map_val1.tid)),
                                shape=(config['num_playlists'], config['num_tracks'])
                                )     
                #-------------------------------------------train the fm model-------------------------------------------
                best_recall = 0
                for i in range(config['epochs_stage1']):      
                        model.fit_partial(train_X_sparse, epochs=config['steps_per_epoch_epoch_stage1'])
                        recall=recall_at_k(model, val_sparse, k=config['top_k_stage1']).mean()
                        print('best_recall:',best_recall,'current_recal:',recall)
                        if recall > best_recall:
                                pickle.dump(model, open(saved_model_name, 'wb'))
                                best_recall = recall

        def train_first_stage_fm_text_models(self,model_name,model,playlist,tracks,map_train,map_val1,map_val2,val1_pids,val2_pids):
                print('\n----------->>fitting fm_text model: ',model_name)
                playlist_name = playlist.set_index('pid').name.sort_index()
                playlist_name = playlist_name.reindex(np.arange(config['num_playlists'])).fillna('')#expand from train size to train+test size

                vectorizer = CountVectorizer(max_features=20000)
                user_features = vectorizer.fit_transform(playlist_name)
                user_features = sp.hstack([sp.eye(config['num_playlists']), user_features])#??

                saved_model_name='./checkpoints/%s.sav'%(model_name)
                train_X_sparse = sp.coo_matrix(
                                                (np.ones(map_train.shape[0]), (map_train.pid, map_train.tid)),
                                                shape=(config['num_playlists'], config['num_tracks'])
                                                )    
                zeros_pids = np.array(list(set(val1_pids.values.T.tolist()[0]).difference(map_train.pid.unique())))#pid of val1-val1^train
                val_sparse = sp.coo_matrix(
                                (np.ones(map_val1.shape[0]), (map_val1.pid, map_val1.tid)),
                                shape=(config['num_playlists'], config['num_tracks'])
                                ) 
                # no_zeros_pids = np.array(list(set(val1_pids.values).difference(zeros_pids))[:1000])#pid of val1^train
                #-------------------------------------------train the fm model-------------------------------------------
                best_recall = 0
                for i in range(config['epochs_stage1']):      
                        model.fit_partial(train_X_sparse, epochs=config['steps_per_epoch_epoch_stage1'], user_features=user_features)
                        recall=recall_at_k(model, val_sparse,user_features=user_features, k=config['top_k_stage1']).mean()
                        # recall_no_zeros=recall_at_k(model, no_zeros_pids[['pid']], k=config['top_k_stage1']).mean()
                        print('best_recall:',best_recall,'current_recal:',recall)
                        if recall > best_recall:
                                pickle.dump(model, open(saved_model_name, 'wb'))
                                best_recall = recall
                pickle.dump(user_features, open('./new_data/user_features.pkl', 'wb'))

        def train_second_stage_model(self,train_X,train_Y):
                print('\n----------->>fitting 2nd model: ')
                #1.-------------------------------------------train the model-------------------------------------------
                saved_model_name='./checkpoints/lgbm.sav'

                model.fit(train_X.values,train_Y.values.ravel())
                pickle.dump(model, open(saved_model_name, 'wb'))
                prob_train=model.predict(train_X)
                return prob_train

        def save_candidates(target_pids, df_size, file_name, df=None,user_seen=None):
                '''
                df_size:{pid:counts(pid)}
                df:map
                '''
                model = pickle.load(open('./checkpoints/lightfm.sav', 'rb'))
                model_text = pickle.load(open('./checkpoints/lightfm_text.sav', 'rb'))
                user_features = pickle.load(open('models/user_features.pkl', 'rb'))

                target_pids_text = list(set(target_pids).difference(train.pid))
                target_pids_no_text = list(set(target_pids).difference(target_pids_text))
                res_notext={}
                res_text={}
                for i in target_pids_text:
                        scores=model_text.predict(i, np.arange(config['num_tracks']))
                        res_text[i]=np.argsort(-scores)[:10000]
                for i in target_pids:
                        scores=model.predict(i, np.arange(config['num_tracks']),user_features=user_features)
                        res_notext[i]=np.argsort(-scores)[:10000]
                res=res_notext.update(res_text)

                if df is not None:
                        val_tracks = df.groupby('pid').tid.apply(set).to_dict()  #{pid:[tracks]}
                
                pids = []
                tids = []
                targets = []
                for pid in target_pids:
                        l = max(df_size[pid] * 15, 700 + df_size[pid])
                        #l = 2000
                        pids += [pid] * l
                        tids += list(res[pid][:l])
                        
                        if df is not None:
                                tracks_t = val_tracks[pid]
                                targets += [i in tracks_t for i in res[pid][:l]]#tracs_true & track_predict_topl

                candidates = pd.DataFrame()
                candidates['pid'] = np.array(pids)
                candidates['tid'] = np.array(tids)
                
                if df is not None:
                        candidates['target'] = np.array(targets).astype(int)

                index = []
                for pid, tid in candidates[['pid', 'tid']].values:
                        index.append((pid, tid) not in user_seen)

                candidates = candidates[index]

                candidates.to_csv(file_name)