import numpy as np
import pandas as pd
import joblib
import xgboost
#import lightfm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder


    
def create_co_occurence_features(df):
    df = df.drop_duplicates(['pid', 'tid'])
    
    num_items = df.tid.max() + 1
    num_users =  df.pid.max() + 1
    
    co_occurence = [defaultdict(int) for i in range(num_items)]
    occurence = [0 for i in range(num_items)]
    for q, (_, df) in enumerate(df.groupby('pid')):
        if q % 100000 == 0:
            print(q / 10000)
        tids = list(df.tid)
        for i in tids:
            occurence[i] += 1#
        for k, i in enumerate(tids):
            for j in tids[k + 1:]:
                co_occurence[i][j] += 1
                co_occurence[j][i] += 1
                #全是一的方阵
                
    def get_f(i, f):#
        if len(i) == 0:
            return -1
        else:
            return f(i)
        
    pids = df.pid.unique()
    seed = df[df.pid.isin(pids)]
    tid_seed = seed.groupby('pid').tid.apply(list)#？？？？
    
    co_occurence_seq = []
    for pid, tid in df[['pid', 'tid']].values:
        tracks = tid_seed.get(pid, [])
        co_occurence_seq.append(np.array([co_occurence[tid][i] for i in tracks]))
        
    df['co_occurence_max'] = [get_f(i, np.max) for i in co_occurence_seq]
    df['co_occurence_min'] = [get_f(i, np.min) for i in co_occurence_seq]
    df['co_occurence_mean'] = [get_f(i, np.mean) for i in co_occurence_seq]
    df['co_occurence_median'] = [get_f(i, np.median) for i in co_occurence_seq]
    
    co_occurence_seq = []
    for pid, tid in df[['pid', 'tid']].values:
        tracks = tid_seed.get(pid, [])
        co_occurence_seq.append(np.array([co_occurence[tid][i] / occurence[i] for i in tracks]))
        
    df['co_occurence_norm_max'] = [get_f(i, np.max) for i in co_occurence_seq]
    df['co_occurence_norm_min'] = [get_f(i, np.min) for i in co_occurence_seq]
    df['co_occurence_norm_mean'] = [get_f(i, np.mean) for i in co_occurence_seq]
    df['co_occurence_norm_median'] = [get_f(i, np.median) for i in co_occurence_seq]
    
    return df

# if __name__=='__main__':

#     track_map = pd.read_csv('./new_data/pl_track_map_test.csv')

#     track_map=create_co_occurence_features(track_map)


