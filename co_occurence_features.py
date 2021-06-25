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
        # if q % 1000 == 0:
        print(q)
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
    print('2')
    co_occurence_seq = []
    for pid, tid in df[['pid', 'tid']].values:
        tracks = tid_seed.get(pid, [])
        co_occurence_seq.append(np.array([co_occurence[tid][i] for i in tracks]))
        
    df['co_occurence_max'] = [get_f(i, np.max) for i in co_occurence_seq]
    df['co_occurence_min'] = [get_f(i, np.min) for i in co_occurence_seq]
    df['co_occurence_mean'] = [get_f(i, np.mean) for i in co_occurence_seq]
    df['co_occurence_median'] = [get_f(i, np.median) for i in co_occurence_seq]
    print('3')
    co_occurence_seq = []
    for pid, tid in df[['pid', 'tid']].values:
        tracks = tid_seed.get(pid, [])
        co_occurence_seq.append(np.array([co_occurence[tid][i] / occurence[i] for i in tracks]))
        
    df['co_occurence_norm_max'] = [get_f(i, np.max) for i in co_occurence_seq]
    df['co_occurence_norm_min'] = [get_f(i, np.min) for i in co_occurence_seq]
    df['co_occurence_norm_mean'] = [get_f(i, np.mean) for i in co_occurence_seq]
    df['co_occurence_norm_median'] = [get_f(i, np.median) for i in co_occurence_seq]
    
 
    

train = pd.read_csv('./res/ii_candidate.csv')
val = pd.read_csv('./res/iii_candidate.csv')
test = pd.read_csv('./res/test_candidate.csv')

create_co_occurence_features(train)
create_co_occurence_features(val)
create_co_occurence_features(test)

train.to_csv('./new_data/ii_co_occurence_features.csv')
val.to_csv('./new_data/iii_co_occurence_features.csv')
test.to_csv('./new_data/test_co_occurence_features.csv')

# if __name__=='__main__':

#     track_map = pd.read_csv('./new_data/pl_track_map_test.csv')

#     track_map=create_co_occurence_features(track_map)


