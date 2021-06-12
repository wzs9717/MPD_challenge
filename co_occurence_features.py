import numpy as np
import pandas as pd
import joblib
import xgboost
#import lightfm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder



data = pd.read_csv('./456.csv')
data = data.drop_duplicates(['pid', 'tid'])

num_items = data.tid.max() + 1
num_users =  data.pid.max() + 1

co_occurence = [defaultdict(int) for i in range(num_items)]
occurence = [0 for i in range(num_items)]
for q, (_, df) in enumerate(data.groupby('pid')):
    if q % 100000 == 0:
        print(q / 10000)
    tids = list(df.tid)
    for i in tids:
        occurence[i] += 1
    for k, i in enumerate(tids):
        for j in tids[k + 1:]:
            co_occurence[i][j] += 1
            co_occurence[j][i] += 1
            
            
def get_f(i, f):
    if len(i) == 0:
        return -1
    else:
        return f(i)
    
def create_co_occurence_features(df):
    
    pids = df.pid.unique()
    seed = data[data.pid.isin(pids)]
    tid_seed = seed.groupby('pid').tid.apply(list)
    
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
    
    
train = pd.read_hdf('df_data/ii_candidate.hdf')
val = pd.read_hdf('df_data/iii_candidate.hdf')
test = pd.read_hdf('df_data/test_candidate.hdf')

create_co_occurence_features(train)
create_co_occurence_features(val)
create_co_occurence_features(test)

train.to_hdf('df_data/ii_co_occurence_features.hdf', key='abc')
val.to_hdf('df_data/iii_co_occurence_features.hdf', key='abc')
test.to_hdf('df_data/test_co_occurence_features.hdf', key='abc')

