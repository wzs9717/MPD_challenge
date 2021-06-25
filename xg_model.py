import pandas as pd
import numpy as np
# 运行 xgboost安装包中的示例程序
from xgboost import XGBClassifier as xgb
# 加载LibSVM格式数据模块
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
from lightgbm import LGBMClassifier

pd.options.display.max_columns = 100

data = pd.read_csv('./new_data/train.csv')

df_playlists = pd.read_csv('./new_data/play_list.csv')
df_playlists_test = pd.read_csv('./new_data/play_list_test.csv')
df_test=pd.read_csv('./new_data/play_list_test.csv')
df_tracks = pd.read_csv('./new_data/tracks.csv')




df_tracks['album'] = LabelEncoder().fit_transform(df_tracks.album_uri)
df_tracks['artist'] = LabelEncoder().fit_transform(df_tracks.artist_uri)

train = pd.read_csv('./res/ii_candidate.csv')
val = pd.read_csv('./res/iii_candidate.csv')
test = pd.read_csv('./res/test_candidate.csv')

train_holdouts = pd.read_csv('./new_data/val1.csv')

val_holdouts = pd.read_csv('./new_data/val2.csv')

train_length = train_holdouts.groupby('pid').tid.nunique()#pid and num_pl_tid
val_length = val_holdouts.groupby('pid').tid.nunique()
test_length = df_playlists_test.set_index('pid').num_holdouts

num_items = data.tid.max() + 1# the number of data

def create_count(df):
    
    tid_count = data.tid.value_counts()#first column is tid , second colunmn is count
    pid_count = data.pid.value_counts()

    df['tid_count'] = df.tid.map(tid_count).fillna(0)
    df['pid_count'] = df.pid.map(pid_count).fillna(0)
    
    album_count = data.tid.map(df_tracks.album).value_counts()
    artist_count = data.tid.map(df_tracks.artist).value_counts()
    
    df['album_count'] = df.tid.map(df_tracks.album).map(album_count).fillna(0)
    df['artist_count'] = df.tid.map(df_tracks.artist).map(artist_count).fillna(0)
     
    album_count

def isin(i, j):
    if j is not np.nan:
        return i in j
    return False#判断是不是

def isin_sum(i, j):
    if j is not np.nan:
        return np.sum(i == j)
    return 0
def creaet_artist_features(df):
    
    data_short = data[data.pid.isin(df.pid)]
    pid_artist = data_short.tid.map(df_tracks.artist).groupby(data_short.pid).apply(np.array)
    df_playlist = df.pid.map(pid_artist)
    df_artist = df.tid.map(df_tracks.artist)
    
    share_unique = pid_artist.apply(np.unique).apply(len) / pid_artist.apply(len)
    
    df['share_of_unique_artist'] = df.pid.map(share_unique).fillna(-1)
    df['sim_artist_in_playlist'] = [isin_sum(i, j) for i, j in zip(df_artist, df_playlist)]
    df['mean_artist_in_playlist'] = (df['sim_artist_in_playlist'] / df.pid.map(pid_artist.apply(len))).fillna(-1)
def creaet_album_features(df):
    
    data_short = data[data.pid.isin(df.pid)]
    pid_album = data_short.tid.map(df_tracks.album).groupby(data_short.pid).apply(np.array)
    df_playlist = df.pid.map(pid_album)
    df_album = df.tid.map(df_tracks.album)
    
    share_unique = pid_album.apply(np.unique).apply(len) / pid_album.apply(len)
    
    df['share_of_unique_album'] = df.pid.map(share_unique).fillna(-1)
    df['sim_album_in_playlist'] = [isin_sum(i, j) for i, j in zip(df_album, df_playlist)]
    df['mean_album_in_playlist'] = (df['sim_album_in_playlist'] / df.pid.map(pid_album.apply(len))).fillna(-1)
def create_features(df, df_length):
    create_count(df)
    creaet_artist_features(df)
    creaet_album_features(df)
    df['tracks_holdout'] = df.pid.map(df_length)
create_features(train, train_length)#map, num_pl_tracks
create_features(val, val_length)
create_features(test, test_length)

train_co = pd.read_csv('./new_data/ii_co_occurence_features.csv').drop('target', axis=1)
val_co = pd.read_csv('./new_data/iii_co_occurence_features.csv').drop('target', axis=1)
test_co = pd.read_csv('./new_data/test_co_occurence_features.csv')

train_lightfm = pd.read_csv('./new_data/ii_lightfm_features.csv').drop('target', axis=1)
val_lightfm = pd.read_csv('./new_data/iii_lightfm_features.csv').drop('target', axis=1)
test_lightfm = pd.read_csv('./new_data/test_lightfm_features.csv')

train = train.merge(train_co, on=['pid', 'tid'])
val = val.merge(val_co, on=['pid', 'tid'])
test = test.merge(test_co, on=['pid', 'tid'])

train = train.merge(train_lightfm, on=['pid', 'tid'])
val = val.merge(val_lightfm, on=['pid', 'tid'])
test = test.merge(test_lightfm, on=['pid', 'tid'])

cols = ['pid', 'tid', 'target']
y_train=train.target
X_train=train.drop(cols, axis=1)
2
y_val=val.target
X_val=val.drop(cols, axis=1)

X_test=test.drop(['pid'],axis=1)
X_test=X_test.drop(['tid'],axis=1)


num_round = 10
lgb = LGBMClassifier()
lgb.fit(X_train.values, y_train.values, early_stopping_rounds=5,eval_set=[(X_train.values, y_train.values), (X_val.values, y_val.values)])
pickle.dump(lgb, open('./checkpoints/lgbm.pkl', 'wb'))
lgb=pickle.load(open('./checkpoints/lgbm.pkl', 'rb'))
p = lgb.predict(X_val)
val['p'] = p


scores = []
for pid, df, in val.sort_values('p', ascending=False).groupby('pid'):
    n = val_length[pid]
    scores.append(df[:n].target.sum() / n)
print(np.mean(scores))

test_pre=lgb.predict_proba(X_test)
test['p'] = test_pre
test = test.sort_values(['pid', 'p'], ascending=[True, False])
recs = test.groupby('pid').tid.apply(lambda x: x.values[:500])
track_uri = df_tracks.track_uri 




















