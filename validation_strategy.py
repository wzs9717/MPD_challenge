
import joblib

import numpy as np
import pandas as pd

def validation():
    df_tracks = pd.read_csv('./new_data/tracks.csv')
    df_playlists = pd.read_csv('./new_data/play_list.csv') 
    df_playlists_map= pd.read_csv('./new_data/pl_track_map_test.csv')
    df_test=pd.read_csv('./new_data/pl_test.csv')
    df_test_map=pd.read_csv('./new_data/pl_test_map_validation.csv')
    
    
    num_tracks = df_playlists.groupby('num_tracks').pid.apply(np.array)
    # num_pl, pl_name_list
# 5, [‘summer’,winter]
# 10,[‘daoxiang’]
    
    validation_playlists = {}
    for i, j in df_test.num_tracks.value_counts().reset_index().values:
         #5,1000
    #10,1200
        validation_playlists[i] = np.random.choice(num_tracks.loc[i], 2 * j, replace=False)#??
         #{5:[‘summer’,winter]
    # 10,[‘daoxiang’,‘daoxiang’]}
    
    val1_playlist = {}
    val2_playlist = {}
    for i in [0, 1, 5, 10, 25, 100]:
        
        val1_playlist[i] = []
        val2_playlist[i] = []
        
        value_counts = df_test.query('num_samples==@i').num_tracks.value_counts()
        for j, k in  value_counts.reset_index().values:
            
            val1_playlist[i] += list(validation_playlists[j][:k])
            validation_playlists[j] = validation_playlists[j][k:]
            
            val2_playlist[i] += list(validation_playlists[j][:k])
            validation_playlists[j] = validation_playlists[j][k:]
    val1_index = df_playlists_map.pid.isin(val1_playlist[0])
    val2_index = df_playlists_map.pid.isin(val2_playlist[0])
    for i in [1, 5, 10, 25, 100]:
        val1_index = val1_index | (df_playlists_map.pid.isin(val1_playlist[i]) & (df_playlists_map.pos >= i))
        val2_index = val2_index | (df_playlists_map.pid.isin(val2_playlist[i]) & (df_playlists_map.pos >= i))
            
    train = df_playlists_map[~(val1_index | val2_index)]

    val1 = df_playlists_map[val1_index]
    val2 = df_playlists_map[val2_index]
    
    val1_pids = np.hstack([val1_playlist[i] for i in val1_playlist])
    val2_pids = np.hstack([val2_playlist[i] for i in val2_playlist])
    
    train = pd.concat([train, df_test_map])
    train.to_csv('./new_data/train.csv',index = None)

    val1.to_csv('./new_data/val1.csv',index = None)
    val2.to_csv('./new_data/val2.csv', index = None)
    
if __name__=='__main__':
          validation()  
            
            
            
            
            
            
            
            