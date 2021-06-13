
import joblib

import numpy as np
import pandas as pd

def validation():
    df_tracks = pd.read_csv('df_data/tracks.csv')
    df_playlists = pd.read_csv('./new_data/play_lists.csv') 
    df_playlists_map= pd.read_csv('./new_data/pl_track_map_test.csv')
    
    
    num_tracks = df_playlists.groupby('num_tracks').pid.apply(np.array)
    # num_pl, pl_name_list
# 5, [‘summer’,winter]
# 10,[‘daoxiang’]
    
    validation_playlists = {}
    for i, j in df_playlists.num_tracks.value_counts().reset_index().values:
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
        
        value_counts = df_playlists_test_info.query('num_samples==@i').num_tracks.value_counts()
        for j, k in  value_counts.reset_index().values:
            
            val1_playlist[i] += list(validation_playlists[j][:k])
            validation_playlists[j] = validation_playlists[j][k:]
            
            val2_playlist[i] += list(validation_playlists[j][:k])
            validation_playlists[j] = validation_playlists[j][k:]
            
            
            
            
            
            
            
            
            
            
            
            
            
            