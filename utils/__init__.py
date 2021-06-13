import numpy as np
import pickle
import pandas as pd 
import scipy.sparse as sp

def get_config():
    df_playlists=pd.read_csv('./new_data/play_list.csv')
    df_tracks=pd.read_csv('./new_data/tracks.csv')
    config={}
    config['num_playlists']=df_playlists.pid.max() + 1
    config['num_tracks']=df_tracks.tid.max() + 1
    config['epochs_stage1']=10
    config['steps_per_epoch_epoch_stage1']=1
    config['top_k_stage1']=600
    return config
