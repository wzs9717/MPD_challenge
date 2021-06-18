import pandas as pd
import json
import numpy as np
import codecs
import glob
import os
from dataAnalys import parsed_tracks 
#1. 分离出来p_test和其他的；
#tracks和playlist可能为空

file_name="./data/test/challenge_set.json"
files = glob.glob(file_name)
Feature_test = ['name', 'num_holdouts', 'num_samples', 'num_tracks', 'pid']
Feature_playlist = ['collaborative', 'duration_ms', 'modified_at', 
                'name', 'num_albums', 'num_artists', 'num_edits',
                'num_followers', 'num_tracks', 'pid']
Feature_tracks = ['album_name', 'album_uri', 'artist_name', 'artist_uri', 
                  'duration_ms', 'track_name', 'track_uri'] 
if not os.path.exists('./new_data'):
        os.makedirs('./new_data')

tracks_all=[]
pl_track_map=[]#[pid,tid,pos]
pl_all=[]
data = json.load(open(files[0]))
df_pl = pd.DataFrame(data["playlists"])
print(df_pl)
test=df_pl[Feature_test]
# for track in df_test_pl['tracks']:
#             playlists_test.append([playlist['pid'], track['track_uri'], track['pos']])
#             if track['track_uri'] not in tracks:
#                 data_tracks.append([track[col] for col in tracks_col])
#                 tracks.add(track['track_uri'])

df_playlists_test = pd.DataFrame(test, columns=Feature_test)#playlist_test是这个

track_list=df_pl[["tracks"]].values.tolist()
pid_list=df_pl[["pid"]].values.tolist()

track_list_flatten=[]
for i,pid in zip(track_list,pid_list):
    for j in i[0]:
            track_list_flatten.append(j)
            pl_track_map.append([pid[0],j['track_uri'],j['pos']])
         

df_pl_track_map=pd.DataFrame(pl_track_map,columns=['pid', 'tid', 'pos'])

df_tracks=pd.DataFrame(track_list_flatten)
df_tracks=df_tracks[Feature_tracks]
df_tracks_unique=df_tracks.drop_duplicates(subset=['track_uri'])
df_tracks_al=pd.read_csv("./new_data/tracks.csv")
track_uri2tid = df_tracks_al.set_index('track_uri').tid


df_pl_track_map.tid = df_pl_track_map.tid.map(track_uri2tid)
df_playlists_test.to_csv("./new_data/pl_test.csv", index = None)
df_pl_track_map.to_csv("./new_data/pl_test_map_validation.csv", index = None)

