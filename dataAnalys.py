
import pandas as pd
import json
import numpy as np
import codecs
import glob
import os
def parsed_tracks():
    file_name='./data/train/*.json'
    files = glob.glob(file_name)                    
   
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
    c=1
    for file in files[:]:
        print(c)
        data = json.load(open(file))
        df = pd.DataFrame(data["playlists"])

        track_list=df[["tracks"]].values.tolist()
        pid_list=df[["pid"]].values.tolist()
        pl_all.append(df[Feature_playlist])
        
        track_list_flatten=[]
        for i,pid in zip(track_list,pid_list):
            for j in i[0]:
                    track_list_flatten.append(j)
                    pl_track_map.append([pid[0],j['track_uri'],j['pos']])
        df_tracks=pd.DataFrame(track_list_flatten)
        df_tracks=df_tracks[Feature_tracks]
        df_tracks_unique=df_tracks.drop_duplicates(subset=['track_uri'])
        tracks_all.append(df_tracks_unique)
        if c%200==0:
            df_tracks_all_tem=pd.concat(tracks_all,ignore_index=True).drop_duplicates(subset=['track_uri'])
            df_tracks_all_tem.to_csv("./new_data/track%s.csv"%c, index = None)
            df_tracks_all_tem=None
            tracks_all=[]
        c+=1
    df_tracks_all=comple_tracks()
    if bool(tracks_all):
        df_tracks_all=pd.concat([df_tracks_all]+tracks_all,ignore_index=True)
    df_tracks_all=df_tracks_all.drop_duplicates(subset=['track_uri'],ignore_index=True)
    df_pl_track_map=pd.DataFrame(pl_track_map,columns=['pid', 'tid', 'pos'])
    df_pl_all=pd.concat(pl_all,axis=0,ignore_index=True)

    df_pl_all['collaborative'] = df_pl_all['collaborative'].map({'false': 0, 'true': 1})
    df_tracks_all['tid'] = df_tracks_all.index

    track_uri2tid = df_tracks_all.set_index('track_uri').tid
    df_pl_track_map.tid = df_pl_track_map.tid.map(track_uri2tid)

    df_tracks_all.to_csv("./new_data/tracks.csv", index = None)
    df_pl_track_map.to_csv("./new_data/pl_track_map.csv", index = None)
    df_pl_all.to_csv("./new_data/play_list.csv", index = None)

def comple_tracks():
    file_cname='./new_data/track*.csv'
    files_c=glob.glob(file_cname)
    print(files_c)
    tracks_all=[]
    for i in files_c[1:]:
        tem=pd.read_csv(i)
        tracks_all.append(tem)
    df_ctracks_all=pd.concat(tracks_all)
    df_ctracks_all=df_ctracks_all.drop_duplicates(subset=['track_uri'],ignore_index=True)
    df_ctracks_all.reset_index(drop=True)
    return df_ctracks_all

def create_df_data():
    
    path = 'data'
    
    playlist_col = ['collaborative', 'duration_ms', 'modified_at', 
                'name', 'num_albums', 'num_artists', 'num_edits',
                'num_followers', 'num_tracks', 'pid']
    tracks_col = ['album_name', 'album_uri', 'artist_name', 'artist_uri', 
                  'duration_ms', 'track_name', 'track_uri'] 
    playlist_test_col = ['name', 'num_holdouts', 'num_samples', 'num_tracks', 'pid']
#---------------------train---------------------
    filenames = os.listdir(path)
    
    data_playlists = []#pl info
    data_tracks = []#track info
    playlists = []#map

    tracks = set()

    for filename in filenames:
        fullpath = os.sep.join((path, filename))
        f = open(fullpath)
        js = f.read()
        f.close()

        mpd_slice = json.loads(js)

        for playlist in mpd_slice['playlists']:
            data_playlists.append([playlist[col] for col in playlist_col])
            for track in playlist['tracks']:
                playlists.append([playlist['pid'], track['track_uri'], track['pos']])
                if track['track_uri'] not in tracks:
                    data_tracks.append([track[col] for col in tracks_col])
                    tracks.add(track['track_uri'])#<-----
#-----------------test--------------------
    f = open('challenge_set.json')
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)

    data_playlists_test = []
    playlists_test = []

    for playlist in mpd_slice['playlists']:
        data_playlists_test.append([playlist.get(col, '') for col in playlist_test_col])
        for track in playlist['tracks']:
            playlists_test.append([playlist['pid'], track['track_uri'], track['pos']])
            if track['track_uri'] not in tracks:
                data_tracks.append([track[col] for col in tracks_col])
                tracks.add(track['track_uri'])
#----------------------------------------
    df_playlists_info = pd.DataFrame(data_playlists, columns=playlist_col)
    df_playlists_info['collaborative'] = df_playlists_info['collaborative'].map({'false': False, 'true': True})

    df_tracks = pd.DataFrame(data_tracks, columns=tracks_col)
    df_tracks['tid'] = df_tracks.index

    track_uri2tid = df_tracks.set_index('track_uri').tid

    df_playlists = pd.DataFrame(playlists, columns=['pid', 'tid', 'pos'])
    df_playlists.tid = df_playlists.tid.map(track_uri2tid)

    df_playlists_test_info = pd.DataFrame(data_playlists_test, columns=playlist_test_col)

    df_playlists_test = pd.DataFrame(playlists_test, columns=['pid', 'tid', 'pos'])
    df_playlists_test.tid = df_playlists_test.tid.map(track_uri2tid)

    df_tracks.to_hdf('df_data/df_tracks.hdf', key='abc')
    df_playlists.to_hdf('df_data/df_playlists.hdf', key='abc')
    df_playlists_info.to_hdf('df_data/df_playlists_info.hdf', key='abc')
    df_playlists_test.to_hdf('df_data/df_playlists_test.hdf', key='abc')
    df_playlists_test_info.to_hdf('df_data/df_playlists_test_info.hdf', key='abc')

if __name__=='__main__':
    parsed_tracks()
    # comple_tracks()
        
