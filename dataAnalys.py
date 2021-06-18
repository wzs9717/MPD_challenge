
import pandas as pd
import json
import numpy as np
import codecs
import glob
import os
def parsed_tracks_train():
    file_name='./data/train/*.json'
    files = glob.glob(file_name)                    
   
    Feature_playlist = ['collaborative', 'duration_ms', 'modified_at', 
                'name', 'num_albums', 'num_artists', 'num_edits',
                'num_followers', 'num_tracks', 'pid']
    Feature_tracks = ['album_name', 'album_uri', 'artist_name', 'artist_uri', 
                  'duration_ms', 'track_name', 'track_uri'] 
    Feature_playlist_test = ['name', 'num_holdouts', 'num_samples', 'num_tracks', 'pid']

    if not os.path.exists('./new_data'):
        os.makedirs('./new_data')

    tracks_all=[]
    pl_track_map=[]#[pid,tid,pos]
    pl_all=[]
    c=1
    for file in files[:]:#
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
    return track_uri2tid


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

def parse_tracks_val():

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
    df_playlists_test.to_csv("./new_data/play_list_test.csv", index = None)
    df_pl_track_map.to_csv("./new_data/pl_track_map_test.csv", index = None)


if __name__=='__main__':
    # parsed_tracks_train()
    parse_tracks_val()


        
