
import pandas as pd
import json
import numpy as np
import codecs
import glob
def parsed_tracks():
    file_name='./data/train/*.json'
    files = glob.glob(file_name)                    
   
    df_tracks_all=None
    c=0
    for file in files:
        print(c)
        data = json.load(open(file))
        df = pd.DataFrame(data["playlists"])
       
        track_list=df[["tracks"]].values.tolist()
        
        
        track_list_flatten=[]
        for i in track_list:
            for j in i[0]:
                    track_list_flatten.append(j)
        df_tracks=pd.DataFrame(track_list_flatten)
        df_tracks=df_tracks.drop(['pos','album_uri','artist_uri'],axis=1)
        df_tracks_unique=df_tracks.drop_duplicates(subset=['track_uri'])
        if  df_tracks_all is None:
            df_tracks_all=df_tracks_unique
        else:
            df_tracks_all=pd.concat([df_tracks_all,df_tracks_unique],ignore_index=True)
        if c%50==0:
            df_tracks_all=df_tracks_all.drop_duplicates(subset=['track_uri'])
            df_tracks_all.to_csv("./new_data/track%s.csv"%c, index = None)
            df_tracks_all=None
        c+=1
        
    df_tracks_all=df_tracks_all.drop_duplicates(subset=['track_uri'])
    df_tracks_all.to_csv("./new_data/track%s.csv"%c, index = None)
def comple_tracks():
    file_cname='./new_data/*.csv'
    files_c=glob.glob(file_cname)
    df_ctracks_all=pd.read_csv(files_c[0])
    for i in files_c[1:]:
        tem=pd.read_csv(i)
        df_ctracks_all=pd.concat([df_ctracks_all,tem])
    df_ctracks_all=df_ctracks_all.drop_duplicates(subset=['track_uri']).drop('track_name',axis=1)
   
    df_ctracks_all.reset_index(drop=True)
    df_ctracks_all.to_csv("./new_data/track.csv")
    
if __name__=='__main__':
    #parsed_tracks()
    comple_tracks()
        
