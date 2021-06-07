import pandas as pd
import numpy as np
from sklearn.utils import resample

def data_generater():
    df_m=pd.read_csv('./new_data/pl_track_map_test.csv')
    df_p=pd.read_csv("./new_data/play_list.csv")
    df_t=pd.read_csv("./new_data/tracks.csv")
    df_map=df_m.head(5000)
    # df_map.to_csv("./new_data/pl_track_map_test.csv")
    # df_p=df_p.head(10)
    # df_t=df_t.head(10)
  
    FEATUREs_m=['pid','tid','pos']
    FEATUREs_p=['collaborative','duration_ms','modified_at','name','num_albums','num_artists','num_edits','num_followers','num_tracks','pid']
    FEATUREs_t=['album_name','album_uri','artist_name','artist_uri','duration_ms','track_name','track_uri','tid']
    # df_p_feature=pd.concat([df_p.loc[df_p.pid==i] for i in df_map.pid],axis=0)
    # df_t_feature=pd.concat([df_t.loc[df_t.tid==i] for i in df_map.tid],axis=0)
    
    df_p_map=df_map.merge(df_p, left_on='pid',right_on='pid').merge(df_t, left_on='tid',right_on='tid')
    #df_p_map=df_p_map.drop(["Unnamed: 0"],axis=1)
    #df_p_map_t=pd.merge(df_p_feature, df_map, on='p)
    #df_p_map.to_csv('./123.csv')
    df_p_map["target"]=1
    df_a=df_m.tail(2000)#这个是随机抽样Pos的那个
    df_b=df_map.head(1000)#这个是不变的那个pid
    df_c=df_m.loc[1001:3000,['tid']]#tid
    df_a=resample(df_a.values,n_samples=1000)
    df_a=pd.DataFrame(df_a)
    df_c=resample(df_c.values,n_samples=1000)
    df_test=pd.DataFrame()
    df_test[['tid']]=df_c
    df_test['pid']=df_b['pid']
    df_test['pos']=df_a['pos']
    
    print(df_p_map)
    
if __name__=='__main__':
     data_generater()