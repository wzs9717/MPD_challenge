import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from co_occurence_features import create_co_occurence_features

def data_gene():
    df_m=pd.read_csv('./new_data/pl_track_map_test.csv')#
    df_p=pd.read_csv("./new_data/play_list.csv")
    df_t=pd.read_csv("./new_data/tracks.csv")
    df_map=df_m.head(5000)
  
    df_map["target"]=1
    df_a=df_m.tail(2000)#这个是随机抽样Pos的那个
    df_b=df_map.head(1000)#这个是不变的那个pid
    df_c=df_m.loc[1001:3000,['tid']]#tid
    df_a=resample(df_a.values,n_samples=1000)
    df_a=pd.DataFrame(df_a)
    df_c=resample(df_c.values,n_samples=1000)
    df_test=pd.DataFrame(df_b['pid'],columns=["pid"])
    #df_test[['tid']]=df_c
    df_test['tid']=df_c
    df_test['pos']=df_a[2]
    df_test['target']=0
    df_total=pd.concat((df_map,df_test),axis=0) 
    df_p_map=df_total.merge(df_p, left_on='pid',right_on='pid').merge(df_t, left_on='tid',right_on='tid')
    track_map=create_co_occurence_features(df_p_map)
   
    #print(df_p_map)
    df_p_map.to_csv('./whole_feature.csv')
    #---------------------------------------------------------------------------
    #预处理
    FEATUREs_drop=['album_uri','artist_uri','track_uri']
    FEATUREs_name=['name','album_name','artist_name','track_name']
    df_p_map=df_p_map.drop(FEATUREs_drop,axis=1)
    df_p_map=df_p_map.drop(FEATUREs_name,axis=1)
    #df_p_map=pd.get_dummies(df_p_map,dummy_na=True,columns=FEATUREs_encoding)
    df_p_map=df_p_map.drop(['pid','tid'],axis=1)
    y=df_p_map['target']
    X=df_p_map.drop('target',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train,X_test,y_train,y_test
    
if __name__=='__main__':
    
    data_gene()
    
