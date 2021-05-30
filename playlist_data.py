import pandas as pd
import json
import numpy as np
import codecs
import glob
def parsed_tracks():
    file_name='./data/train/*.json'
    files = glob.glob(file_name)                      
    df_tracks_all=None
    for file in files:
        data = json.load(open(file))
        df = pd.DataFrame(data["playlists"])
        df=df.drop(["pid","description"],axis=1)
        if  df_tracks_all is None:
            df_tracks_all=df
        else:
            df_tracks_all=pd.concat([df_tracks_all,df],ignore_index=True)