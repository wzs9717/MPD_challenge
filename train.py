import numpy as np
import pandas as pd 
import os
from model import TwoStageModel
def main():
    if not os.path.exists('./res'):
        os.makedirs('./res')
    # // load data
    df_tracks=pd.read_csv("./new_data/tracks.csv")
    df_pl=pd.read_csv("./new_data/play_list.csv")
    df_pl_track_map=pd.read_csv("./new_data/pl_track_map.csv")

    df_pl_test=pd.read_csv("./new_data/play_list_test.csv")
    df_pl_track_map_test=pd.read_csv("./new_data/pl_track_map_test.csv")

    train_map = pd.read_csv("./new_data/train.csv")
    val1_map = pd.read_csv("./new_data/val1.csv")
    val1_pids = pd.read_csv("./new_data/val1_pids.csv")
    val2_map = pd.read_csv("./new_data/val2.csv")
    val2_pids = pd.read_csv("./new_data/val2_pids.csv")

    # // train first stage model
    model=TwoStageModel()
    model.train(df_pl,df_tracks,train_map,val1_map,val2_map,val1_pids,val2_pids,df_pl_test)
    # // train second stage model

if __name__ == "__main__":
    main()