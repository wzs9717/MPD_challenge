import numpy as np
import pandas as pd 

from model import TwoStageModel
def main():
    # // load data
    df_tracks=read_csv("./new_data/tracks.csv")
    df_pl=read_csv("./new_data/play_list.csv")
    df_pl_track_map=read_csv("./new_data/pl_track_map.csv")

    df_pl_test=read_csv("./new_data/play_list_test.csv")
    df_pl_track_map_test=read_csv("./new_data/pl_track_map_test.csv")

    train_map = None
    val1_map = None
    val1_pids = None
    val2_map = None
    val2_pids = None

    # // train first stage model
    model=TwoStageModel()
    model.train(df_pl,df_tracks,train_map,val1_map,val2_map,val1_pids,val2_pids)
    # // train second stage model

if __name__ == "__main__":
    main()