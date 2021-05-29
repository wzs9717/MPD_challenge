
import pandas as pd
import json
file_name='./data/mpd.slice.0-999.json'
output_name='res.csv'

data = json.load(open(file_name))
df = pd.DataFrame(data["playlists"]).head(10)

df=df.drop(["pid","description"],axis=1)

df.to_csv(output_name, index = None)