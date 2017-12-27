import pandas as pd
import os, ast, glob
from collections import Counter
import operator

allFiles = glob.glob('genrelocmap2/*')

frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)
frame = frame[['genre','track_id','track_id','artist_latitude','artist_longitude']]
frame
frame.to_csv("glm.csv")
