import numpy as np
import h5py
import pandas as pd
import os
import fnmatch

matches = list()
df_final = pd.DataFrame()
f= open("~/MSD.csv","w")
for root,dirnames,filenames in os.walk('/scratch/work/public/MillionSongDataset/data'):
    for filename in fnmatch.filter(filenames, '*.h5'):
        #print(os.path.join(root, filename))
        hdf = pd.HDFStore(os.path.join(root, filename),mode ='r')
        df1 = hdf.get('/analysis/songs/')
        df2 = hdf.get('/metadata/songs/')
        df3 = hdf.get('/musicbrainz/songs/')
        df = pd.concat([df1,df2,df3], axis = 1, join_axes = [df1.index])
        df.to_csv(f,mode='a', header=False)
        hdf.close()
f.close()

