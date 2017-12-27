import pandas as pd
import os, ast, glob
from collections import Counter
import operator
import csv

allFiles = glob.glob('lygn/*')

frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)
frame = frame[['genre', 'word_count']]

# Change this for different genres
g = "Religious"

frame = frame[frame['genre']==g]
frame['word_count'] = frame['word_count'].apply(ast.literal_eval)

d = frame['word_count'].tolist()
final = Counter()
for i in d:
    final += Counter(i)
final = dict(final)

from stop_words import get_stop_words
stop_words = get_stop_words('english')
for i in stop_words:
    try:
        del final[i]
    except KeyError:
        pass

popular = sorted(final.items(), key=operator.itemgetter(1), reverse=True)

with open(g+'_words.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['word','count'])
    for row in popular:
        csv_out.writerow(row)
