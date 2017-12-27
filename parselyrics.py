import csv

fname = 'mxm_dataset_train.txt'
with open(fname) as f:
  content = f.readlines()

for c in content:
	if c[0] == '#':
		continue
	elif c[0] == '%':
		words = c[1:].strip().split(',')
		break

track_lyrics = []

for c in content:
	track = []
	if c[0] == '#' or c[0] == '%':
		continue
	else:
		word_bag = {}
		song = c.strip().split(',')
		track.append(song[0])
		track.append(song[1])
		for i in xrange(2,len(song)):
			s = song[i].strip().split(':')
			word_bag[words[int(s[0])-1]] = int(s[1])
		track.append(word_bag)
	track_lyrics.append(track)

i = 1
with open("output.csv", "wb") as f:
  writer = csv.writer(f)
  if i == 1:
  	writer.writerow(["track_id", "mxm_track_id", "word_count"])
  	i = 0
  writer.writerows(track_lyrics)
