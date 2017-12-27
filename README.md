# Million Song Dataset Analysis

## Overview
This project on the publicly available Million Song Dataset aims to address three separate questions - recommending songs to a user based on his play history, visualizing trends in music across the years and finally predicting the genre of an unknown song based on its lyrics. The main aim of the project is not to provide precise and groundbreaking results, but to use the concepts of Big Data Analytics on large data that can not be analyzed and processed using conventional methods.

Folder consists of the following code files:-
* makecsv.py : script to cache the dataset into a csv file
* process.scala : code to process and join all datasets and save the necessary fields
* optimum.scala : find the optimal number of clusters and plot the elbow plot
* clustering.scala : main clustering algorithm and recommender system code
* concat.py : parse output files to single file
* prediction.py : genre prediction model
* parselyrics.py : convert the lyrics to dictionary format
* combine.py : parse output files and select fields to plot locaitons
* analyasis.py : parse output files and select fields to plot word clouds into a single file

## Dataset
The main dataset for the project is the Million Song Dataset (MSD), a freely-available collection of audio features and metadata for a million contemporary popular music tracks. The dataset consists of over 50 features and metadata extracted from the song audio. The dataset is stored in a hierarchal HDF5 format and is available to download from the Open Science Data Cloud and AWS.

Additionally, the following 3 auxiliary datasets were used:-
* MusiXmatch Lyrics Dataset : lyrics (where applicable) for the above available as an indexed data
structure
* TU Wien Genre Dataset : categorization of the above dataset into 21 different genres
* Echonest User Datset : song play history for over 1 million users
The size of all the datasets is 300GB, too large for conventional processing.

## Data Processing :
To load the Million Song Dataset, the HDF5 files in the directory structure had to be read into a Spark dataframe. Only the dataset files in the HDF5 files were read and the group and metadata files were ignored. This step took over 6 hours to complete. Further cleaning had to be done by removing incomplete records and records with too many empty features. The lyrics dataset consisted of lyrics as a bag of words (the original lyric files are copyright protected)
indexed to a word list. To be able to use this, the indexed structure had to be flattened into a dictionary {word : count} format for each track. No processing had to be done for the user and genre datasets.

## Feature Engineering :
We only considered the following features for our use : TrackID, Artist name, Artist location, User play count, Genre, Lyrics, Mode (major/minor), Song Duration, Song Loudness, Song hotness, Artist hotness, Tempo (in bpm), Time Signature and Time Signature Confidence. As some of these features were available across different dataframes, they had to be joined together to a single dataframe. Additionally, rows with a time signature confidence threshold less than 0.5 had to be dropped because they were filled with garbage values which was not accurate. Another feature “Speed” had to be calculated from each tracks’ tempo and time signature by calculating the number of beats per measure rather than the number of beats per time (tempo). This is done to normalize how “fast” a song sounds to human ears. Natural Language Processing techniques like standardizing (removing non-english and other irrelevant characters) and stop word removal (remove frequent meaningless words like ‘I’, ‘the’, ‘a’, etc.) were also done on the lyrics to preserve only the words with actual meaning.

## Song Recommender System :
We wanted to build a system to recommend new songs to a user based on his tastes and listening history. The traditional approach of collaborative filtering was discarded because we do not have a metric of user ratings (only play counts) and song features would not be used in this case. To solve this problem, we used a clustering technique (k-means) in order to group songs together, which would incorporate song features as well as reduce the search space while recommending new songs. The model was trained on the following features : mode, duration, loudness, genre, speed and hotness. Each of the features had to be cleaned to remove false outliers and then min-max normalized (scaled) to a value between 1 and 5. The optimal number of clusters was found by calculating the sum of square errors within each cluster for different values of k. The optimal value was found to be 7. We now look at the user history and retrieve his/her 5 most played songs. For each of these, we search
within the corresponding cluster for the 3 most similar songs. The similarity measure used is the Manhattan distance between scaled features. The 15 obtained results are ranked based on the user’s favorite genres and similarity scores provided for each. Limitations for the above approach include that it is not suitable for dynamic data (retraining the model) and that there is no objective way to measure performance.

## Trend Analysis :
By grouping together songs, lyrics and locations, an exploratory analysis was performed to visualize the trends in music across the years. Lyrics were grouped by genre and word clouds (most frequent word visualizations) were made for each of the top genres. The results were consistent with our domain knowledge - for example - religious songs tend to have more mentions of “lord” and pop songs “love”. Locations were grouped by genre and plotted on a world map in order to visualize the geographical trends in music. The results were once again consistent - for example - country music was heavily concentrated in Southern USA, electronic music in urban Europe and USA and reggae music in the
Caribbean.

## Genre Prediction :
The final problem we worked involves predicting the genre of a song based on an input phrase. The machine learning model used in this case was a Naive Bayes model (since we just have word counts and no contextual information). The lyric features were already processed and vectorized (see ‘Feature Engineering’ above). The model was trained on 70% of the data and tested on the remaining 30%. The accuracy of the model was 74%. However, overfitting may have been a problem and the genre labels were not uniformly distributed across all songs.