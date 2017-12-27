import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('genrelyrics.csv')
data['word_count'] = data['word_count'].apply(ast.literal_eval)
dict_vect = DictVectorizer(sparse=False)
X = dict_vect.fit_transform(list(data['word_count']))

tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X_tf = tf_transformer.transform(X)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)

clf = MultinomialNB().fit(X, data.genre)

#Test phrase
docs_new = [ast.literal_eval("{'all': 1, 'help': 2, 'just': 6, 'soon': 1, 'through': 13, 'go': 13, 'yes': 2, 'givin': 2, 'how': 1, 'sorri': 3, 'forev': 1, 'comprehend': 1, 'better': 1, 'to': 16, 'els': 1, 'has': 1, 'might': 14, 'do': 25, 'good': 1, 'get': 2, 'gonna': 2, 'lovin': 1, 'know': 1, 'new': 1, 'not': 9, 'now': 3, 'tri': 12, 'somewher': 1, 'like': 14, 'pleas': 2, 'right': 1, 'yeah': 3, 'beg': 2, 'see': 5, 'are': 1, 'plead': 2, 'close': 1, 'folk': 1, 'what': 18, 'for': 6, 'leav': 1, 'got': 2, 'whatev': 1, 'be': 4, 'we': 1, 'sole': 1, 'never': 12, 'focus': 1, 'here': 1, 'blame': 2, 'let': 1, 'free': 1, 'along': 1, 'come': 1, 'chapter': 1, 'on': 1, 'last': 1, 'of': 1, 'mani': 1, 'wanna': 2, 'or': 1, 'love': 1, 'ca': 3, 'open': 1, 'your': 1, 'use': 1, 'eye': 1, 'live': 1, 'much': 4, 'too': 4, 'life': 1, 'that': 14, 'but': 15, 'understand': 1, 'nobodi': 1, 'me': 9, 'this': 6, 'gotta': 4, 'up': 2, 'will': 13, 'can': 2, 'my': 1, 'and': 20, 'have': 14, 'is': 3, 'am': 24, 'it': 32, 'as': 1, 'want': 2, 'need': 3, 'seem': 14, 'if': 1, 'selfish': 17, 'no': 1, 'alway': 4, 'role': 1, 'you': 25, 'noth': 1, 'play': 1, 'may': 3, 'who': 1, 'someon': 1, 'i': 87, 'think': 16, 'caus': 1, 'the': 1}")]

Z = dict_vect.transform(docs_new)
Z_new_counts = tf_transformer.transform(Z)
Z_new_tfidf = tfidf_transformer.transform(Z_new_counts)
predicted = clf.predict(Z_new_tfidf)
print predicted

X_train, X_test, y_train, y_test = train_test_split(data['word_count'], data.genre, test_size=0.33, random_state=42)

X = dict_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)
clf = MultinomialNB().fit(X, y_train)

Z = dict_vect.transform(X_test)
Z_new_counts = tf_transformer.transform(Z)
Z_new_tfidf = tfidf_transformer.transform(Z_new_counts)
predicted = clf.predict(Z_new_tfidf)
print accuracy_score(y_test, predicted) * 100
