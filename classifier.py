import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#for word embedding
import gensim
from gensim.models import Word2Vec

from convokit import Corpus

import time

start = time.time()

# con = sqlite3.connect("dataset_2015/database.sqlite")

# cur = con.cursor()

# X = []
# subreddits = ["funny", "gaming", "aww", "science", "movies"]
# for subreddit in subreddits:
#     query = "SELECT body, subreddit FROM May2015 WHERE subreddit='" + subreddit + "';"
#     for index, content in enumerate(cur.execute(query)):
#         body, subreddit = content
#         X.append([body, subreddit])
#         if index == 5000:
#             break

# df = pd.DataFrame(X, columns=["text", "label"])
X = []
subs = [
    "Cornell", 
    "IndianaUniversity",
    "lincoln",
    "msu",
    "Northwestern",
    "OSU",
    "PennStateUniversity",
    "Purdue",
    "rutgers",
    "uiowa",
    "UIUC",
    "UMD",
    "uofm",
    "uofmn",
    "UWMadison"
]

for sub in subs:
    count = 0
    sr = pd.read_json("CornellSubreddits/" + sub + ".corpus/utterances.jsonl", lines=True)
    for text in sr["text"]:
        # if count > 10000:
        #     break
        # if len(text) > 20:
        X.append([text, sub])
        count += 1
    end = time.time()
    print("Loaded " + sub + " in " + str(end - start) + "seconds")
df = pd.DataFrame(X, columns=["text", "label"])
end = time.time()
print(df.describe())
print("Datasets Loaded! time: " + str(end - start))


#-------------------preprocessing--------------------#

#convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

 
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)
#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
df['clean_text'] = df['text'].apply(lambda x: finalpreprocess(x))
end = time.time()
print("Dataset Cleaned! time: " + str(end - start))
print(df.describe())

#-------------------bayes multinomia--------------------#

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.clean_text).toarray()
labels = df.label
print(features.shape)
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], random_state = 0)


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# clf = MultinomialNB().fit(X_train_tfidf, y_train)
import pickle
# f = open('bigtenNBM.pickle', 'wb')
# pickle.dump(clf, f)
# f.close()

# print(clf.predict(count_vect.transform(["This dog is so cute"])))
# print(clf.predict(count_vect.transform(["have you even seen war dogs?"])))
# print(clf.predict(count_vect.transform(["I find sausage dogs absolutely hilarious"])))

# print("accuracy: " + str(clf.score(count_vect.transform(X_test), y_test)))

#-------------------4 models--------------------#

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
models = [
    # RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    # MultinomialNB(),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
  start = time.time()
  model_name = model.__class__.__name__
  clf = model.fit(X_train_tfidf, y_train)
  # accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  f = open('bigtenLinearSVC.pickle', 'wb')
  pickle.dump((tfidf, count_vect, clf), f)
  f.close()
  end = time.time()
  print(model_name + ": " + str(end - start))
  print("accuracy: " + str(clf.score(count_vect.transform(X_test), y_test)))
#   for fold_idx, accuracy in enumerate(accuracies):
#     entries.append((model_name, fold_idx, accuracy))
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
# import seaborn as sns
# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
#               size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.show()