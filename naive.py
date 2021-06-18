import os
import numpy as np
import string
import pickle

import tensorflow as tf
import tensorflow_datasets as tfds
from official.nlp import optimization 

import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

import nltk
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer

# get stopwords
nltk.download('stopwords')
stopword = set(stopwords.words('english'))

def process_text(teksts):
    # lowercase
    teksts = teksts.lower()

    # remove punctuation
    teksts = teksts.translate(str.maketrans("","", string.punctuation))

    # remove stop words
    jauns_teksts = []
    for word in teksts.split(" "):
        if word in stopword:
            continue
        jauns_teksts.append(word)
    
    teksts = " ".join(jauns_teksts)


    return teksts


# training sentences
SENTENCES = 1200000

# validation sententces
VAL_SENTENCES = 400000

# get sentiment dataset
train_data, validation_data, test_data = tfds.load(
    name="sentiment140",
    split=('train[:80%]', 'train[80%:]', 'test'),
    as_supervised=True)


# get sentiment corpus
corpus = list()
corpus_sentiment = list()
unique = set()

training_data = train_data.take(SENTENCES)
# process tensorflow text to python text & label pair
for pair in training_data:
    teksts, sentiments = pair
    teksts = teksts.numpy().decode("UTF-8")
    sentiments = sentiments.numpy()

    teksts = process_text(teksts)

    unique.add(sentiments)
    corpus.append(teksts)
    corpus_sentiment.append(sentiments)

# get validation corpus
val_corpus = list()
val_corpus_sentiment = list()
for pair in validation_data.take(VAL_SENTENCES):
    teksts, sentiments = pair
    teksts = teksts.numpy().decode("UTF-8")
    sentiments = sentiments.numpy()

    teksts = process_text(teksts)

    val_corpus.append(teksts)
    val_corpus_sentiment.append(sentiments)

# count words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)


# BUILD NAIVE BAYES CLASSIFIER
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(corpus, corpus_sentiment)
filename = 'naive-classifier.pck'
pickle.dump(model, open(filename, 'wb'))

# test on validation data
predicted_categories = model.predict(val_corpus)

# create confusion matrix
mat = confusion_matrix(val_corpus_sentiment, predicted_categories)
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=["Negative", "Postive"], yticklabels=["Negative", "Postive"])
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()

print(f"Accuracy score for Naive Bayes: {accuracy_score(val_corpus_sentiment, predicted_categories)}")