import string
import sys
import pickle
import nltk
from nltk.corpus import stopwords

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

from absl import app
from absl import flags

# get stopwords from nltk
stopword = set(stopwords.words('english'))

FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, "String to be processed by Naive Bayes Classifier")

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


def main(argv):
    if FLAGS.input is None:
        print("Please provide an input string")
        sys.exit()

    # load saved model
    model = pickle.load(open("naive-classifier.pck", 'rb'))

    input_string = FLAGS.input
    input_string = process_text(input_string)

    # predict model
    prediction = model.predict([input_string])[0]

    predicted_class = "Negative" if prediction == 0 else "Positive"

    print(f"Text: {FLAGS.input}")
    print(f"Processed text: {input_string}")
    print(f"Sentiment: {predicted_class}")

if __name__ == '__main__':
    app.run(main)