# %%
import pandas as pd
from imblearn.over_sampling import SMOTE

# %%
from cgi import test
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.pipeline import Pipeline

from NPF.teachers_and_covid import start
from NPF.library import classify

FILE_NAME = start.MAIN_DIR + "model statistics victim.txt"
f = open(FILE_NAME, "w+")


# %%
tweets = pd.read_csv(start.MAIN_DIR + "tweets_full.csv")
tweets = tweets[
    [
        "unique_id",
        "text",
        "created",
        "likes",
        "retweets",
        "quotes",
        "replies",
        "author_id",
        "geo",
        "random_set",
    ]
]
annotations = pd.read_csv(start.MAIN_DIR + "annotations_characters.csv")


df = annotations.merge(
    tweets[["unique_id", "text", "random_set"]],
    how="left",
    left_on="unique_id",
    right_on="unique_id",
)

# %%
df = df.sample(len(annotations), random_state=68)
testing = df[df.random_set == 3]
training = df[df.random_set != 3]

testing_outcome = testing.victim
training_outcome = training.victim


count_vect = CountVectorizer(
    strip_accents="unicode",
    stop_words=classify.stop_words,
    ngram_range=(1, 3),
    min_df=10,
)
training_matrix = count_vect.fit_transform(training.text)
testing_matrix = count_vect.transform(testing.text)
# %%
oversample = SMOTE()
X, y = oversample.fit_resample(training_matrix, training_outcome)


# %%

# %%
clf = svm.LinearSVC()
clf = clf.fit(X, y)

training["classification_svm"] = clf.predict(training_matrix)
testing["classification_svm"] = clf.predict(testing_matrix)

print("")
print("SVM Classifier")
classify.print_statistics(
    classification=testing.classification_svm,
    ground_truth=testing_outcome,
    model_name="SVM Classifier",
)
# %%
