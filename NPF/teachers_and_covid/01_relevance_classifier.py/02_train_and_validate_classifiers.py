# %%
from cgi import test
import re
import pandas as pd
import numpy as np
import pickle

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

FILE_NAME = start.CLEAN_DIR + "relevance model statistics.txt"
f = open(FILE_NAME, "w+")


# %%
tweets = pd.read_csv(start.CLEAN_DIR + "tweets_full.csv")
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
annotations = pd.read_csv(start.CLEAN_DIR + "annotations.csv")
annotations = annotations[
    [
        "unique_id",
        "tweet_id",
        "random_set",
        "relevant",
        "category",
        "covid",
        "character",
    ]
]

df = annotations.merge(
    tweets[["unique_id", "text"]],
    how="left",
    left_on="unique_id",
    right_on="unique_id",
)

# %%
df = df.sample(len(annotations), random_state=68)
testing = df[df.random_set == 3]
training = df[df.random_set != 3]

# %%


# %% SVM
pipeline = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                strip_accents="unicode",
                stop_words=classify.stop_words,
                ngram_range=(1, 3),
                min_df=10,
            ),
        ),
        ("clf", svm.LinearSVC()),
    ],
)


clf = pipeline.fit(training.text, training.relevant)


testing["classification_svm"] = clf.predict(testing.text)
testing["score_svm"] = clf.decision_function(testing.text)

training["classification_csv"] = clf.predict(training.text)
training["score_svm"] = clf.decision_function(training.text)

print("")
print("SVM Classifier")
classify.print_statistics(
    classification=testing.classification_svm,
    ground_truth=testing.relevant,
    model_name="SVM Classifier",
    file_name=FILE_NAME,
)

pickle.dump(clf, open(start.TEMP_DIR + "model_svm.sav", "wb"))

precisions, recalls, thresholds = precision_recall_curve(
    training.relevant, clf.decision_function(training.text)
)

threshold_recall = thresholds[np.argmax(recalls >= 0.85)]

testing["classification_svm_recall"] = (
    clf.decision_function(testing.text) >= threshold_recall
)
training["classification_svm_recall"] = (
    clf.decision_function(training.text) >= threshold_recall
)

print("")
print("SVM Classifier with Threshold")
classify.print_statistics(
    classification=testing.classification_svm_recall,
    ground_truth=testing.relevant,
    model_name="SVM Classifier with Threshold",
    file_name=FILE_NAME,
)


# %% Stochastic Gradient Descent
pipeline = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                strip_accents="unicode",
                stop_words=classify.stop_words,
                ngram_range=(1, 3),
                min_df=10,
            ),
        ),
        ("clf", SGDClassifier(random_state=87)),
    ],
)
clf = pipeline.fit(training.text, training.relevant)

testing["classification_sgd"] = clf.predict(testing.text)
testing["score_sgd"] = clf.decision_function(testing.text)

training["classification_sgd"] = clf.predict(training.text)
training["score_sgd"] = clf.decision_function(training.text)

print("")
print("SGD Classifier")
classify.print_statistics(
    classification=testing.classification_sgd,
    ground_truth=testing.relevant,
    model_name="SGD Classifier",
    file_name=FILE_NAME,
)
pickle.dump(clf, open(start.TEMP_DIR + "model_sgd.sav", "wb"))


# %% Random Forest
pipeline = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                strip_accents="unicode",
                stop_words=classify.stop_words,
                ngram_range=(1, 3),
                min_df=10,
            ),
        ),
        ("clf", RandomForestClassifier(random_state=2)),
    ],
)

clf = pipeline.fit(training.text, training.relevant)

testing["classification_rf"] = clf.predict(testing.text)
testing["score_rf"] = [proba[1] for proba in clf.predict_proba(testing.text)]

training["classification_rf"] = clf.predict(training.text)
training["score_rf"] = [proba[1] for proba in clf.predict_proba(training.text)]

print("")
print("Random Forest Classifier")
classify.print_statistics(
    classification=testing.classification_rf,
    ground_truth=testing.relevant,
    model_name="Random Forest Classifier",
    file_name=FILE_NAME,
)
pickle.dump(clf, open(start.TEMP_DIR + "model_rf.sav", "wb"))

# %% Bernoulli Naive Bayes
pipeline = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                strip_accents="unicode",
                stop_words=classify.stop_words,
                ngram_range=(1, 3),
                min_df=10,
            ),
        ),
        ("clf", BernoulliNB()),
    ],
)

clf = pipeline.fit(training.text, training.relevant)

testing["classification_nb"] = clf.predict(testing.text)
testing["score_nb"] = [proba[1] for proba in clf.predict_proba(testing.text)]


training["classification_nb"] = clf.predict(training.text)
training["score_nb"] = [proba[1] for proba in clf.predict_proba(training.text)]

print("Naive Bayes")
classify.print_statistics(
    classification=testing.classification_nb,
    ground_truth=testing.relevant,
    model_name="Naive Bayes",
    file_name=FILE_NAME,
)
pickle.dump(clf, open(start.TEMP_DIR + "model_nb.sav", "wb"))


probas = [proba[1] for proba in clf.predict_proba(training.text)]
precisions, recalls, thresholds = precision_recall_curve(training.relevant, probas)
threshold_recall = thresholds[np.argmax(recalls >= 0.70)]

testing["classification_nb_recall"] = [
    proba[1] for proba in clf.predict_proba(testing.text)
] >= threshold_recall
training["classification_nb_recall"] = [
    proba[1] for proba in clf.predict_proba(training.text)
] >= threshold_recall

print("")
print("Naive Bayes Classifier with Threshold")
classify.print_statistics(
    classification=testing.classification_nb_recall,
    ground_truth=testing.relevant,
    model_name="Naive Bayes Classifier with Threshold",
    file_name=FILE_NAME,
)
# %%
pipeline = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                strip_accents="unicode",
                stop_words=classify.stop_words,
                ngram_range=(1, 3),
                min_df=10,
            ),
        ),
        ("clf", LogisticRegression(penalty="l2")),
    ],
)
clf = pipeline.fit(training.text, training.relevant)

testing["classification_ridge"] = clf.predict(testing.text)
testing["score_ridge"] = [proba[1] for proba in clf.predict_proba(testing.text)]

training["classification_ridge"] = clf.predict(training.text)
training["score_ridge"] = [proba[1] for proba in clf.predict_proba(training.text)]

print("")
print("Ridge Classifier")
classify.print_statistics(
    classification=testing.classification_ridge,
    ground_truth=testing.relevant,
    model_name="Ridge Classifier",
    file_name=FILE_NAME,
)

cf_matrix = confusion_matrix(testing.relevant, testing.classification_ridge)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)

pickle.dump(clf, open(start.TEMP_DIR + "model_ridge.sav", "wb"))


probas = [proba[1] for proba in clf.predict_proba(training.text)]
precisions, recalls, thresholds = precision_recall_curve(training.relevant, probas)
threshold_recall = thresholds[np.argmax(recalls >= 0.70)]


testing["classification_ridge_recall"] = [
    proba[1] for proba in clf.predict_proba(testing.text)
] >= threshold_recall
training["classification_ridge_recall"] = [
    proba[1] for proba in clf.predict_proba(training.text)
] >= threshold_recall

training["classification_ridge_recall_score"] = [
    proba[1] for proba in clf.predict_proba(training.text)
]

testing["classification_ridge_recall_score"] = [
    proba[1] for proba in clf.predict_proba(testing.text)
]

print("")
print("Ridge Classifier with Threshold")
classify.print_statistics(
    classification=testing.classification_ridge_recall,
    ground_truth=testing.relevant,
    model_name="Ridge Classifier with Threshold",
    file_name=FILE_NAME,
)

cf_matrix = confusion_matrix(testing.relevant, testing.classification_ridge_recall)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)


roc_auc_score(testing.relevant, [proba[1] for proba in clf.predict_proba(testing.text)])

# %%
training.to_csv(start.TEMP_DIR + "training_models.csv")
testing.to_csv(start.TEMP_DIR + "testing_models.csv")
