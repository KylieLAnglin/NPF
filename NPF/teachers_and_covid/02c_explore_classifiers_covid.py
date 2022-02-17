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
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from NPF.teachers_and_covid import start
from NPF.library import classify

FILE_NAME = start.MAIN_DIR + "model statistics covid.txt"
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
annotations = pd.read_csv(start.MAIN_DIR + "annotations.csv")
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
df = df[~df.covid.isnull()]
testing = df[df.random_set == 3]
training = df[df.random_set == 4]

# %%
count_vect = CountVectorizer(
    strip_accents="unicode",
    stop_words=classify.stop_words,
    ngram_range=(1, 3),
    min_df=10,
)
train_matrix = count_vect.fit_transform(training.text)
train_matrix.shape

test_matrix = count_vect.transform(testing.text)
test_matrix.shape

# %% SVM
clf = svm.LinearSVC()
clf = clf.fit(train_matrix, training.covid)


testing["classification_svm"] = clf.predict(test_matrix)

print("")
print("SVM Classifier")
classify.print_statistics(
    classification=testing.classification_svm,
    ground_truth=testing.covid,
    model_name="SVM Classifier",
    file_name=FILE_NAME,
)

# %% Stochastic Gradient Descent

clf = SGDClassifier(random_state=87)
clf = clf.fit(train_matrix, training.covid)

testing["classification_sgd"] = clf.predict(test_matrix)
training["classification_sgd"] = clf.predict(train_matrix)

print("")
print("SGD Classifier")
classify.print_statistics(
    classification=testing.classification_sgd,
    ground_truth=testing.covid,
    model_name="SGD Classifier",
    file_name=FILE_NAME,
)
roc_auc_score(testing.covid, clf.decision_function(test_matrix))


# %% Random Forest
clf = RandomForestClassifier(random_state=2)
clf = clf.fit(train_matrix, training.covid)

testing["classification_rf"] = clf.predict(test_matrix)
training["classification_rf"] = clf.predict(train_matrix)

print("")
print("Random Forest Classifier")
classify.print_statistics(
    classification=testing.classification_rf,
    ground_truth=testing.covid,
    model_name="Random Forest Classifier",
    file_name=FILE_NAME,
)

# %% Bernoulli Naive Bayes
clf = BernoulliNB()
clf = clf.fit(train_matrix, training.covid)

testing["classification_nb"] = clf.predict(test_matrix)
training["classification_nb"] = clf.predict(train_matrix)

print("Naive Bayes")
classify.print_statistics(
    classification=testing.classification_nb,
    ground_truth=testing.covid,
    model_name="Naive Bayes",
    file_name=FILE_NAME,
)

# %%
clf = LogisticRegression(penalty="l2")
clf = clf.fit(train_matrix, training.covid)

testing["classification_ridge"] = clf.predict(test_matrix)
training["classification_ridge"] = clf.predict(train_matrix)

print("")
print("Ridge Classifier")
classify.print_statistics(
    classification=testing.classification_ridge,
    ground_truth=testing.covid,
    model_name="Ridge Classifier",
    file_name=FILE_NAME,
)

cf_matrix = confusion_matrix(testing.covid, testing.classification_ridge)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)


# %% SVM with Threshold
clf = svm.LinearSVC()
clf = clf.fit(train_matrix, training.covid)
precisions, recalls, thresholds = precision_recall_curve(
    training.covid, clf.decision_function(train_matrix)
)

threshold_recall = thresholds[np.argmax(recalls >= 0.85)]

testing["classification_svm_recall"] = (
    clf.decision_function(test_matrix) >= threshold_recall
)
training["classification_svm_recall"] = (
    clf.decision_function(train_matrix) >= threshold_recall
)

print("")
print("SVM Classifier with Threshold")
classify.print_statistics(
    classification=testing.classification_svm_recall,
    ground_truth=testing.covid,
    model_name="SVM Classifier with Threshold",
    file_name=FILE_NAME,
)


# %%

# %% Bernoulli Naive Bayes
clf = BernoulliNB()
clf = clf.fit(train_matrix, training.covid)

probas = [proba[1] for proba in clf.predict_proba(train_matrix)]
precisions, recalls, thresholds = precision_recall_curve(training.covid, probas)
threshold_recall = thresholds[np.argmax(recalls >= 0.70)]

testing["classification_nb_recall"] = [
    proba[1] for proba in clf.predict_proba(test_matrix)
] >= threshold_recall
training["classification_nb_recall"] = [
    proba[1] for proba in clf.predict_proba(train_matrix)
] >= threshold_recall

print("")
print("Naive Bayes Classifier with Threshold")
classify.print_statistics(
    classification=testing.classification_nb_recall,
    ground_truth=testing.covid,
    model_name="Naive Bayes Classifier with Threshold",
    file_name=FILE_NAME,
)

# %%

# %% Ridge
clf = LogisticRegression(penalty="l2")
clf = clf.fit(train_matrix, training.covid)

probas = [proba[1] for proba in clf.predict_proba(train_matrix)]
precisions, recalls, thresholds = precision_recall_curve(training.covid, probas)
threshold_recall = thresholds[np.argmax(recalls >= 0.70)]


testing["classification_ridge_recall"] = [
    proba[1] for proba in clf.predict_proba(test_matrix)
] >= threshold_recall
training["classification_ridge_recall"] = [
    proba[1] for proba in clf.predict_proba(train_matrix)
] >= threshold_recall

training["classification_ridge_recall_score"] = [
    proba[1] for proba in clf.predict_proba(train_matrix)
]

testing["classification_ridge_recall_score"] = [
    proba[1] for proba in clf.predict_proba(test_matrix)
]

print("")
print("Ridge Classifier with Threshold")
classify.print_statistics(
    classification=testing.classification_ridge_recall,
    ground_truth=testing.covid,
    model_name="Ridge Classifier with Threshold",
    file_name=FILE_NAME,
)

cf_matrix = confusion_matrix(testing.covid, testing.classification_ridge_recall)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)


roc_auc_score(testing.covid, [proba[1] for proba in clf.predict_proba(test_matrix)])

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
        ("clf", svm.LinearSVC()),
    ]
)

clf = pipeline.fit(training.text, training.covid)


f1s = []
recalls = []
sizes = [100, 200, 300, 400, 500, 600, 700, 800, 898]
for size in sizes:
    clf_size = pipeline.fit(
        training.text.sample(size, random_state=4),
        training.covid.sample(size, random_state=4),
    )
    recall = recall_score(testing.covid, clf_size.predict(testing.text))
    print(recall)
    f1 = f1_score(testing.covid, clf_size.predict(testing.text))
    f1s.append(f1)
    recalls.append(recall)

plt.plot(sizes, recalls)
plt.show()

# %%
