# %%
import re
import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start
from NPF.library import classify

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

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
# %%
annotations1 = pd.read_csv(
    start.MAIN_DIR + "training_batch1_annotated.csv", encoding="utf-8’"
)
annotations2 = pd.read_csv(
    start.MAIN_DIR + "training_batch2_annotated.csv", encoding="utf-8’"
)

annotations = pd.concat([annotations1, annotations2])

# %%
df = annotations[(annotations.relevant == 0) | (annotations.relevant == 1)]
df.sample(len(df), random_state=67)
testing = df.head(150)
training = df.tail(len(df) - 150)

# %%
FILE_NAME = start.MAIN_DIR + "model statistics.txt"
f = open(FILE_NAME, "w+")

# %%
def print_statistics(classification, ground_truth, model_name: str):
    file_object = open(FILE_NAME, "a")
    file_object.write(model_name)
    file_object.write("\n")

    precision = precision_score(y_true=ground_truth, y_pred=classification)
    recall = recall_score(y_true=ground_truth, y_pred=classification)
    f1 = f1_score(y_true=ground_truth, y_pred=classification)

    print(
        "Precision = ",
        str(round(precision, 2)),
    )
    file_object.write("Precision = " + str(round(precision, 2)))
    file_object.write("\n")

    print(
        "Recall = ",
        str(round(recall, 2)),
    )
    file_object.write(
        "Recall = " + str(round(recall, 2)),
    )
    file_object.write("\n")

    print(
        "F1 = ",
        str(round(f1, 2)),
    )
    file_object.write("F1 = " + str(round(f1, 2)))
    file_object.write("\n")
    file_object.write("\n")
    file_object.write("\n")

    file_object.close()


# %%
count_vect = CountVectorizer(
    strip_accents="unicode",
    max_features=500,
)
X_train_counts = count_vect.fit_transform(training.text)
X_train_counts.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# %% SVM
clf = svm.LinearSVC()
clf = clf.fit(X_train_tfidf, training.relevant)

X_test_counts = count_vect.transform(testing.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

testing["classification_svm"] = clf.predict(X_test_tfidf)
training["classification_svm"] = clf.predict(X_train_tfidf)

print("")
print("SVM Classifier")
print_statistics(testing.classification_svm, testing.relevant, "SVM Classifier")

# %% Stochastic Gradient Descent

clf = SGDClassifier(random_state=87)
clf = clf.fit(X_train_tfidf, training.relevant)

X_test_counts = count_vect.transform(testing.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

testing["classification_sgd"] = clf.predict(X_test_tfidf)
training["classification_sgd"] = clf.predict(X_train_tfidf)

print("")
print("SGD Classifier")
print_statistics(testing.classification_sgd, testing.relevant, "SGD Classifier")


# %% Random Forest
clf = RandomForestClassifier(random_state=2)
clf = clf.fit(X_train_tfidf, training.relevant)

X_test_counts = count_vect.transform(testing.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

testing["classification_rf"] = clf.predict(X_test_tfidf)
training["classification_rf"] = clf.predict(X_train_tfidf)

print("")
print("Random Forest Classifier")
print_statistics(
    testing.classification_rf, testing.relevant, "Random Forest Classifier"
)

# %% Bernoulli Naive Bayes
clf = BernoulliNB()
clf = clf.fit(X_train_tfidf, training.relevant)

X_test_counts = count_vect.transform(testing.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

testing["classification_nb"] = clf.predict(X_test_tfidf)
training["classification_nb"] = clf.predict(X_train_tfidf)

print("")
print("Naive Bayes Classifier")
print_statistics(testing.classification_nb, testing.relevant, "Naive Bayes Classifier")

# %%
clf = LogisticRegression(penalty="l2")
clf = clf.fit(X_train_tfidf, training.relevant)

X_test_counts = count_vect.transform(testing.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

testing["classification_lasso"] = clf.predict(X_test_tfidf)
training["classification_lasso"] = clf.predict(X_train_tfidf)

print("")
print("Lasso Classifier")
print_statistics(testing.classification_lasso, testing.relevant, "Lasso Classifier")

# %% SVM with Threshold
clf = svm.LinearSVC()
clf = clf.fit(X_train_tfidf, training.relevant)
precisions, recalls, thresholds = precision_recall_curve(
    training.relevant, clf.decision_function(X_train_tfidf)
)

threshold_recall = thresholds[np.argmax(recalls >= 0.80)]

testing["classification_svm_recall"] = (
    clf.decision_function(X_test_tfidf) >= threshold_recall
)
training["classification_svm_recall"] = (
    clf.decision_function(X_train_tfidf) >= threshold_recall
)

print("")
print("SVM Classifier with Threshold")
print_statistics(
    testing.classification_svm_recall,
    testing.relevant,
    "SVM Classifier with Threshold",
)


training.classification_svm.mean()

testing["accurate"] = np.where(testing.relevant == testing.classification_svm, 1, 0)
testing.accurate.mean()
