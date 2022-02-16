# %%
from operator import neg
import re
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from NPF.teachers_and_covid import start
from NPF.library import classify

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

df = df.sample(len(annotations), random_state=68)
testing = df[df.random_set == 3]
training = df[df.random_set != 3]

X_train = training.text
X_test = testing.text
y_train = training.relevant
y_test = testing.relevant
# %%
feature_df = pd.read_csv(start.MAIN_DIR + "feature_importance_annotated.csv")
positive_features = feature_df[(feature_df.importance > 0) & (feature_df.keep == 1)]

negative_features = feature_df[feature_df.importance < 0]
negative_features = negative_features[~negative_features.term.str.contains(" ")]

features = positive_features.append(negative_features)


# %%
count_vect = CountVectorizer(
    stop_words=classify.stop_words,
    ngram_range=(1, 3),
    vocabulary=features.term,
)
X_train_counts = count_vect.fit_transform(training.text)
X_train_counts.shape


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# %%
FILE_NAME = start.MAIN_DIR + "model statistics hand selected features.txt"
f = open(FILE_NAME, "w+")

# %% SVM
clf = svm.LinearSVC()
clf = clf.fit(X_train_tfidf, training.relevant)

X_test_counts = count_vect.transform(testing.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

testing["classification_svm"] = clf.predict(X_test_tfidf)
training["classification_svm"] = clf.predict(X_train_tfidf)

print("")
print("SVM Classifier")
classify.print_statistics(
    classification=testing.classification_svm,
    ground_truth=testing.relevant,
    model_name="SVM Classifier",
    file_name=FILE_NAME,
)


# %% Stochastic Gradient Descent

clf = SGDClassifier(random_state=87)
clf = clf.fit(X_train_tfidf, training.relevant)

X_test_counts = count_vect.transform(testing.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

testing["classification_sgd"] = clf.predict(X_test_tfidf)
training["classification_sgd"] = clf.predict(X_train_tfidf)

print("")
print("SGD Classifier")
classify.print_statistics(
    classification=testing.classification_sgd,
    ground_truth=testing.relevant,
    model_name="SGD Classifier",
    file_name=FILE_NAME,
)


# %% Random Forest
clf = RandomForestClassifier(random_state=2)
clf = clf.fit(X_train_tfidf, training.relevant)

X_test_counts = count_vect.transform(testing.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

testing["classification_rf"] = clf.predict(X_test_tfidf)
training["classification_rf"] = clf.predict(X_train_tfidf)

print("")
print("Random Forest Classifier")
classify.print_statistics(
    classification=testing.classification_rf,
    ground_truth=testing.relevant,
    model_name="Random Forest Classifier",
    file_name=FILE_NAME,
)

# %% Bernoulli Naive Bayes
clf = BernoulliNB()
clf = clf.fit(X_train_tfidf, training.relevant)

X_test_counts = count_vect.transform(testing.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

testing["classification_nb"] = clf.predict(X_test_tfidf)
training["classification_nb"] = clf.predict(X_train_tfidf)

print("Naive Bayes")
classify.print_statistics(
    classification=testing.classification_nb,
    ground_truth=testing.relevant,
    model_name="Naive Bayes",
    file_name=FILE_NAME,
)

# %%
clf = LogisticRegression(penalty="l2")
clf = clf.fit(X_train_tfidf, training.relevant)

X_test_counts = count_vect.transform(testing.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

testing["classification_lasso"] = clf.predict(X_test_tfidf)
training["classification_lasso"] = clf.predict(X_train_tfidf)

print("")
print("Lasso Classifier")
classify.print_statistics(
    classification=testing.classification_lasso,
    ground_truth=testing.relevant,
    model_name="Lasso Classifier",
    file_name=FILE_NAME,
)


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
classify.print_statistics(
    classification=testing.classification_svm_recall,
    ground_truth=testing.relevant,
    model_name="SVM Classifier with Threshold",
    file_name=FILE_NAME,
)


# %%
X_train = training.text
X_test = testing.text
y_train = training.relevant
y_test = testing.relevant
# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                ngram_range=(1, 3),
                vocabulary=features.term,
            ),
        ),
        ("tfidf", TfidfTransformer()),
        ("clf", svm.LinearSVC(random_state=593)),
    ]
)

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    "vect__binary": (True, False),
    "tfidf__use_idf": (True, False),
    "tfidf__norm": ("l1", "l2"),
    "clf__C": (1, 10, 100),
    "clf__loss": ("hinge", "squared_hinge"),
    "clf__dual": (True, False),
}

grid_search = classify.grid_search(pipeline, parameters, X_train, y_train)

# %%

FILE_NAME = start.MAIN_DIR + "Params_SVM_tailored.txt"
f = open(FILE_NAME, "w+")
file_object = open(FILE_NAME, "a")
file_object.write(str(grid_search.best_params_))


with open(start.MAIN_DIR + "Params_SVM_tailored.pickle", "wb") as handle:
    pickle.dump(grid_search.best_params_, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%

with open(start.MAIN_DIR + "Params_SVM_tailored.pickle", "rb") as handle:
    SVM_params = pickle.load(handle)

print(SVM_params)

pipeline_svm = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                binary=SVM_params["vect__binary"], vocabulary=features.term
            ),
        ),
        (
            "tfidf",
            TfidfTransformer(
                # norm=SVM_params["tfidf__norm"],
                # use_idf=SVM_params["tfidf__use_idf"],
            ),
        ),
        (
            "clf",
            svm.LinearSVC(
                C=SVM_params["clf__C"],
                loss=SVM_params["clf__loss"],
                dual=SVM_params["clf__dual"],
                random_state=593,
            ),
        ),
    ]
)

clf_svm = pipeline_svm.fit(X_train, y_train)
pickle.dump(clf_svm, open(start.MAIN_DIR + "model_SVM_tailored.sav", "wb"))


# %%
clf_svm = pickle.load(open(start.MAIN_DIR + "model_SVM_tailored.sav", "rb"))

y_test_predict = clf_svm.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_test_predict)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)
recall_score(y_test, y_test_predict)
f1_score(y_test, y_test_predict)

testing["classification_final"] = y_test_predict

precisions, recalls, thresholds = precision_recall_curve(
    y_test, clf_svm.decision_function(X_test)
)
classify.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

threshold_recall = thresholds[np.argmin(recalls >= 0.75)]

y_test_predict_score = clf_svm.decision_function(X_test)
y_test_predict_threshold = clf_svm.decision_function(X_test) >= threshold_recall
recall_score(y_test, y_test_predict_threshold)
precision_score(y_test, y_test_predict_threshold)
f1_score(y_test, y_test_predict_threshold)
cf_matrix = confusion_matrix(y_test, y_test_predict_threshold)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)

testing["classification_final_threshold"] = y_test_predict_threshold

# %%

coefs, names = zip(
    *sorted(
        zip(
            pipeline_svm.named_steps["clf"].coef_[0],
            pipeline_svm.named_steps["vect"].get_feature_names(),
        )
    )
)

feature_importance = {name: coef for name, coef in zip(names, coefs)}
feature_importance_df = pd.DataFrame(
    list(feature_importance.items()), columns=["term", "importance"]
)
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
)
feature_importance_df.head(20)
feature_importance_df.tail(20)

# %%
test = testing[
    [
        "unique_id",
        "tweet_id",
        "text",
        "relevant",
        "category",
        "covid",
        "character",
        "classification_final",
        "classification_final_threshold",
    ]
]

len(test[test.classification_final == 1])
len(
    test[test.classification_final_threshold == 1]
)  # would need to review 1/3 of tweets to capture 72% of relevant tweets
