# %%
import pandas as pd
import numpy as np
from pprint import pprint
from time import time
import pickle


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

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
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", svm.LinearSVC(random_state=593)),
    ]
)

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
# parameters = {
#     "vect__max_df": (0.5, 0.75, 1.0),
#     "vect__min_df": (0.01, 0.05, 0.10, 1),
#     "vect__max_features": (None, 100, 500, 1000, 2000),
#     "vect__ngram_range": ((1, 1), (1, 2), (1, 3)),
#     "vect__stop_words": (None, classify.stop_words),
#     "vect__binary": (True, False),
#     "tfidf__use_idf": (True, False),
#     "tfidf__norm": ("l1", "l2"),
#     "clf__C": (1, 10, 100),
#     "clf__loss": ("hinge", "squared_hinge"),
#     "clf__dual": (True, False),
# }

# grid_search = classify.grid_search(pipeline, parameters, X_train, y_train)

# %%

# FILE_NAME = start.MAIN_DIR + "Params_SVM.txt"
# f = open(FILE_NAME, "w+")
# file_object = open(FILE_NAME, "a")
# file_object.write(str(grid_search.best_params_))


# with open(start.MAIN_DIR + "Params_SVM.pickle", "wb") as handle:
#     pickle.dump(grid_search.best_params_, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%

with open(start.MAIN_DIR + "Params_SVM.pickle", "rb") as handle:
    SVM_params = pickle.load(handle)

print(SVM_params)

pipeline_svm = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                max_df=SVM_params["vect__max_df"],
                min_df=SVM_params["vect__min_df"],
                max_features=SVM_params["vect__max_features"],
                ngram_range=SVM_params["vect__ngram_range"],
                stop_words=SVM_params["vect__stop_words"],
                binary=SVM_params["vect__binary"],
            ),
        ),
        (
            "tfidf",
            TfidfTransformer(
                norm=SVM_params["tfidf__norm"],
                use_idf=SVM_params["tfidf__use_idf"],
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
# pickle.dump(clf_svm, open(start.MAIN_DIR + "model_SVM.sav", "wb"))


# %%
# clf_svm = pickle.load(open(start.MAIN_DIR + "model_SVM.sav", "rb"))

y_test_predict = clf_svm.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_test_predict)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)
recall_score(y_test, y_test_predict)
f1_score(y_test, y_test_predict)

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

feature_importance_df.head(20)
feature_importance_df.tail(20)

feature_importance_limited = feature_importance_df.head(5000).append(
    feature_importance_df.tail(5000)
)

feature_importance_limited.sort_values(by="importance", ascending=False).to_csv(
    start.MAIN_DIR + "feature_importance.csv"
)

# %%


f1s = []
recalls = []
sizes = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
for size in sizes:
    clf_size = pipeline_svm.fit(
        X_train.sample(size, random_state=4), y_train.sample(size, random_state=4)
    )
    recall = recall_score(y_test, clf_size.predict(X_test))
    print(recall)
    f1 = f1_score(y_test, clf_size.predict(X_test))
    f1s.append(f1)
    recalls.append(recall)

plt.plot(sizes, recalls)
plt.show()
# %%
