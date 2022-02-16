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
        # ("tfidf", TfidfTransformer()),
        ("clf", svm.LinearSVC(random_state=593)),
    ]
)

clf = pipeline.fit(X_train, y_train)

# %%


y_test_predict = clf.predict(X_test)
classify.create_plot_confusion_matrix(
    cf_matrix=confusion_matrix(y_test, y_test_predict)
)
recall_score(y_test, y_test_predict)

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
            pipeline.named_steps["clf"].coef_[0],
            pipeline.named_steps["vect"].get_feature_names(),
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

feature_importance_df.to_csv(start.MAIN_DIR + "feature_importance.csv")

# %%

f1s = []
recalls = []
sizes = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2400, 2500, 2600, 2750, 3000]
for size in sizes:
    clf_size = pipeline.fit(
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
