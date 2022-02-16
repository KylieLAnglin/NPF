# %%
import pandas as pd
import numpy as np
from pprint import pprint
from time import time
import pickle


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

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
    ]
)

clf = pipeline.fit(training.text, training.relevant)

# %%


testing["classification"] = clf.predict(testing.text)
classify.create_plot_confusion_matrix(
    cf_matrix=confusion_matrix(testing.relevant, testing.classification)
)
recall_score(testing.relevant, testing.classification)

precisions, recalls, thresholds = precision_recall_curve(
    testing.relevant, clf.decision_function(testing.text)
)
classify.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

threshold_recall = thresholds[np.argmin(recalls >= 0.75)]

testing["classification_score"] = clf.decision_function(testing.text)
testing["classification_threshold"] = (
    clf.decision_function(testing.text) >= threshold_recall
)
recall_score(testing.relevant, testing.classification_threshold)

cf_matrix = confusion_matrix(testing.relevant, testing.classification_threshold)
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

feature_importance_df.to_csv(start.MAIN_DIR + "feature_importance_ridge.csv")

# %%

f1s = []
recalls = []
sizes = [
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
    2100,
    2200,
    2300,
    2400,
    2500,
    2600,
    2700,
    2800,
    2900,
    3000,
    3100,
]
for size in sizes:
    clf_size = pipeline.fit(
        training.text.sample(size, random_state=4),
        training.relevant.sample(size, random_state=4),
    )
    recall = recall_score(testing.relevant, clf_size.predict(testing.text))
    print(recall)
    f1 = f1_score(testing.relevant, clf_size.predict(testing.text))
    f1s.append(f1)
    recalls.append(recall)

plt.plot(sizes, recalls)
plt.show()
# %%
