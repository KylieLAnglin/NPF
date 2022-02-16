import re
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
from pprint import pprint
from time import time
import pickle


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


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
testing = df.head(300)
training = df.tail(len(df) - 150)


# %%
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", RandomForestClassifier(random_state=593)),
    ]
)

parameters = {
    "vect__max_df": (0.5, 1.0),
    "vect__min_df": (0.05, 1),
    "vect__max_features": (None, 1000),
    "vect__ngram_range": ((1, 1), (1, 2), (1, 3)),
    "vect__stop_words": (None, classify.stop_words),
    "vect__binary": (True, False),
    "tfidf__use_idf": (True, False),
    "tfidf__norm": ("l1", "l2"),
}

# %%

grid_search = classify.grid_search(
    pipeline, parameters, training.text, training.relevant
)
print(grid_search.best_params_)

# %%
FILE_NAME = start.MAIN_DIR + "Params Random Forest.txt"
f = open(FILE_NAME, "w+")
file_object = open(FILE_NAME, "a")
file_object.write(str(grid_search.best_params_))


with open(start.MAIN_DIR + "Params_RF.pickle", "wb") as handle:
    pickle.dump(grid_search.best_params_, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(start.MAIN_DIR + "Params_RF.pickle", "rb") as handle:
    params = pickle.load(handle)

# %%
pipeline = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                max_df=params["vect__max_df"],
                min_df=params["vect__min_df"],
                max_features=params["vect__max_features"],
                ngram_range=params["vect__ngram_range"],
                stop_words=params["vect__stop_words"],
                binary=params["vect__binary"],
            ),
        ),
        (
            "tfidf",
            TfidfTransformer(
                norm=params["tfidf__norm"],
                use_idf=params["tfidf__use_idf"],
            ),
        ),
        (
            "clf",
            RandomForestClassifier(
                random_state=593,
            ),
        ),
    ]
)

clf_best = pipeline.fit(training.text, training.relevant)

# %%
y_test_predict = clf_best.predict(testing.text)
cf_matrix = confusion_matrix(testing.relevant, y_test_predict)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)
recall_score(testing.relevant, y_test_predict)
f1_score(testing.relevant, y_test_predict)


coefs, names = zip(
    *sorted(
        zip(
            pipeline.named_steps["clf"].feature_importances_,
            pipeline.named_steps["vect"].get_feature_names(),
        )
    )
)
feature_importance = {name: coef for name, coef in zip(names, coefs)}
feature_importance_df = pd.DataFrame(
    list(feature_importance.items()), columns=["term", "importance"]
)


feature_importance_df.head(20)
feature_importance_df.tail(20)
