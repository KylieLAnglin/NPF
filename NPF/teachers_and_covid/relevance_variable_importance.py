# %%

from pprint import pprint
from time import time
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

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
dev = df.head(150)
train = df.tail(len(df) - 150)

# %%


X_train = train.text
X_test = dev.text
y_train = train.relevant
y_test = dev.relevant
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
parameters = {
    "vect__max_df": (0.5, 0.75, 1.0),
    "vect__min_df": (0.01, 0.05, 0.10, 1),
    "vect__max_features": (None, 100, 500, 1000, 2000),
    "vect__ngram_range": ((1, 1), (1, 2), (1, 3)),
    "vect__stop_words": (None, "english"),
    "vect__binary": (True, False),
    "tfidf__use_idf": (True, False),
    "tfidf__norm": ("l1", "l2"),
    "clf__C": (1, 10, 100),
    "clf__loss": ("hinge", "squared_hinge"),
    "clf__dual": (True, False),
}

grid_search = classify.grid_search(pipeline, parameters, X_train, y_train)

# %%
with open(start.MAIN_DIR + "SVM_Params.pickle", "wb") as handle:
    pickle.dump(grid_search.best_params_, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%

with open(start.MAIN_DIR + "SVM_Params.pickle", "rb") as handle:
    SVM_params = pickle.load(handle)

print(SVM_params)

# %%
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
pickle.dump(clf_svm, open(start.MAIN_DIR + "model_SVM.sav", "wb"))

#

# %%
clf_svm = pickle.load(open(start.MAIN_DIR + "model_SVM.sav", "rb"))

y_test_predict = clf_svm.predict(X_test)
y_test_predict_score = clf_svm.decision_function(X_test)

cf_matrix = confusion_matrix(y_test, y_test_predict)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)
f1_score(y_test, y_test_predict)
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

feature_importance_df.to_csv(start.MAIN_DIR + "relevance_feature_importance.csv", index=False)

# %%
dev["score"] = y_test_predict_score
dev["predict"] = y_test_predict

# high confidence true positives
tp = dev[(dev.predict == 1) & (dev.relevant == 1)].sort_values(
    by="score", ascending=False
)
list(tp.text)

# high confidence false negatives
fn = dev[(dev.predict == 0) & (dev.relevant == 1)].sort_values(
    by="score", ascending=True
)
list(fn.head().text)

# high confidence false positives
fp = dev[(dev.predict == 1) & (dev.relevant == 1)].sort_values(
    by="score", ascending=True
)