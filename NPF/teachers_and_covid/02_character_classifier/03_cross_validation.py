# %%
import pandas as pd
import numpy as np
from NPF.library import classify
from NPF.teachers_and_covid import start
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# %% Imports
tweets = pd.read_csv(
    start.MAIN_DIR + "data/clean/tweets_relevant.csv", index_col="unique_id"
)
matrix = pd.read_csv(
    start.MAIN_DIR + "data/clean/matrix_annotations.csv", index_col="unique_id"
)
annotations = pd.read_csv(
    start.CLEAN_DIR + "annotations_characters.csv", index_col="unique_id"
)
features = pd.read_csv(
    start.MAIN_DIR + "data/clean/features_annotations.csv", index_col="unique_id"
)

# %%
df = annotations[["split", "text", "hero", "victim", "villain"]]
df = df.rename(
    columns={
        "split": "tweet_split",
        "text": "tweet_text",
        "hero": "tweet_hero",
        "villain": "tweet_villain",
        "victim": "tweet_victim",
    }
)

# %%
matrix = matrix.loc[:, matrix.sum() >= 20]

lsa = features[[col for col in features.columns if col.startswith("lsa")]]
# %%
train = df[df.tweet_split == "training"]
test = df[df.tweet_split == "testing"]
train_x = lsa[lsa.index.isin(train.index)]
test_x = lsa[lsa.index.isin(test.index)]

# %%
###
# Victim classifiers
###


# %% Logistic regression
lm = LogisticRegression()
lm.fit(train_x, train.tweet_victim)
lm_predictions = lm.predict(test_x)
print(classification_report(test.tweet_victim, lm_predictions))


# %% SVM
grid = {
    # "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf"],
}

svm_cv = GridSearchCV(estimator=SVC(), param_grid=grid, cv=5, verbose=3)
svm_cv.fit(train_x, train.tweet_victim)
print(svm_cv.best_params_)
svm_predictions = svm_cv.predict(test_x)
print(classification_report(test.tweet_victim, svm_predictions))


# %% Random forest
grid = {
    "n_estimators": [10, 100, 1000],
    "max_depth": [5, 10, 50],
    "max_features": [None, "sqrt"],
    "random_state": [92],
}

rf_cv = GridSearchCV(
    estimator=RandomForestClassifier(), param_grid=grid, cv=5, verbose=3
)
rf_cv.fit(train_x, train.tweet_victim)
print(rf_cv.best_params_)
rf_predictions = rf_cv.predict(test_x)
print(classification_report(test.tweet_victim, rf_predictions))


# %%
