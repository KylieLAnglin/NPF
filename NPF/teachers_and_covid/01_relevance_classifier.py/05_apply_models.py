# %%
import re
import pandas as pd
import numpy as np
import pickle


from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    TfidfTransformer,
)
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    plot_roc_curve,
)

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
from NPF.teachers_and_covid import start
from NPF.library import classify

# %%
clf_nb = pickle.load(open(start.TEMP_DIR + "model_nb.sav", "rb"))
clf_rf = pickle.load(open(start.TEMP_DIR + "model_rf.sav", "rb"))
clf_ridge = pickle.load(open(start.TEMP_DIR + "model_ridge.sav", "rb"))
clf_svm = pickle.load(open(start.TEMP_DIR + "model_svm.sav", "rb"))
clf_sgd = pickle.load(open(start.TEMP_DIR + "model_sgd.sav", "rb"))

spacy_df = pd.read_csv(start.TEMP_DIR + "model_spacy.csv")
spacy_df["unique_id"] = pd.to_numeric(spacy_df.unique_id, errors="coerce")

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
tweets["text"] = [re.sub(r"[^\w\s]", "", s) for s in tweets.text]  # remove punctuation
tweets["text"] = tweets.text.str.replace("\n", " ", regex=False)  # remove new line
tweets["text"] = tweets.text.str.replace("\xa0", " ", regex=False)  # remove utf errors

tweets = tweets[tweets.unique_id != 186803]
tweets = tweets.merge(
    spacy_df[["unique_id", "score_spacy"]], left_on="unique_id", right_on="unique_id"
)
# %%

# %%
tweets["score_nb"] = [proba[1] for proba in clf_nb.predict_proba(tweets.text)]
tweets["score_svm"] = clf_svm.decision_function(tweets.text)
tweets["score_sgd"] = clf_sgd.decision_function(tweets.text)
tweets["score_rf"] = [proba[1] for proba in clf_rf.predict_proba(tweets.text)]
tweets["score_ridge"] = [proba[1] for proba in clf_ridge.predict_proba(tweets.text)]

# tweets["classification"] = np.where(
#     (tweets.score_spacy > 0.5)
#     | (tweets.score_ridge > 0.5)
#     | (tweets.score_svm > 0.5)
#     | (tweets.score_sgd > 0.5)
#     | (tweets.score_nb > 0.5)
#     | (tweets.score_rf > 0.5),
#     1,
#     0,
# )

tweets["classification"] = np.where(tweets.score_nb > 0.5, 1, 0)
# %%
tweets[
    [
        "unique_id",
        "score_nb",
        "score_cvm",
        "score_sgd",
        "score_rf",
        "score_ridge",
        "classification_rule",
    ]
].to_csv(start.TEMP_DIR + "tweets_classified.csv", index=False)
