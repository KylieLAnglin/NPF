# %%
import os

import pandas as pd
import numpy as np
import string
from NPF.teachers_and_covid import start

# %%
df_tf = pd.read_excel(start.ANNOTATIONS_DIR + "word_coherence_ratings_TF.xlsx")
df_tf = df_tf.head(7)
df_tf
# %%
df_zm = pd.read_excel(start.ANNOTATIONS_DIR + "word_coherence_ratings_ZM.xlsx")
df_zm = df_zm.head(7)
df_zm

# %%
df_tf_long = df_tf.T
df_tf_long.columns = df_tf_long.iloc[0]
df_tf_long = df_tf_long.drop(df_tf_long.index[0])
df_tf_long = df_tf_long.rename(
    columns={"Coherence Rating": "coherence_tf", "Specificity Rating": "specificity_tf"}
)
df_tf_long["score_tf"] = df_tf_long.coherence_tf * df_tf_long.specificity_tf
df_tf_long
# %%
df_zm_long = df_zm.T
df_zm_long.columns = df_zm_long.iloc[0]
df_zm_long = df_zm_long.drop(df_zm_long.index[0])
df_zm_long = df_zm_long.rename(
    columns={"Coherence Rating": "coherence_zm", "Specificity Rating": "specificity_zm"}
)
df_zm_long["score_zm"] = df_zm_long.coherence_zm * df_zm_long.specificity_zm

df_zm_long

# %%
df_long = df_tf_long.merge(
    df_zm_long[["coherence_zm", "specificity_zm", "score_zm"]],
    left_index=True,
    right_index=True,
)
df_long = df_long.sort_index()
df_long["coherence_tf"] = df_long.coherence_tf.astype(int)
df_long["coherence_zm"] = df_long.coherence_zm.astype(int)
df_long["score_tf"] = df_long.score_tf.astype(int)
df_long["score_zm"] = df_long.score_zm.astype(int)

df_long
# %%
df_long["coherence_tf"].corr(df_long["coherence_zm"])
df_long["score_tf"].corr(df_long["score_zm"])

# %%
df_long["coherence"] = (df_long.coherence_zm + df_long.coherence_tf) / 2
df_long["specificity"] = (df_long.specificity_zm + df_long.specificity_tf) / 2
df_long["score"] = (df_long.score_zm + df_long.score_tf) / 2

df_long.sort_values(by="coherence").head(5)
# %%
df_long = df_long.reset_index()
df_long = df_long.rename(columns={"index": "source"})
df_long[["model", "topic"]] = df_long["source"].str.split("_Word_", 1, expand=True)
df_long
# %%
df_topics = df_long[
    [
        "model",
        "topic",
        "score",
        "coherence",
        "specificity",
        "score_tf",
        "coherence_tf",
        "specificity_tf",
        "score_zm",
        "coherence_zm",
        "specificity_zm",
    ]
]
df_topics
df_topics.sort_values(by=["model", "topic"])
df_topics.to_csv(start.CLEAN_DIR + "model_topic_ratings.csv")
# %%
METRIC = "score"
# METRIC = "coherence"

df_best_topics = df_topics[["model", METRIC]].groupby("model").head(5)
df_models = df_best_topics[["model", METRIC]].groupby("model").mean()
df_models = df_models.sort_values(by=METRIC, ascending=False)
df_models
# %%
MODEL = "topic_10_no_below_50_no_above_1"
TOPIC = "7"
df_topics[df_topics.model == MODEL].sort_values(by="score", ascending=False)

tweets = pd.read_csv(start.RESULTS_DIR + "topic_models/" + MODEL + "/tweet_topics.csv")
tweets = tweets.sort_values(by=TOPIC, ascending=False)
list(tweets.head(10).text)

words = pd.read_csv(start.RESULTS_DIR + "topic_models/" + MODEL + "/topics.csv")
# %%
# MODEL = "topic_20_no_below_100_no_above_1"
# df_topics[df_topics.model == MODEL].sort_values(by="score", ascending=False)

# TOPIC = "15"

# tweets = pd.read_csv(start.RESULTS_DIR + "topic_models/" + MODEL + "/tweet_topics.csv")
# tweets = tweets.sort_values(by=TOPIC, ascending=False)
# list(tweets.head(5).text)
# %%
words = pd.read_csv(start.RESULTS_DIR + "topic_models/" + MODEL + "/topics.csv")
