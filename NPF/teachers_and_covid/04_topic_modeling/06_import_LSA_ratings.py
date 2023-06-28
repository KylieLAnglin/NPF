# %%
import os

import pandas as pd
import numpy as np
import string
from NPF.teachers_and_covid import start

# %%
df_tf = pd.read_excel(start.ANNOTATIONS_DIR + "word_coherence_ratings_lsa_TF.xlsx")
df_tf = df_tf.head(7)
df_tf
# %%
df_zm = pd.read_excel(start.ANNOTATIONS_DIR + "word_coherence_ratings_lsa_ZM.xlsx")
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
df_long
# %%
df_long = df_long.reset_index()
df_long = df_long.rename(columns={"index": "source"})
df_long[["model", "topic"]] = df_long["source"].str.rsplit(pat="_", n=1, expand=True)
df_long
# %%
df_long["coherence"] = (df_long.coherence_zm + df_long.coherence_tf) / 2
df_long["specificity"] = (df_long.specificity_zm + df_long.specificity_tf) / 2
df_long["score"] = (df_long.score_zm + df_long.score_tf) / 2
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

# %%
df_best_topics_score = df_topics[["model", "score"]].groupby("model").head(5)
df_models_score = df_best_topics_score[["model", "score"]].groupby("model").mean()

df_best_topics_coherence = df_topics[["model", "coherence"]].groupby("model").head(5)
df_models_coherence = (
    df_best_topics_coherence[["model", "coherence"]].groupby("model").mean()
)

df_best_topics_specificity = (
    df_topics[["model", "specificity"]].groupby("model").head(5)
)
df_models_specificity = (
    df_best_topics_specificity[["model", "specificity"]].groupby("model").mean()
)

df_models = df_models_coherence.merge(
    df_models_specificity, left_index=True, right_index=True
)
df_models = df_models.merge(df_models_score, left_index=True, right_index=True)
df_models

# %%

MODEL = "lsa_no_below_10_no_above_1.0"  # top score, top coherence

# %%
words = pd.read_csv(start.RESULTS_DIR + "LSA/" + MODEL + "/lsa_words.csv")
words

words.to_excel(start.ANNOTATIONS_DIR + "topic_interpretation/words_" + MODEL + ".xlsx")

# %%
tweets = pd.read_csv(start.RESULTS_DIR + "LSA/" + MODEL + "/lsa_tweets.csv")
tweets = tweets.drop_duplicates(subset="tweet_text", keep="first")


topics = [col for col in words.columns if "Word" in col]

top_tweets_lists = []
for topic in topics:
    topic = topic.replace("Word_", "")
    tweets = tweets.sort_values(by=topic, ascending=False)
    top_tweets_lists.append(list(tweets.head(250).tweet_text))

tweets_df = pd.DataFrame(top_tweets_lists).T
tweets_df.to_excel(
    start.ANNOTATIONS_DIR + "topic_interpretation/tweets_" + MODEL + ".xlsx"
)

# %%
