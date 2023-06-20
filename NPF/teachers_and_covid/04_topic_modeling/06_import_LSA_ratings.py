# %%
import os

import pandas as pd
import numpy as np
import string
from NPF.teachers_and_covid import start

# %%
df = pd.read_excel(start.ANNOTATIONS_DIR + "word_coherence_ratings_lsa_TF.xlsx")
df = df.head(7)
df
# %%
df_long = df.T
df_long.columns = df_long.iloc[0]
df_long = df_long.drop(df_long.index[0])
df_long = df_long.rename(
    columns={"Coherence Rating": "coherence", "Specificity Rating": "specificity"}
)
df_long["score"] = df_long.coherence * df_long.specificity
df_long
# %%
df_long = df_long.reset_index()
df_long = df_long.rename(columns={"index": "source"})
df_long[["model", "topic"]] = df_long["source"].str.rsplit(pat="_", n=1, expand=True)
df_long
# %%
df_topics = df_long[
    [
        "model",
        "topic",
        "score",
        "coherence",
        "specificity",
    ]
]
df_topics
df_topics.sort_values(by=["model", "topic"])
df_topics = df_topics[df_topics.topic < "20"]
# %%
df_best_topics = df_topics[["model", "score"]].groupby("model").head(5)
df_models = df_best_topics[["model", "score"]].groupby("model").mean()
df_models = df_models.sort_values(by="score", ascending=False)
df_models
# %%
MODEL = "lsa_no_below_500_no_above_1.0"
df_topics[df_topics.model == MODEL].sort_values(by="score", ascending=False)

TOPIC = "17"

tweets = pd.read_csv(start.RESULTS_DIR + "LSA/" + MODEL + "/lsa_tweets.csv")
tweets = tweets.sort_values(by=TOPIC, ascending=False)
list(tweets.head(5).tweet_text)

# %%
words = pd.read_csv(start.RESULTS_DIR + "LSA/" + MODEL + "/lsa_words.csv")
words = words.sort_values(by=TOPIC, ascending=False)
words.head()
# %%
