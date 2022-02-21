# %%
import pandas as pd
import os
import spacy
from spacy.util import minibatch, compounding
import numpy as np
import re

MAIN_DIR = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/"

# %%
tweets = pd.read_csv(MAIN_DIR + "tweets_full.csv")
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
annotations = pd.read_csv(MAIN_DIR + "annotations.csv")
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

df["labels"] = np.where(df.relevant == 1, "POSITIVE", "NEGATIVE")

df["text"] = [re.sub(r"[^\w\s]", "", s) for s in df.text]  # remove punctuation
df["text"] = df.text.str.replace("\n", " ", regex=False)  # remove new line
df["text"] = df.text.str.replace("\xa0", " ", regex=False)  # remove utf errors
training = df[df.random_set != 3]
testing = df[df.random_set == 3]


# %%
model_dir = "/Users/kla21002/textcat_tweets/packages/en_textcat_demo-0.0.0/en_textcat_demo/en_textcat_demo-0.0.0"
# apply the saved model
print("Loading from", model_dir)
nlp = spacy.load(model_dir)
categories = []
for text in testing.text:
    doc = nlp(text)
    categories.append(doc.cats)

# %%
testing["classification"] = [label["POSITIVE"] for label in categories]
testing.to_csv(MAIN_DIR + "testing_spacy.csv")
# %%
