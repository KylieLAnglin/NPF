# %%
import pandas as pd
import os
import spacy
from spacy.util import minibatch, compounding
import numpy as np
import re

CLEAN_DIR = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/"
MODEL_DIR = "/Users/kla21002/textcat_tweets/packages/en_textcat_demo-0.0.0/en_textcat_demo/en_textcat_demo-0.0.0"
TEMP_DIR = "/Users/kla21002/Dropbox/Active/NPF/data/"
# %%
tweets = pd.read_csv(CLEAN_DIR + "tweets_full.csv")
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

annotations = pd.read_csv(CLEAN_DIR + "annotations.csv")
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

training = df[df.random_set != 3]
testing = df[df.random_set == 3]

# %%
# apply the saved model
print("Loading from", MODEL_DIR)
nlp = spacy.load(MODEL_DIR)
categories = []
for text in testing.text:
    doc = nlp(text)
    categories.append(doc.cats)

testing["classification"] = [label["POSITIVE"] for label in categories]
testing.to_csv(TEMP_DIR + "testing_spacy.csv")
# %%

# apply the saved model
print("Loading from", MODEL_DIR)
nlp = spacy.load(MODEL_DIR)
categories = []
for text in training.text:
    doc = nlp(text)
    categories.append(doc.cats)

training["classification"] = [label["POSITIVE"] for label in categories]
training.to_csv(TEMP_DIR + "training_spacy.csv")


# %%

# apply the saved model
print("Loading from", MODEL_DIR)
nlp = spacy.load(MODEL_DIR)
categories = []
for text in tweets.text:
    doc = nlp(text)
    categories.append(doc.cats)

tweets["score_spacy"] = [label["POSITIVE"] for label in categories]
tweets[["unique_id", "score_spacy"]].to_csv(TEMP_DIR + "model_spacy.csv", index=False)
