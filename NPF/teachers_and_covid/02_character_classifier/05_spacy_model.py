# %%
import pandas as pd
import os
import spacy
from spacy.util import minibatch, compounding
import numpy as np
import re

CLEAN_DIR = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/"
MODEL_DIR = "/Users/kla21002/textcat_tweets_characters/packages/en_textcat_main-0.0.0/en_textcat_demo/en_textcat_demo-0.0.0"
TEMP_DIR = "/Users/kla21002/Dropbox/Active/NPF/data/"
# %%
annotations = pd.read_csv(CLEAN_DIR + "annotations_characters.csv")[
    ["unique_id", "category"]
]
tweets_full = pd.read_csv(CLEAN_DIR + "tweets_full.csv")
tweets = annotations.merge(tweets_full, left_on="unique_id", right_on="unique_id")


# %%

tweets["tweet_text"] = [
    re.sub(r"[^\w\s]", "", s) for s in tweets.text
]  # remove punctuation
tweets["tweet_text"] = tweets.tweet_text.str.replace(
    "\n", " ", regex=False
)  # remove new line
tweets["tweet_text"] = tweets.tweet_text.str.replace(
    "\xa0", " ", regex=False
)  # remove utf errors


# %%
training = tweets[tweets.random_set != 3]
testing = tweets[tweets.random_set == 3]

# %%
# apply the saved model
print("Loading from", MODEL_DIR)
nlp = spacy.load(MODEL_DIR)
categories = []
for text in testing.tweet_text:
    doc = nlp(text)
    categories.append(doc.cats)

testing["spacy_hero"] = [label["HERO"] for label in categories]
testing["spacy_villain"] = [label["VILLAIN"] for label in categories]
testing["spacy_victim"] = [label["VICTIM"] for label in categories]
testing["spacy_other"] = [label["OTHER"] for label in categories]

testing.to_csv(TEMP_DIR + "testing_characters_spacy.csv")


# %%

# apply the saved model
print("Loading from", MODEL_DIR)
nlp = spacy.load(MODEL_DIR)
categories = []
for text in training.tweet_text:
    doc = nlp(text)
    categories.append(doc.cats)

training["spacy_hero"] = [label["HERO"] for label in categories]
training["spacy_villain"] = [label["VILLAIN"] for label in categories]
training["spacy_victim"] = [label["VICTIM"] for label in categories]
training["spacy_other"] = [label["OTHER"] for label in categories]
training.to_csv(TEMP_DIR + "training_characters_spacy.csv")


# %%
tweets_relevant = pd.read_csv(TEMP_DIR + "tweets_relevant.csv")

# apply the model to full dataset
tweets_relevant["tweet_text"] = [
    re.sub(r"[^\w\s]", "", s) for s in tweets_relevant.tweet_text
]  # remove punctuation
tweets_relevant["tweet_text"] = tweets_relevant.tweet_text.str.replace(
    "\n", " ", regex=False
)  # remove new line
tweets_relevant["tweet_text"] = tweets_relevant.tweet_text.str.replace(
    "\xa0", " ", regex=False
)  # remove utf errors


print("Loading from", MODEL_DIR)
nlp = spacy.load(MODEL_DIR)
categories = []
for text in tweets_relevant.tweet_text:
    doc = nlp(text)
    categories.append(doc.cats)

tweets_relevant["spacy_hero"] = [label["HERO"] for label in categories]
tweets_relevant["spacy_villain"] = [label["VILLAIN"] for label in categories]
tweets_relevant["spacy_victim"] = [label["VICTIM"] for label in categories]
tweets_relevant["spacy_other"] = [label["OTHER"] for label in categories]

# %%
tweets_relevant.to_csv(CLEAN_DIR + "tweets_final.csv", index=False)
