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

annotations = pd.read_csv(CLEAN_DIR + "annotations_characters.csv")
annotations = annotations[
    [
        "unique_id",
        "category",
    ]
]

df = tweets.merge(annotations, how="left", left_on="unique_id", right_on="unique_id")


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
for text in training.text:
    doc = nlp(text)
    categories.append(doc.cats)

training["spacy_hero"] = [label["HERO"] for label in categories]
training["spacy_villain"] = [label["VILLAIN"] for label in categories]
training["spacy_victim"] = [label["VICTIM"] for label in categories]
training["spacy_other"] = [label["OTHER"] for label in categories]
training.to_csv(TEMP_DIR + "training_characters_spacy.csv")


# %%

# apply the saved model
print("Loading from", MODEL_DIR)
nlp = spacy.load(MODEL_DIR)
categories = []
for text in tweets.text:
    doc = nlp(text)
    categories.append(doc.cats)

tweets["spacy_hero"] = [label["HERO"] for label in categories]
tweets["spacy_villain"] = [label["VILLAIN"] for label in categories]
tweets["spacy_victim"] = [label["VICTIM"] for label in categories]
tweets["spacy_other"] = [label["OTHER"] for label in categories]

tweets[
    [
        "unique_id",
        "spacy_hero",
        "spacy_villain",
        "spacy_victim",
        "spacy_other",
    ]
].to_csv(TEMP_DIR + "model_characters_spacy.csv", index=False)
