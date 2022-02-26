# %%

import pandas as pd
import numpy as np
import re

from NPF.teachers_and_covid import start


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

relevance_annotations = pd.read_csv(start.MAIN_DIR + "annotations.csv")
relevance_annotations = relevance_annotations[["unique_id", "relevant", "category"]]

character_annotations5 = pd.read_csv(
    start.MAIN_DIR + "annotations/training_batch5_annotated.csv"
)[["unique_id", "category", "irrelevant"]]
character_annotations6 = pd.read_csv(
    start.MAIN_DIR + "annotations/training_batch6_annotated.csv"
)[["unique_id", "category", "irrelevant"]]
character_annotations7 = pd.read_csv(
    start.MAIN_DIR + "annotations/training_batch7_annotated.csv"
)[["unique_id", "category", "irrelevant"]]

character_annotations = pd.concat(
    [
        character_annotations5,
        character_annotations6,
        character_annotations7,
    ]
)

df = pd.concat(
    [
        relevance_annotations,
        character_annotations,
    ]
)

df = df.merge(tweets, left_on=["unique_id"], right_on=["unique_id"])


# %%
df = df[df.category.isin([1, 2, 3, 4])]
df = df[(df.relevant != 0) | (df.category == 1)]
df = df[df.irrelevant != 1]

# %%

df["hero"] = np.where(df.category == 1, 1, 0)
df["villain"] = np.where(df.category == 2, 1, 0)
df["victim"] = np.where(df.category == 3, 1, 0)
df["other"] = np.where(df.category == 4, 1, 0)

df.to_csv(
    start.MAIN_DIR + "annotations_characters.csv",
    encoding="utf-8",
)
# %%
