# %%

import pandas as pd
import numpy as np
import re

from scipy import rand

from NPF.teachers_and_covid import start

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

character_annotations = pd.read_excel(
    start.ANNOTATIONS_DIR + "character_annotations_TF.xlsx"
)

# %%


df_merge = character_annotations[["unique_id", "character", "character_final"]].merge(
    tweets, left_on=["unique_id"], right_on=["unique_id"], how="left"
)

df_merge = df_merge.sort_values(by="random_set")

training = df_merge.head(2400)
training["split"] = "training"

remaining = df_merge.tail(600)
testing = remaining.head(300)
testing["split"] = "testing"

validation = remaining.tail(300)
validation["split"] = "validation"
# %%
df = pd.concat([training, testing, validation])
df.split.value_counts()

# %%
df.character.value_counts()

df["tweet_hero"] = np.where(
    (df.character == "Hero")
    | (df.character == "Hero and victim")
    | (df.character == "Hero and villain"),
    1,
    0,
)

df["tweet_victim"] = np.where(
    (df.character == "Victim")
    | (df.character == "Hero and victim")
    | (df.character == "Vicim and villain"),
    1,
    0,
)

df["tweet_villain"] = np.where(
    (df.character == "Villain")
    | (df.character == "Hero and villain")
    | (df.character == "Vicim and villain"),
    1,
    0,
)

df.to_csv(
    start.CLEAN_DIR + "annotations_characters.csv",
    encoding="utf-8",
)

# %%
