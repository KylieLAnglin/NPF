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

# relevance_annotations = pd.read_csv(start.CLEAN_DIR + "annotations.csv")
# relevance_annotations = relevance_annotations[["unique_id", "relevant", "category"]]

character_annotations5 = pd.read_csv(
    start.ANNOTATIONS_DIR + "zz_archive/training_batch5_annotated.csv"
)[["unique_id", "category", "irrelevant"]]
character_annotations6 = pd.read_csv(
    start.ANNOTATIONS_DIR + "zz_archive/training_batch6_annotated.csv"
)[["unique_id", "category", "irrelevant"]]
character_annotations7 = pd.read_csv(
    start.ANNOTATIONS_DIR + "zz_archive/training_batch7_annotated.csv"
)[["unique_id", "category", "irrelevant"]]
character_annotations8 = pd.read_csv(
    start.ANNOTATIONS_DIR + "zz_archive/training_batch8_annotated.csv"
)[["unique_id", "category", "irrelevant"]]
character_annotations9 = pd.read_csv(
    start.ANNOTATIONS_DIR + "zz_archive/training_batch9_annotated.csv"
)[["unique_id", "category", "irrelevant"]]
character_annotations10 = pd.read_csv(
    start.ANNOTATIONS_DIR + "zz_archive/training_batch10_annotated.csv"
)[["unique_id", "category", "irrelevant"]]


character_annotations = pd.concat(
    [
        character_annotations5,
        character_annotations6,
        character_annotations7,
        character_annotations8,
        # character_annotations9,
        character_annotations10,
    ]
)

# df = pd.concat(
#     [
#         relevance_annotations,
#         character_annotations,
#     ]
# )

df = character_annotations.merge(
    tweets, left_on=["unique_id"], right_on=["unique_id"], how="left"
)
df = df[df.category.isin([1, 2, 3, 4])]

df["victim"] = np.where(df.category == 3, 1, 0)
df[["unique_id", "text", "victim"]]

training = df[df.random_set != 6]
testing = df[df.random_set == 6]

training[["unique_id", "text", "victim"]].to_csv(
    "/Users/kla21002/Dropbox/Active/Teaching/EPSY 5643 Text Analytics/Week 11 Classification Cont/"
    + "tweet_training.csv"
)
testing[["unique_id", "text", "victim"]].to_csv(
    "/Users/kla21002/Dropbox/Active/Teaching/EPSY 5643 Text Analytics/Week 11 Classification Cont/"
    + "tweet_testing.csv"
)
