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
    [relevance_annotations[relevance_annotations.relevant == 1], character_annotations]
)

df = df.merge(tweets, left_on=["unique_id"], right_on=["unique_id"])


# %%
df = df[df.category.isin([1, 2, 3, 4])]

df["hero"] = np.where(df.category == 1, 1, 0)
df["villain"] = np.where(df.category == 2, 1, 0)
df["victim"] = np.where(df.category == 3, 1, 0)
df["other"] = np.where(df.category == 4, 1, 0)


df["text"] = [re.sub(r"[^\w\s]", "", s) for s in df.text]  # remove punctuation
df["text"] = df.text.str.replace("\n", " ", regex=False)  # remove new line
df["text"] = df.text.str.replace("\xa0", " ", regex=False)  # remove utf errors

# %%
training = df[df.random_set != 3]
testing = df[df.random_set == 3]

# %%
train_cats = [
    {"HERO": hero, "VILLAIN": villain, "VICTIM": victim, "OTHER": other}
    for hero, villain, victim, other in zip(
        training.hero, training.villain, training.victim, training.other
    )
]


train_list = []
for text, cat in zip(training.text, train_cats):
    train_list.append({"text": text, "cats": cat})
train_list = [str(item) for item in train_list]

with open(
    "/Users/kla21002/textcat_tweets_characters/assets/json_training.jsonl", "w"
) as f:
    for line in train_list:
        f.write(line.replace("'", '"'))
        f.write("\n")
# %%
# test_cats = [
#     {"HERO": hero, "VILLAIN": villain, "VICTIM": victim, "OTHER": other}
#     for hero, villain, victim, other in zip(
#         testing.hero, testing.villain, testing.victim, testing.other
#     )
# ]


test_list = []
for text, cat in zip(testing.text, test_cats):
    test_list.append({"text": text, "cats": cat})

test_list = [str(item) for item in test_list]

with open(
    "/Users/kla21002/textcat_tweets_characters/assets/json_testing.jsonl", "w"
) as f:
    for line in test_list:
        f.write(line.replace("'", '"'))
        f.write("\n")

# %%
df[
    [
        "unique_id",
        "category",
        "relevant",
        "irrelevant",
        "hero",
        "villain",
        "victim",
        "other",
    ]
].to_csv(start.MAIN_DIR + "annotations_characters.csv")

# %%
