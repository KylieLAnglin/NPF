# %%
import re
from typing import Dict
import pandas as pd
import numpy as np
import json


from NPF.teachers_and_covid import start
from NPF.library import classify


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
# %%
annotations1 = pd.read_csv(
    start.MAIN_DIR + "annotations/training_batch1_annotated.csv", encoding="utf-8"
)
annotations2 = pd.read_csv(
    start.MAIN_DIR + "annotations/training_batch2_annotated.csv", encoding="utf-8"
)

annotations3 = pd.read_csv(
    start.MAIN_DIR + "annotations/training_batch3_annotated.csv", encoding="utf-8"
)

annotations4 = pd.read_csv(
    start.MAIN_DIR + "annotations/training_batch4_annotated.csv", encoding="utf-8"
)


annotations = pd.concat([annotations1, annotations2, annotations3, annotations4])
# %%
annotations = annotations[~annotations.relevant.isnull()]
# %%

print(len(annotations), " annotations")

annotations.relevant.value_counts()

print(len(annotations[annotations.relevant == 1]), " relevant")

annotations.category.value_counts()


# %%
annotations[annotations.random_set == 3]["covid"].value_counts()

annotations[annotations.random_set == 3]["character"].value_counts()

pd.crosstab(index=annotations["covid"], columns=annotations["character"])

annotations.to_csv(start.MAIN_DIR + "annotations.csv")

# %%
annotations_autoai = annotations[["random_set", "text", "relevant"]]
annotations_autoai["labels"] = np.where(
    annotations.relevant == 1, "POSITIVE", "NEGATIVE"
)

# %%
annotations_autoai_training = annotations_autoai[annotations_autoai.random_set != 3]
annotations_autoai_training[["text", "labels"]].to_csv(
    start.MAIN_DIR + "annotations_autoai_training.csv", index=False
)

annotations_autoai_testing = annotations_autoai[annotations_autoai.random_set == 3]
annotations_autoai_testing[["text", "labels"]].to_csv(
    start.MAIN_DIR + "annotations_autoai_testing.csv", index=False
)
# %%

training_json = annotations_autoai_training[["text", "labels"]].to_json(
    orient="records", lines=True
)
with open(start.MAIN_DIR + "json_training.jsonl", "w") as outfile:
    outfile.write(training_json)


testing_json = annotations_autoai_testing[["text", "labels"]].to_json(
    orient="records", lines=True
)
with open(start.MAIN_DIR + "json_testing.jsonl", "w") as outfile:
    outfile.write(testing_json)

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
annotations = pd.read_csv(start.MAIN_DIR + "annotations.csv")
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
df["text"] = [re.sub(r"[^\w\s]", "", s) for s in df.text]
df["text"] = df.text.str.replace("\n", " ", regex=False)
df["text"] = df.text.str.replace("\xa0", " ", regex=False)

training = df[df.random_set != 3]
testing = df[df.random_set == 3]


# %%


train_cats = [{"POSITIVE": cat, "NEGATIVE": 1 - cat} for cat in training.relevant]

train_list = []
for text, cat in zip(training.text, train_cats):
    train_list.append({"text": text, "cats": cat})

train_list = [str(item) for item in train_list]

with open("/Users/kla21002/textcat_tweets/assets/training.jsonl", "w") as f:
    for line in train_list:
        f.write(line)
        f.write("\n")
# %%
test_cats = [{"POSITIVE": cat, "NEGATIVE": 1 - cat} for cat in testing.relevant]

test_list = []
for text, cat in zip(testing.text, test_cats):
    test_list.append({"text": text, "cats": cat})

test_list = [str(item) for item in test_list]

with open("/Users/kla21002/textcat_tweets/assets/testing.jsonl", "w") as f:
    for line in test_list:
        f.write(line)
        f.write("\n")

# %%
