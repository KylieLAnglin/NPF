# %%
import re

import pandas as pd
import numpy as np


from NPF.teachers_and_covid import start
from NPF.library import process_text


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


# df["text"] = [
#     process_text.process_text_nltk(
#         tweet, lower_case=True, remove_punct=True, remove_stopwords=True, lemma=True
#     )
#     for tweet in df.text
# ]
# %%
df["text"] = [re.sub(r"[^\w\s]", "", s) for s in df.text]  # remove punctuation
df["text"] = df.text.str.replace("\n", " ", regex=False)  # remove new line
df["text"] = df.text.str.replace("\xa0", " ", regex=False)  # remove utf errors


# %%
training = df[df.random_set != 3]
testing = df[df.random_set == 3]

# %%
train_cats = [{"POSITIVE": cat, "NEGATIVE": 1 - cat} for cat in training.relevant]

train_list = []
for text, cat in zip(training.text, train_cats):
    train_list.append({"text": text, "cats": cat})

train_list = [str(item) for item in train_list]

with open("/Users/kla21002/textcat_tweets/assets/json_training.jsonl", "w") as f:
    for line in train_list:
        f.write(line.replace("'", '"'))
        f.write("\n")
# %%
test_cats = [{"POSITIVE": cat, "NEGATIVE": 1 - cat} for cat in testing.relevant]

test_list = []
for text, cat in zip(testing.text, test_cats):
    test_list.append({"text": text, "cats": cat})

test_list = [str(item) for item in test_list]

with open("/Users/kla21002/textcat_tweets/assets/json_testing.jsonl", "w") as f:
    for line in test_list:
        f.write(line.replace("'", '"'))
        f.write("\n")

# %%
