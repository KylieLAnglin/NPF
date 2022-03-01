# %%

import pandas as pd
import numpy as np
import re
import pickle

from NPF.teachers_and_covid import start


# %%
df = pd.read_csv(start.CLEAN_DIR + "annotations_characters.csv")

# %%
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
test_cats = [
    {"HERO": hero, "VILLAIN": villain, "VICTIM": victim, "OTHER": other}
    for hero, villain, victim, other in zip(
        testing.hero, testing.villain, testing.victim, testing.other
    )
]


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

# %%
