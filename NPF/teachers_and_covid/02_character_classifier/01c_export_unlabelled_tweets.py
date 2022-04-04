# %%

import pandas as pd
import numpy as np
import re
import pickle

from NPF.teachers_and_covid import start


# %%
df = pd.read_csv(start.CLEAN_DIR + "tweets_full.csv")

# %%
df["text"] = [re.sub(r"[^\w\s]", "", s) for s in df.text]  # remove punctuation
df["text"] = df.text.str.replace("\n", " ", regex=False)  # remove new line
df["text"] = df.text.str.replace("\xa0", " ", regex=False)  # remove utf errors

# %%
training = df[df.random_set == 5]

# %%
train_cats = [
    {"HERO": "", "VILLAIN": "", "VICTIM": "", "OTHER": ""}
    for hero, villain, victim, other in zip(
        training.text, training.text, training.text, training.text
    )
]


new_train_list = []
for unique_id, text, cat in zip(training.unique_id, training.text, train_cats):
    new_train_list.append(
        {
            "unique_id": unique_id,
            "text": text,
        }
    )
new_train_list = [str(item) for item in new_train_list]

with open(
    "/Users/kylieanglin/Dropbox/Active/NPF/data/batch5_characters.jsonl", "w"
) as f:
    for line in new_train_list:
        line = line.replace("'", '"')
        f.write(line)
        f.write("\n")


# %%
df = pd.read_csv(start.CLEAN_DIR + "annotations_characters.csv")

df["text"] = [re.sub(r"[^\w\s]", "", s) for s in df.text]  # remove punctuation
df["text"] = df.text.str.replace("\n", " ", regex=False)  # remove new line
df["text"] = df.text.str.replace("\xa0", " ", regex=False)  # remove utf errors

training = df[df.random_set != 3]
testing = df[df.random_set == 3]


# %%

training["label"] = np.where(
    training.hero == 1,
    "HERO",
    np.where(
        training.villain == 1,
        "VILLAIN",
        np.where(training.victim == 1, "VICTIM", "OTHER"),
    ),
)


train_list = []
for unique_id, text, label in zip(training.unique_id, training.text, training.label):
    train_list.append(
        {"unique_id": unique_id, "text": text, "label": label, "answer": "accept"}
    )
train_list = [str(item) for item in train_list]

with open(
    "/Users/kylieanglin/Dropbox/Active/NPF/data/character_training_for_prodigy.jsonl",
    "w",
) as f:
    for line in train_list:
        f.write(line.replace("'", '"'))
        f.write("\n")

    # for line in new_train_list:
    #   line = line.replace("'", '"')
    #   f.write(line)
    #   f.write("\n")


import pandas as pd

prodigy = pd.read_json(
    "/Users/kylieanglin/Dropbox/Active/NPF/data/prodigy_character_annotations.jsonl",
    lines=True,
)

prodigy = prodigy[prodigy.label.isin(["VICTIM", "OTHER", "HERO", "VILLAIN"])]
