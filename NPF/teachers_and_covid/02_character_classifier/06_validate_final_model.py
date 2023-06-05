# %%
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from NPF.teachers_and_covid import start
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import cohen_kappa_score

# %%
annotations = pd.read_csv(
    start.CLEAN_DIR + "annotations_characters.csv", index_col="unique_id"
)
annotations["label"] = annotations.character_final.map(
    {"Other/None": 0, "Hero": 1, "Victim": 2, "Villain": 3}
)

validation_df = annotations[annotations.split == "validation"]
validation_df = validation_df.rename(columns = {"label": "human_label"})

all_tweets = pd.read_csv(start.CLEAN_DIR + "tweets_relevant_labeled.csv", index_col="unique_id")
all_tweets = all_tweets.rename(columns = {"label": "classification_label"})

df = all_tweets.merge(validation_df[[ "human_label"]], how="right", left_index=True, right_index=True)
len(df)
# %%

print(classification_report(df.human_label, df.classification_label))
cohen = cohen_kappa_score(df.human_label, df.classification_label)
print(cohen)
# %%