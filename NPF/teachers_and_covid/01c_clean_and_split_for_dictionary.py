# %%
import string
import unicodedata

import pandas as pd
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


from NPF.teachers_and_covid import start


# %% Read
df = pd.read_excel(start.ANNOTATIONS_DIR + "relevance_key_words_edited.xlsx")
df = df.rename(columns={"Key words if relevant": "annotation"})

# %% Clean
df = df[~df.annotation.isnull()]
df["relevant"] = np.where(df.annotation == 0, 0, 1)
print(df.relevant.mean())# %%


# %% # %% Create validation set

np.random.seed(570)

df["random_order"] = np.random.randint(1, 10000, size=len(df))

df = df.sort_values(by=["random_order"])

validation = df.head(200)

df = df.merge(
    validation[["unique_id"]],
    how="left",
    left_on="unique_id",
    right_on="unique_id",
    indicator="_validation",
)


df["validation"] = np.where(df._validation == "both", 1, 0)
df["group"] = np.where(df._validation == "both", "validation", "")
# %% Create testing set
training_and_test = df[df.validation == 0]

testing = training_and_test.head(200)

df = df.merge(
    testing[["unique_id"]],
    how="left",
    left_on="unique_id",
    right_on="unique_id",
    indicator="_testing",
)

df["group"] = np.where(df._testing == "both", "testing", df.group)
df["group"] = np.where(df.group == "", "training", df.group)

df.group.value_counts()
df["testing"] = np.where(df.group == "testing", 1, 0)
df["training"] = np.where(df.group == "training", 1, 0)


df.to_csv(start.MAIN_DIR + "temp/creating_dictionary.csv")

# %%