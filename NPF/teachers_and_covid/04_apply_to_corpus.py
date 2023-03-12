# %%
import pandas as pd
import numpy as np

from NPF.library import process_dictionary
from NPF.teachers_and_covid import start

# %% Import key words

key_words = pd.read_excel(start.RESULTS_DIR + "key_word_precision.xlsx")
key_words = key_words[key_words.word_precision > 0.5]
key_words = list(key_words.word)
len(key_words)
# %%
df = pd.read_csv(start.CLEAN_DIR + "tweets_full.csv")
df = df[
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

df["text_clean"] = df.text.apply(lambda x: x.lower())
df["text_clean"] = df.text_clean.apply(process_dictionary.remove_punctuation)
df["text_clean"] = df.text_clean.apply(process_dictionary.remove_non_unicode)
df["text_clean"] = df.text_clean.apply(process_dictionary.stem_string)

df['positive'] = df[key_words].max(axis=1)

# %%