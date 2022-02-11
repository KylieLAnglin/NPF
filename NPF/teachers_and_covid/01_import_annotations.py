# %%
import re
import pandas as pd
import numpy as np


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

annotations = pd.concat([annotations1, annotations2, annotations3])

# %%
annotations = annotations[~annotations.relevant.isnull()]
# %%

print(len(annotations), " annotations")

annotations.relevant.value_counts()

print(len(annotations[annotations.relevant == 1]), " relevant")

annotations.category.value_counts()


# %%
annotations[annotations.random_set == 3]["Covid policy related"].value_counts()

# %%
annotations[annotations.random_set == 3]["Teacher character"].value_counts()
# %%
pd.crosstab(
    index=annotations["Covid policy related"], columns=annotations["Teacher character"]
)
# %%
annotations.to_csv(start.MAIN_DIR + "annotations.csv")

# %%
