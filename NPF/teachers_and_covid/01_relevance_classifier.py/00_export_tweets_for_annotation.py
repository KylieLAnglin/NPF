import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start

# %%

df = pd.read_csv(
    start.CLEAN_DIR + "tweets_full.csv",
    encoding="utf-8",
)

df_to_code = df[df.random_set == 1]
df_to_code = df_to_code[
    ["unique_id", "tweet_id", "random_set", "text", "relevant", "category", "notes"]
]
df_to_code.to_csv(
    start.CLEAN_DIR + "temp/training_batch1.csv",
    index=False,
)


df_to_code = df[df.random_set == 2]
df_to_code = df_to_code[
    ["unique_id", "tweet_id", "random_set", "text", "relevant", "category", "notes"]
]
df_to_code.to_csv(
    start.CLEAN_DIR + "temp/training_batch2.csv",
    index=False,
)

df_to_code = df[df.random_set == 3]
df_to_code = df_to_code[
    ["unique_id", "tweet_id", "random_set", "text", "relevant", "category", "notes"]
]
df_to_code.to_csv(
    start.CLEAN_DIR + "temp/training_batch3.csv",
    index=False,
)

df_to_code = df[df.random_set == 4]
df_to_code = df_to_code[
    ["unique_id", "tweet_id", "random_set", "text", "relevant", "category", "notes"]
]
df_to_code.to_csv(
    start.CLEAN_DIR + "training_batch4.csv",
    index=False,
)
