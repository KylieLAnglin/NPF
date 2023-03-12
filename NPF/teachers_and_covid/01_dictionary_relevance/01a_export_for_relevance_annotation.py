import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start

# %%

df = pd.read_csv(
    start.CLEAN_DIR + "tweets_full.csv",
    encoding="utf-8",
)

df["relevant"] = ""
df["category"] = 0
df["notes"] = ""

# %%
list(df["text"].sample(10, random_state=10))

# %%
df_to_code = df[df.random_set == 7]
df_to_code = df_to_code[["unique_id", "tweet_id", "random_set", "text"]]
df_to_code.to_excel(
    start.MAIN_DIR + "temp/relevance_key_words.xlsx",
    index=False,
)

# %%
