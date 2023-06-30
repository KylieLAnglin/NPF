# %%
import os

import pandas as pd
import numpy as np
import string
from sentistrength import PySentiStr
from NPF.teachers_and_covid import start

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr


# %%
df = pd.read_csv(start.CLEAN_DIR + "tweets_relevant_labeled.csv", index_col="unique_id")
df = df.drop(["Unnamed: 0"], axis=1)
df["hero"] = np.where(df.character_classification == "Hero", 1, 0)
df["victim"] = np.where(df.character_classification == "Victim", 1, 0)
df["villain"] = np.where(df.character_classification == "Villain", 1, 0)
# %%

MODEL = "topic_20_no_below_100_no_above_1"
topic_rankings = pd.read_csv(
    start.RESULTS_DIR + "topic_models/" + MODEL + "/tweet_topics.csv"
)
topic_rankings = topic_rankings.rename(
    columns={
        "0": "topic1",
        "1": "topic2",
        "2": "topic3",
        "3": "topic4",
        "4": "topic5",
        "5": "topic6",
        "6": "topic7",
        "7": "topic8",
        "8": "topic9",
        "9": "topic10",
        "10": "topic11",
        "11": "topic12",
        "12": "topic13",
        "13": "topic14",
        "14": "topic15",
        "15": "topic16",
        "16": "topic17",
        "17": "topic18",
        "18": "topic19",
        "19": "topic20",
    }
)
topic_rankings = topic_rankings.set_index("unique_id")
topic_rankings = topic_rankings[[col for col in topic_rankings if "topic" in col]]
df = df.merge(topic_rankings, how="left", left_index=True, right_index=True)
# %%
liwc = pd.read_csv(start.CLEAN_DIR + "liwc.csv")
liwc["unique_id"] = liwc.unique_id.astype(int)
liwc = liwc.set_index("unique_id")


# %%
liwc = liwc[["liwc_tone_pos", "liwc_tone_neg"]]
df = df.merge(liwc, left_index=True, right_index=True)

# %%
mod = smf.ols(formula="topic3 ~ liwc_tone_pos", data=df)
res = mod.fit()
print(res.summary())
# %%

print(pearsonr(df.topic3, df.liwc_tone_pos))
print(pearsonr(df.topic3, df.liwc_tone_neg))
print(pearsonr(df.topic3, df.hero))
print(pearsonr(df.topic3, df.victim))
print(pearsonr(df.topic3, df.villain))

# %%
print(pearsonr(df.topic6, df.liwc_tone_pos))
print(pearsonr(df.topic6, df.liwc_tone_neg))
print(pearsonr(df.topic6, df.hero))
print(pearsonr(df.topic6, df.victim))
print(pearsonr(df.topic6, df.villain))

# %%
print(pearsonr(df.topic16, df.liwc_tone_pos))
print(pearsonr(df.topic16, df.liwc_tone_neg))
print(pearsonr(df.topic16, df.hero))
print(pearsonr(df.topic16, df.victim))
print(pearsonr(df.topic16, df.villain))

# %%
print(pearsonr(df.topic7, df.liwc_tone_pos))
print(pearsonr(df.topic7, df.liwc_tone_neg))
print(pearsonr(df.topic7, df.hero))
print(pearsonr(df.topic7, df.victim))
print(pearsonr(df.topic7, df.villain))

# %%
print(pearsonr(df.topic13, df.liwc_tone_pos))
print(pearsonr(df.topic13, df.liwc_tone_neg))
print(pearsonr(df.topic13, df.hero))
print(pearsonr(df.topic13, df.victim))
print(pearsonr(df.topic13, df.villain))

# %%
print(pearsonr(df.topic19, df.liwc_tone_pos))
print(pearsonr(df.topic19, df.liwc_tone_neg))
print(pearsonr(df.topic19, df.hero))
print(pearsonr(df.topic19, df.victim))
print(pearsonr(df.topic19, df.villain))

# %%
print(pearsonr(df.topic10, df.liwc_tone_pos))
print(pearsonr(df.topic10, df.liwc_tone_neg))
print(pearsonr(df.topic10, df.hero))
print(pearsonr(df.topic10, df.victim))
print(pearsonr(df.topic10, df.villain))

# %%
print(pearsonr(df.topic12, df.liwc_tone_pos))
print(pearsonr(df.topic12, df.liwc_tone_neg))
print(pearsonr(df.topic12, df.hero))
print(pearsonr(df.topic12, df.victim))
print(pearsonr(df.topic12, df.villain))

# %%
print(pearsonr(df.topic11, df.liwc_tone_pos))
print(pearsonr(df.topic11, df.liwc_tone_neg))
print(pearsonr(df.topic11, df.hero))
print(pearsonr(df.topic11, df.victim))
print(pearsonr(df.topic11, df.villain))

# %%
