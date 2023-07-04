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
places = pd.read_csv(start.RAW_DIR + "places.csv")
places = places.set_index("id")
places = places.rename(columns={"name": "place"})

df = df.merge(places, left_on="geo", right_index=True)
df["tweet_count"] = 1

df[["tweet_count", "author_id"]].groupby("author_id").sum().sort_values(
    by="tweet_count", ascending=False
)
df[["tweet_count", "author_id"]].groupby("author_id").sum().hist()
# %%
# %%

df = df.sample(frac=1, random_state=20).groupby("author_id").head(10)
df_places = (
    df[["place", "hero", "villain", "victim", "tweet_count"]].groupby("place").sum()
)
# %%

df_places["hero_p"] = round(df_places.hero / df_places.tweet_count, 2)
df_places["villain_p"] = round(df_places.villain / df_places.tweet_count, 2)

df_places["victim_p"] = round(df_places.victim / df_places.tweet_count, 2)
df_places = df_places.sort_values(by="tweet_count", ascending=False)
df_places[["hero_p", "villain_p", "victim_p", "tweet_count"]].head(10)
# %%
df[df.place == "Chicago"]
df[df.place == "Chicago"][df.hero == 1].to_csv(
    start.RESULTS_DIR + "landmark_locations/" + "chicago_heroes.csv"
)
df[df.place == "Chicago"][df.villain == 1].to_csv(
    start.RESULTS_DIR + "landmark_locations/" + "chicago_villains.csv"
)
df[df.place == "Chicago"][df.victim == 1].to_csv(
    start.RESULTS_DIR + "landmark_locations/" + "chicago_victims.csv"
)


df[df.place == "Houston"][df.hero == 1].to_csv(
    start.RESULTS_DIR + "landmark_locations/" + "houston_heroes.csv"
)
df[df.place == "Houston"][df.villain == 1].to_csv(
    start.RESULTS_DIR + "landmark_locations/" + "houston_villains.csv"
)
df[df.place == "Houston"][df.victim == 1].to_csv(
    start.RESULTS_DIR + "landmark_locations/" + "houston_victims.csv"
)


df[df.place == "Florida"][df.hero == 1].to_csv(
    start.RESULTS_DIR + "landmark_locations/" + "florida_heroes.csv"
)
df[df.place == "Florida"][df.villain == 1].to_csv(
    start.RESULTS_DIR + "landmark_locations/" + "florida_villains.csv"
)
df[df.place == "Florida"][df.victim == 1].to_csv(
    start.RESULTS_DIR + "landmark_locations/" + "florida_victims.csv"
)

# %%
