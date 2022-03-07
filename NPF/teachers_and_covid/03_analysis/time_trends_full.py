# %%
import pandas as pd
import numpy as np

from NPF.teachers_and_covid import start
import matplotlib.pyplot as plt

classifications = pd.read_csv(start.TEMP_DIR + "model_characters_spacy.csv")

tweets = pd.read_csv(start.CLEAN_DIR + "tweets_full.csv")

df = tweets.merge(
    classifications,
    left_on="unique_id",
    right_on="unique_id",
    how="left",
    indicator=True,
)

df.to_csv(start.CLEAN_DIR + "tweets_final.csv")
# %%
df["hero"] = np.where(df.spacy_hero > 0.5, 1, 0)
df["villain"] = np.where(df.spacy_villain > 0.5, 1, 0)
df["victim"] = np.where(df.spacy_victim > 0.5, 1, 0)

df["date"] = pd.to_datetime(df.created, errors="coerce")
df["month"] = df["date"].dt.month

df = df.set_index("date")


# %%
monthly = df[["hero", "villain", "victim"]].resample("M").sum()

# %%
plt.plot(monthly.index, monthly.hero, label="Hero")
plt.plot(monthly.index, monthly.villain, label="Villain")
plt.plot(monthly.index, monthly.victim, label="Victim")
plt.legend()
# %%

# %%
daily = df[["hero", "villain", "victim"]].resample("D").sum()

# plt.plot(daily.index, daily.hero, label="Hero")
# plt.plot(daily.index, daily.villain, label="Villain", color="orange")
plt.plot(daily.index, daily.victim, label="Victim", color="green")
plt.xticks(
    [
        "2020-03-01 00:00:00+00:00",
        "2020-04-01 00:00:00+00:00",
        "2020-05-01 00:00:00+00:00",
        "2020-06-01 00:00:00+00:00",
        "2020-07-01 00:00:00+00:00",
        "2020-08-01 00:00:00+00:00",
        "2020-09-01 00:00:00+00:00",
        "2020-10-01 00:00:00+00:00",
        "2020-11-01 00:00:00+00:00",
        "2020-12-01 00:00:00+00:00",
        "2021-01-01 00:00:00+00:00",
        "2021-02-01 00:00:00+00:00",
    ],
    [
        "3-20",
        "4-20",
        "5-20",
        "6-20",
        "7-20",
        "8-20",
        "9-20",
        "10-20",
        "11-20",
        "12-20",
        "1-21",
        "2-21",
    ],
)
plt.legend()
plt.ylim(0, 2500)
plt.xlabel("Date")
plt.ylabel("Number of Tweets")

# %%

df["hero_weighted"] = df.hero * df.likes
df["villain_weighted"] = df.villain * df.likes
df["victim_weighted"] = df.victim * df.likes

daily_weighted = (
    df[["hero_weighted", "villain_weighted", "victim_weighted"]].resample("D").sum()
)

plt.plot(daily_weighted.index, daily_weighted.hero, label_weighted="Hero")
plt.plot(daily_weighted.index, daily_weighted.villain, label="Villain")
plt.plot(daily_weighted.index, daily_weighted.victim, label="Victim")
plt.legend()

# %%
