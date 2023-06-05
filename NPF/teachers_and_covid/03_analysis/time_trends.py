# %%

import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start
import matplotlib.pyplot as plt

MAX_TWEETS = 800

df = pd.read_csv(start.CLEAN_DIR + "tweets_relevant_labeled.csv")
df["hero"] = np.where(df.character_classification == "Hero", 1, 0)
df["victim"] = np.where(df.character_classification == "Victim", 1, 0)
df["villain"] = np.where(df.character_classification == "Villain", 1, 0)

df["date"] = pd.to_datetime(df.created, errors="coerce")
df["month"] = df["date"].dt.month

df = df.set_index("date")

daily = df[["hero", "villain", "victim"]].resample("D").sum()

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot Hero Tweets
axs[0, 0].plot(daily.index, daily.hero, label="Hero Tweets", color="black")
axs[0, 0].set_xticks([
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
])
axs[0, 0].set_xticklabels([
    "3-20", "4-", "5-", "6-", "7-", "8-", "9-", "10-", "11-", "12-", "1-21", "2-",
])
axs[0, 0].legend()
axs[0, 0].set_ylim(0, MAX_TWEETS)
axs[0, 0].set_xlabel("Date")
axs[0, 0].set_ylabel("Number of Tweets")

# Plot Victim Tweets
axs[0, 1].plot(daily.index, daily.victim, label="Victim Tweets", color="black")
axs[0, 1].set_xticks([
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
])
axs[0, 1].set_xticklabels([
    "3-20", "4-", "5-", "6-", "7-", "8-", "9-", "10-", "11-", "12-", "1-21", "2-",
])
axs[0, 1].legend()
axs[0, 1].set_ylim(0, MAX_TWEETS)
axs[0, 1].set_xlabel("Date")
axs[0, 1].set_ylabel("Number of Tweets")

# Plot Villain Tweets
axs[1, 0].plot(daily.index, daily.villain, label="Villain Tweets", color="black")
axs[1, 0].set_xticks([
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
])
axs[1, 0].set_xticklabels([
    "3-20", "4-", "5-", "6-", "7-", "8-", "9-", "10-", "11-", "12-", "1-21", "2-",
])
axs[1, 0].legend()
axs[1, 0].set_ylim(0, MAX_TWEETS)
axs[1, 0].set_xlabel("Date")
axs[1, 0].set_ylabel("Number of Tweets")
axs[1,1].set_axis_off()
plt.savefig(start.RESULTS_DIR + "villain_over_time.png")

# %%
