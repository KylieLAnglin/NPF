# %%
import pandas as pd
import numpy as np

from NPF.teachers_and_covid import start
import matplotlib.pyplot as plt

df = pd.read_csv(start.CLEAN_DIR + "tweets_final.csv")

liwc = pd.read_csv(start.TEMP_DIR + "features_liwc.csv")

df = df.merge(
    liwc, left_on="unique_id", right_on="unique_id", how="inner", indicator="liwc_merge"
)

df["liwc_death"] = np.where(df.death > 1, 1, 0)

# %%

df["date"] = pd.to_datetime(df.created, errors="coerce")
df["month"] = df["date"].dt.month

df = df.set_index("date")

# %%
daily = df[["liwc_death"]].resample("D").sum()

# plt.plot(daily.index, daily.hero, label="Hero")
# plt.plot(daily.index, daily.villain, label="Villain", color="orange")
plt.plot(daily.index, daily.thank, label="Tweets that Death Terms", color="black")
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
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
# %%


# %%
thank_list = ["thank" in text.lower() for text in df.text_x]
df["thank"] = np.where(thank_list, 1, 0)
daily = df[["thank"]].resample("D").sum()


plt.plot(daily.index, daily.thank, label="Thank you", color="black")
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
plt.xlabel("Date")
plt.ylabel("Number of Tweets")

# %%
