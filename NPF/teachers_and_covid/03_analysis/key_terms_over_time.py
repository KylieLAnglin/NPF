# %%
import pandas as pd
import numpy as np

from NPF.teachers_and_covid import start
import matplotlib.pyplot as plt

df = pd.read_csv(start.CLEAN_DIR + "tweets_full.csv")

# liwc = pd.read_csv(start.TEMP_DIR + "features_liwc.csv")

# df = df.merge(
#     liwc, left_on="unique_id", right_on="unique_id", how="inner", indicator="liwc_merge"
# )

# df["liwc_death"] = np.where(df.death > 1, 1, 0)

# %%

df["date"] = pd.to_datetime(df.created, errors="coerce")
df["month"] = df["date"].dt.month

df = df.set_index("date")

# %%
risk_list = [" risk " in text.lower() for text in df.text]
df["risk"] = np.where(risk_list, 1, 0)
daily_death = df[["risk"]].resample("D").sum()

# plt.plot(daily.index, daily.hero, label="Hero")
# plt.plot(daily.index, daily.villain, label="Villain", color="orange")
plt.plot(
    daily_death.index,
    daily_death.risk,
    label="Tweets that Death Terms",
    color="black",
)
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
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "1",
        "2",
    ],
)
plt.xlabel("Date")
plt.ylabel("Number of Tweets")

# %%
thank_list = ["thank" in text.lower() for text in df.text]
df["thank"] = np.where(thank_list, 1, 0)
daily_thank = df[["thank"]].resample("D").sum()


plt.plot(daily_thank.index, daily_thank.thank, label="Thank you", color="black")
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
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "1",
        "2",
    ],
)
plt.xlabel("Date")
plt.ylabel("Number of Tweets")

# %%
union_list = ["union" in text.lower() for text in df.text]
df["union"] = np.where(union_list, 1, 0)
daily_union = df[["union"]].resample("D").sum()


plt.plot(daily_union.index, daily_union.union, label="Union", color="black")
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
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "1",
        "2",
    ],
)
plt.xlabel("Date")
plt.ylabel("Number of Tweets")

# %%
