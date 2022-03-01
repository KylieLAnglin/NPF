# %%
import pandas as pd
import numpy as np

from NPF.teachers_and_covid import start
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(start.CLEAN_DIR + "annotations_characters.csv")
df["hero"] = np.where(df.category == 1, 1, 0)
df["villain"] = np.where(df.category == 2, 1, 0)
df["victim"] = np.where(df.category == 3, 1, 0)

df["date"] = pd.to_datetime(df.created)
df["month"] = df["date"].dt.month

melted = df[["month", "hero", "villain", "victim"]].groupby("month").agg("sum")
# %%
df = df.set_index("date")
monthly = df[["hero", "villain", "victim"]].resample("M").sum()

# %%
plt.plot(monthly.index, monthly.hero, label="Hero")
plt.plot(monthly.index, monthly.villain, label="Villain")
plt.plot(monthly.index, monthly.victim, label="Victim")
plt.legend()
# %%

# %%
daily = df[["hero", "villain", "victim"]].resample("D").sum()

plt.plot(daily.index, daily.hero, label="Hero")
plt.plot(daily.index, daily.villain, label="Villain")
plt.plot(daily.index, daily.victim, label="Victim")
plt.legend()

# %%
df2 = df
df2["hero"] = df2.hero * df2.likes
df2["villain"] = df2.villain * df2.likes
df2["victim"] = df2.victim * df2.likes

daily = df2[["hero", "villain", "victim"]].resample("D").sum()

plt.plot(daily.index, daily.hero, label="Hero")
plt.plot(daily.index, daily.villain, label="Villain")
plt.plot(daily.index, daily.victim, label="Victim")
plt.legend()

# %%
