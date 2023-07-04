# %%

import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv(start.CLEAN_DIR + "tweets_relevant_labeled.csv")
df["hero"] = np.where(df.character_classification == "Hero", 1, 0)
df["victim"] = np.where(df.character_classification == "Victim", 1, 0)
df["villain"] = np.where(df.character_classification == "Villain", 1, 0)

df["date"] = pd.to_datetime(df.created, errors="coerce")
df["month"] = df["date"].dt.month

df["month_centered"] = np.where(df.month == 1, 13, df.month)
df["month_centered"] = np.where(df.month == 2, 14, df.month_centered)
df["month_centered"] = df.month_centered - 3
# %%
mod = smf.ols(formula="hero ~ month_centered", data=df)
res = mod.fit()
print(res.summary())
# %%
mod = smf.ols(formula="hero ~ C(month_centered)", data=df)
res = mod.fit()
print(res.summary())
# %%
