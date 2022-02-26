# %%
import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start

teacher = pd.read_csv(start.CLEAN_DIR + "teacher.csv")
teacher["set"] = 1
teachers = pd.read_csv(start.CLEAN_DIR + "teachers.csv")
teachers["set"] = 2

df = teacher.append(teachers)
df["unique_id"] = df["set"].astype("str") + df.index.astype("str")

len(df)

df = df.sample(len(df), random_state=100)
# %%

np.random.seed(5)
df["random_set"] = np.random.randint(1, 300, size=len(df))
df["relevant"] = ""
df["category"] = 0
df["notes"] = ""

df.to_csv(
    start.CLEAN_DIR + "tweets_full.csv",
    encoding="utf-8",
)
