# %%
import pandas as pd
import numpy as np

teacher = pd.read_csv(
    "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/teacher.csv"
)
teacher["set"] = 1
teachers = pd.read_csv(
    "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/teachers.csv"
)
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
    "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/tweets_full.csv",
    encoding="utf-8",
)
