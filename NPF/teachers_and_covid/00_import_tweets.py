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
df = pd.read_csv(
    "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/tweets_full.csv",
    encoding="utf-8",
)

df_to_code = df[df.random_set == 1]
df_to_code = df_to_code[
    ["unique_id", "tweet_id", "random_set", "text", "relevant", "category", "notes"]
]
df_to_code.to_csv(
    "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/training_batch1.csv",
    index=False,
)


df_to_code = df[df.random_set == 2]
df_to_code = df_to_code[
    ["unique_id", "tweet_id", "random_set", "text", "relevant", "category", "notes"]
]
df_to_code.to_csv(
    "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/training_batch2.csv",
    index=False,
)

df_to_code = df[df.random_set == 3]
df_to_code = df_to_code[
    ["unique_id", "tweet_id", "random_set", "text", "relevant", "category", "notes"]
]
df_to_code.to_csv(
    "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/training_batch3.csv",
    index=False,
)

df_to_code = df[df.random_set == 4]
df_to_code = df_to_code[
    ["unique_id", "tweet_id", "random_set", "text", "relevant", "category", "notes"]
]
df_to_code.to_csv(
    "/Volumes/GoogleDrive/.shortcut-targets-by-id/1wL0tiiBznVSn5Gij9hYTiC6HTPCExAn8/Policy Narratives and NLP/Data and Code/training_batch4.csv",
    index=False,
)
