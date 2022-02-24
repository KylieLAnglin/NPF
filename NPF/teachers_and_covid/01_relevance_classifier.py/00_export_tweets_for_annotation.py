import pandas as pd
import numpy as np


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
