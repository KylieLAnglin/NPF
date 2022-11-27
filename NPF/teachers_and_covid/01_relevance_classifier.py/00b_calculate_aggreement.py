# %%

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import simpledorff

from NPF.teachers_and_covid import start


# %%
tweets = pd.read_csv(start.CLEAN_DIR + "tweets_full.csv", low_memory=False)
tweets = tweets[
    [
        "unique_id",
        "text",
        "created",
        "likes",
        "retweets",
        "quotes",
        "replies",
        "author_id",
        "geo",
        "random_set",
    ]
]
# %%
annotations_A = pd.read_csv(
    start.CLEAN_DIR + "annotations/validation_relevance_batch1_annotated_JE.csv",
    encoding="utf-8",
)
annotations_A["annotator"] = "A"

annotations_B = pd.read_csv(
    start.CLEAN_DIR + "annotations/validation_relevance_batch1_annotated_KA.csv"
)
annotations_B["annotator"] = "B"

annotations_long = pd.concat([annotations_A, annotations_B])
annotations_long = annotations_long[["unique_id", "relevant", "category", "annotator"]]

# %% Annotations wide

annotations_A = annotations_A.rename(columns={"relevant": "relevant_A"})
annotations_B = annotations_B.rename(columns={"relevant": "relevant_B"})

annotations = annotations_A[
    ["unique_id", "tweet_id", "random_set", "text", "relevant_A"]
].merge(
    annotations_B[["unique_id", "tweet_id", "relevant_B"]],
    left_on=["unique_id", "tweet_id"],
    right_on=["unique_id", "tweet_id"],
)
# %%
annotations = annotations.head(500)
annotations.loc[:, "relevant_A"] = np.where(annotations.relevant_A == 1, 1, 0)
annotations.loc[:, "relevant_B"] = np.where(annotations.relevant_B == 1, 1, 0)

annotations.loc[:, "agree"] = np.where(
    annotations.relevant_A == annotations.relevant_B, 1, 0
)
aggreement = annotations.agree.mean()
print(f"Agreement is {aggreement.round(2)}.")
# %%

cohen_kappa_score(annotations.relevant_A, annotations.relevant_B)
# Minimal aggreement...

# %%
simpledorff.calculate_krippendorffs_alpha_for_df(
    annotations_long,
    experiment_col="unique_id",
    annotator_col="annotator",
    class_col="relevant",
)

# %%
