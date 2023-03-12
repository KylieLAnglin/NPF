# %%

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score
import simpledorff

from NPF.teachers_and_covid import start


# %%
all_tweets = pd.read_csv(start.CLEAN_DIR + "tweets_full.csv")
relevant_tweets = pd.read_csv(start.CLEAN_DIR + "tweets_relevant.csv")
relevant_tweets = relevant_tweets[relevant_tweets.positive == 1]
tweets = relevant_tweets.merge(
    all_tweets[["unique_id"]],
    left_on="unique_id",
    right_on="unique_id",
    how="left",
    indicator=True,
)

# %%
annotations_A = pd.read_csv(
    start.ANNOTATIONS_DIR + "validation_batch1_annotated_JE.csv",
    encoding="utf-8",
)
annotations_A["annotator"] = "A"

annotations_B = pd.read_csv(
    start.ANNOTATIONS_DIR + "validation_batch1_annotated_KA.csv",
    encoding="ISO-8859-1",
)
annotations_B["annotator"] = "B"

annotations_long = pd.concat([annotations_A, annotations_B])
annotations_long = annotations_long[["unique_id", "category", "key_words", "annotator"]]

# %% Annotations wide

annotations_A = annotations_A.rename(
    columns={"category": "category_A", "key_words": "key_words_A"}
)
annotations_B = annotations_B.rename(
    columns={"category": "category_B", "key_words": "key_words_B"}
)

annotations = annotations_A[
    ["unique_id", "tweet_id", "random_set", "text", "category_A", "key_words_A"]
].merge(
    annotations_B[["unique_id", "tweet_id", "category_B", "key_words_B"]],
    left_on=["unique_id", "tweet_id"],
    right_on=["unique_id", "tweet_id"],
)



# %%
annotations = annotations.head(500)
# NOTE: Un comment for only updated relevant tweets
# annotations = annotations.merge(tweets[["unique_id", "positive"]], how="inner", on="unique_id")

# %%

annotations.loc[:, "category_A"] = np.where(
    annotations.category_A.isin([1, 2, 3]), annotations.category_A, 0
)
annotations.loc[:, "category_B"] = np.where(
    annotations.category_A.isin([1, 2, 3]), annotations.category_B, 0
)

annotations = annotations.astype({"category_A": "int", "category_B": "int"})


annotations.loc[:, "agree"] = np.where(
    annotations.category_A == annotations.category_B, 1, 0
)
aggreement = annotations.agree.mean()
print(f"Agreement is {aggreement.round(2)}.")

# %% Hero
annotations["either_hero"] = np.where(
    (annotations.category_A == 1) | (annotations.category_B == 1), 1, 0
)
either_hero = annotations.either_hero.mean()
print(f"At least one of us selected her {either_hero.round(2)} percent of the time")
agreement_hero = annotations[annotations.either_hero == 1].agree.mean()

print(f"Of those we agreed {agreement_hero.round(2)} percent of the time")

# %% Villain
annotations["either_villain"] = np.where(
    (annotations.category_A == 2) | (annotations.category_B == 2), 1, 0
)
either_villain = annotations.either_villain.mean()
print(
    f"At least one of us selected villain {either_villain.round(3)} percent of the time"
)
agreement_villain = annotations[annotations.either_villain == 1].agree.mean()

print(f"Of those we agreed {agreement_villain.round(3)} percent of the time")

# %% Victim
annotations["either_victim"] = np.where(
    (annotations.category_A == 3) | (annotations.category_A == 3), 1, 0
)
either_victim = annotations.either_victim.mean()
print(
    f"At least one of us selected victim {either_victim.round(2)} percent of the time"
)
agreement_victim = annotations[annotations.either_victim == 1].agree.mean()

print(f"Of those we agreed {agreement_victim.round(2)} percent of the time")

# %%
cohen = cohen_kappa_score(annotations.category_A, annotations.category_B)
print(f"Cohen's Kappa {cohen.round(3)} ")

# Minimal aggreement...

# %%
simpledorff.calculate_krippendorffs_alpha_for_df(
    annotations_long,
    experiment_col="unique_id",
    annotator_col="annotator",
    class_col="category",
)

# %%
