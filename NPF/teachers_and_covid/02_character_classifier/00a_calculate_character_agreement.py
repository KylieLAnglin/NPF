# %%

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score
import simpledorff

from NPF.teachers_and_covid import start

# %%
jessica_codes = pd.read_excel(
    start.ANNOTATIONS_DIR + "character_annotations_for_triple_coding_JG.xlsx"
)
jessica_codes = jessica_codes.head(100)
jessica_codes["character"] = np.where(
    jessica_codes.character.isnull(), "Irrelevant", jessica_codes.character
)
jessica_codes = jessica_codes.rename(columns={"character": "character_JG"})
# %%
joe_codes = pd.read_excel(
    start.ANNOTATIONS_DIR + "character_annotations_for_triple_coding_JE.xlsx"
)
joe_codes = joe_codes.head(100)
joe_codes["character"] = np.where(
    joe_codes.character.isnull(), "Other", joe_codes.character
)
joe_codes = joe_codes.rename(columns={"character": "character_JE"})

# %%
kylie_codes = pd.read_excel(
    start.ANNOTATIONS_DIR + "character_annotations_for_triple_coding_KLA.xlsx"
)
kylie_codes = kylie_codes.head(100)
kylie_codes["character"] = np.where(
    kylie_codes.character.isnull(), "Other", kylie_codes.character
)
kylie_codes = kylie_codes.rename(columns={"character": "character_KLA"})


# %% ZM Codes
zack_codes = pd.read_excel(
    start.ANNOTATIONS_DIR + "character_annotations_for_triple_coding_ZM.xlsx"
)
zack_codes = zack_codes.head(100)
zack_codes["character"] = np.where(
    zack_codes.character.isnull(), "Other", zack_codes.character
)
zack_codes = zack_codes.rename(columns={"character": "character_ZM"})



# %% TF Codes
teagan_codes = pd.read_excel(
    start.ANNOTATIONS_DIR + "character_annotations_for_triple_coding_TF.xlsx"
)
teagan_codes = teagan_codes.head(100)
teagan_codes["character"] = np.where(
    teagan_codes.character.isnull(), "Other", teagan_codes.character
)
teagan_codes = teagan_codes.rename(columns={"character": "character_TF"})


# %% Merge to view
df = kylie_codes[["unique_id", "character_KLA"]].merge(
    jessica_codes[["unique_id", "character_JG"]], how="outer", on=["unique_id"]
)

df = df.merge(
    joe_codes[["unique_id", "character_JE"]], how="outer", on=["unique_id"]
)

df = df.merge(
    zack_codes[["unique_id", "character_ZM"]], how="outer", on=["unique_id"]
)

df = df.merge(
    teagan_codes[["unique_id", "character_TF", "text"]], how="outer", on=["unique_id"]
)
df.to_excel(
    start.ANNOTATIONS_DIR + "character_annotations_for_triple_coding_compare.xlsx"
)

# %%
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
