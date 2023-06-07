# %%
import pandas as pd
from NPF.teachers_and_covid import start

# %%
relevant_tweets = pd.read_csv(start.CLEAN_DIR + "tweets_relevant.csv")

batch1 = relevant_tweets[relevant_tweets.random_set < 8]
batch2 = relevant_tweets[
    (relevant_tweets.random_set < 18) & (relevant_tweets.random_set > 8)
]
batch = batch1.append(batch2)

# %%
triple_code = batch.tail(len(batch) - 3000)
triple_code[
    [
        "unique_id",
        "text",
    ]
].to_excel(
    start.MAIN_DIR + "data/temp/character_annotations_for_triple_coding.xlsx",
    index=False,
)


# %%
batch = batch.head(3000)

# %%
annotations = batch.sort_values(by="unique_id")
annotations[
    [
        "unique_id",
        "text",
    ]
].to_excel(start.MAIN_DIR + "data/annotations/character_annotations.xlsx", index=False)


# %%


# %%
