# %%
import pandas as pd
from NPF.teachers_and_covid import start
# from NPF.library import classify

# %%
# NOTE: This is not up to date. 
# Annotations were applied to previous relevance rules. 
relevant_tweets = pd.read_csv(start.CLEAN_DIR + "tweets_relevant.csv")

batch1 = relevant_tweets[relevant_tweets.random_set < 8]
batch2 = relevant_tweets[(relevant_tweets.random_set < 17) & (relevant_tweets.random_set > 8)]

batch = batch1.append(batch2)
batch = batch.head(3000)

jessica = batch.head(1000)
jessica["annotator"] = "JG"
joe = batch.tail(2000).head(1000)
joe["annotator"] = "JE"
kylie = batch.tail(1000)
kylie["annotator"] = "KLA"

annotations = jessica.append(joe).append(kylie)
annotations[["unique_id", "annotator", "text", ]].to_excel(start.MAIN_DIR + "data/temp/character_annotations.xlsx", index=False)
# %%
all_tweets = pd.read_csv(start.CLEAN_DIR + "tweets_full.csv")
relevant_tweets = relevant_tweets[relevant_tweets.positive == 1]
tweets = relevant_tweets.merge(
    all_tweets[["unique_id"]],
    left_on="unique_id",
    right_on="unique_id",
    how="left",
    indicator=True,
)


# %%
# training_batch_5 = tweets[tweets.random_set == 5]
# training_batch_5.to_csv(
#     start.CLEAN_DIR + "temp/training_batch5.csv", encoding="utf-8"
# )  # joe

# training_batch_6 = tweets[tweets.random_set == 6]
# training_batch_6.to_csv(
#     start.CLEAN_DIR + "temp/training_batch6.csv", encoding="utf-8"
# )  # jessica

# training_batch_7 = tweets[tweets.random_set == 7]
# training_batch_7.to_csv(
#     start.CLEAN_DIR + "temp/training_batch7.csv", encoding="utf-8"
# )  # kylie


# training_batch_8 = tweets[tweets.random_set == 8]
# training_batch_8.to_csv(
#     start.CLEAN_DIR + "temp/training_batch8.csv", encoding="utf-8"
# )  # joe

# training_batch_9 = tweets[tweets.random_set == 9]
# training_batch_9.to_csv(
#     start.CLEAN_DIR + "temp/training_batch9.csv", encoding="utf-8"
# )  # jessica coded by Kylie

# training_batch_10 = tweets[tweets.random_set == 10]
# training_batch_10.to_csv(
#     start.CLEAN_DIR + "temp/training_batch10.csv", encoding="utf-8"
# )  # all

# training_batch_11 = tweets[tweets.random_set == 11]
# training_batch_11.to_csv(
#     start.CLEAN_DIR + "temp/training_batch11.csv", encoding="utf-8"
# )  # all

# %%
