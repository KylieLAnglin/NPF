# %%
import pandas as pd
from NPF.teachers_and_covid import start
from NPF.library import classify

# %%
tweet_classifications = pd.read_csv(start.TEMP_DIR + "tweets_classified.csv")
tweet_classifications["unique_id"] = pd.to_numeric(
    tweet_classifications.unique_id, errors="coerce"
)
tweets = pd.read_csv(start.CLEAN_DIR + "tweets_full.csv")

tweets = tweets.merge(
    tweet_classifications[["unique_id", "classification"]],
    left_on="unique_id",
    right_on="unique_id",
    how="left",
    indicator=True,
)

tweets = tweets[tweets.classification == 1]

# %%
training_batch_5 = tweets[tweets.random_set == 5]
training_batch_5.to_csv(
    start.CLEAN_DIR + "temp/training_batch5.csv", encoding="utf-8"
)  # joe

training_batch_6 = tweets[tweets.random_set == 6]
training_batch_6.to_csv(
    start.CLEAN_DIR + "temp/training_batch6.csv", encoding="utf-8"
)  # jessica

training_batch_7 = tweets[tweets.random_set == 7]
training_batch_7.to_csv(
    start.CLEAN_DIR + "temp/training_batch7.csv", encoding="utf-8"
)  # kylie


training_batch_8 = tweets[tweets.random_set == 8]
training_batch_8.to_csv(
    start.CLEAN_DIR + "temp/training_batch8.csv", encoding="utf-8"
)  # joe

training_batch_9 = tweets[tweets.random_set == 9]
training_batch_9.to_csv(
    start.CLEAN_DIR + "temp/training_batch9.csv", encoding="utf-8"
)  # jessica coded by Kylie

training_batch_10 = tweets[tweets.random_set == 10]
training_batch_10.to_csv(
    start.CLEAN_DIR + "temp/training_batch10.csv", encoding="utf-8"
)  # all

training_batch_11 = tweets[tweets.random_set == 11]
training_batch_11.to_csv(
    start.CLEAN_DIR + "temp/training_batch11.csv", encoding="utf-8"
)  # all

# %%
