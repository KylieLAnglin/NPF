# %%
import pandas as pd
from NPF.teachers_and_covid import start
from NPF.library import classify

# %%
tweet_classifications = pd.read_csv(start.MAIN_DIR + "tweets_classified.csv")
tweet_classifications["unique_id"] = pd.to_numeric(
    tweet_classifications.unique_id, errors="coerce"
)
tweets = pd.read_csv(start.MAIN_DIR + "tweets_full.csv")

tweets = tweets.merge(
    tweet_classifications[["unique_id", "classification_rule"]],
    left_on="unique_id",
    right_on="unique_id",
    how="left",
    indicator=True,
)

tweets = tweets[tweets.classification_rule == 1]

# %%
training_batch_5 = tweets[tweets.random_set == 5]
training_batch_5.to_csv(start.MAIN_DIR + "training_batch5.csv")  # joe

training_batch_6 = tweets[tweets.random_set == 6]
training_batch_6.to_csv(start.MAIN_DIR + "training_batch6.csv")  # jessica

training_batch_7 = tweets[tweets.random_set == 7]
training_batch_7.to_csv(start.MAIN_DIR + "training_batch7.csv")  # kylie

# %%
