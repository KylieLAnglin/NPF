# %%
import pandas as pd
from NPF.teachers_and_covid import start
from NPF.library import classify

# %%
tweets = pd.read_csv(start.MAIN_DIR + "tweets_classified.csv")
tweets = tweets[tweets.classification_rule == 1]


# %%
training_batch_5 = tweets[tweets.random_set == 5].to_csv(
    start.MAIN_DIR + "training_batch5.csv"
)
training_batch_6 = tweets[tweets.random_set == 6].to_csv(
    start.MAIN_DIR + "training_batch6.csv"
)
training_batch_7 = tweets[tweets.random_set == 7].to_csv(
    start.MAIN_DIR + "training_batch7.csv"
)
