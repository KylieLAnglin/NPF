# %%
import os

import pandas as pd
import numpy as np
import string
from NPF.teachers_and_covid import start

NUM_WORDS_TO_RATE = 5
SEED = 295

NO_BELOW = [10, 100, 500, 1000]
NO_ABOVE = [0.25, 0.5, 1.0]
# %%

# %%

# %% Word ratings
big_words_df = pd.DataFrame()

for no_below, no_above in zip(NO_BELOW, NO_ABOVE):
    folder_name = "lsa_no_below_" + str(no_below) + "_no_above_" + str(no_above)

    folder = start.RESULTS_DIR + "LSA/" + folder_name
    topic_words = pd.read_csv(folder + "/lsa_words.csv")
    topic_words = topic_words.rename(columns={"Unnamed: 0": "word"})

    topic_names = []
    list_words = []
    topic_list = list(range(0, 100))
    for topic in topic_list:
        words = list(
            topic_words.sort_values(by=str(topic), ascending=False).head(5).word
        )
        list_words.append(words)
        topic_names.append(folder_name + "_" + str(topic))

    words_df = pd.DataFrame(list_words).T
    words_df.columns = topic_names
    big_words_df = pd.concat([big_words_df, words_df], axis=1)


big_words_df_for_export = big_words_df.sample(frac=1, axis=1, random_state=SEED)
big_words_df_for_export.to_excel(
    start.ANNOTATIONS_DIR + "word_coherence_ratings_lsa.xlsx"
)
# %% Tweet ratings

big_tweets_df = pd.DataFrame()

for no_below, no_above in zip(NO_BELOW, NO_ABOVE):
    folder_name = "lsa_no_below_" + str(no_below) + "_no_above_" + str(no_above)
    folder = start.RESULTS_DIR + "LSA/" + folder_name
    tweet_topic_assignments = pd.read_csv(folder + "/lsa_tweets.csv")

    topic_names = []
    list_tweets = []
    topic_list = list(range(0, 100))
    for topic in topic_list:
        tweets = list(
            tweet_topic_assignments.sort_values(by=str(topic), ascending=False)
            .sample(100, random_state=SEED)
            .head(5)
            .tweet_text
        )
        list_tweets.append(tweets)
        topic_names.append(folder_name + "_" + str(topic))

    tweets_df = pd.DataFrame(list_tweets).T
    tweets_df.columns = topic_names
    big_tweets_df = pd.concat([big_tweets_df, tweets_df], axis=1)

big_tweets_df_for_export = big_tweets_df.sample(frac=1, axis=1, random_state=SEED)
big_tweets_df_for_export.to_excel(
    start.ANNOTATIONS_DIR + "tweet_coherence_ratings_lsa.xlsx"
)
# %%
