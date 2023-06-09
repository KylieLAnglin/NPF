# %%
import os

import pandas as pd
import numpy as np
import string
from NPF.teachers_and_covid import start

NUM_WORDS_TO_RATE = 5
SEED = 295
# %%
coherence_df = pd.read_csv(start.RESULTS_DIR + "topic_models_coherence.csv")

no_below_mean = coherence_df[["no_below", "coherence"]].groupby("no_below").mean()
no_above_mean = coherence_df[["no_above", "coherence"]].groupby("no_above").mean()

# Keep if coherence less than 4
# Create filename for import

# %%
better_models_df = coherence_df.sort_values(by="coherence", ascending=False).head(12)

# %% Word ratings
big_words_df = pd.DataFrame()

for num_topics, no_below, no_above in zip(
    better_models_df.num_topics, better_models_df.no_below, better_models_df.no_above
):
    if no_above == 1.0:
        no_above = 1
    folder_name = (
        "topic_"
        + str(num_topics)
        + "_no_below_"
        + str(no_below)
        + "_no_above_"
        + str(no_above)
    )

    folder = start.RESULTS_DIR + "topic_models/" + folder_name
    topic_words = pd.read_csv(folder + "/topics.csv")
    words = topic_words.head(5)
    words = words[[col for col in words if "Word" in col]]
    words.columns = [folder_name + "_" + col for col in words.columns]
    big_words_df = pd.concat([big_words_df, words], axis=1)


big_words_df_for_export = big_words_df.sample(frac=1, axis=1, random_state=SEED)
big_words_df_for_export.to_excel(start.ANNOTATIONS_DIR + "word_coherence_ratings.xlsx")
# %% Tweet ratings

big_tweets_df = pd.DataFrame()

for num_topics, no_below, no_above in zip(
    better_models_df.num_topics, better_models_df.no_below, better_models_df.no_above
):
    if no_above == 1.0:
        no_above = 1
    folder_name = (
        "topic_"
        + str(num_topics)
        + "_no_below_"
        + str(no_below)
        + "_no_above_"
        + str(no_above)
    )

    folder = start.RESULTS_DIR + "topic_models/" + folder_name
    tweet_topic_assignments = pd.read_csv(folder + "/tweet_topics.csv")
    topic_names = []
    list_tweets = []
    topic_list = list(range(0, num_topics))
    for topic in topic_list:
        tweets = list(
            tweet_topic_assignments.sort_values(by=str(topic), ascending=False)
            .head(100)
            .sample(5, random_state=SEED)
            .text
        )
        list_tweets.append(tweets)
        topic_names.append(folder_name + "_" + str(topic))

    tweets_df = pd.DataFrame(list_tweets).T
    tweets_df.columns = topic_names
    big_tweets_df = pd.concat([big_tweets_df, tweets_df], axis=1)

big_tweets_df_for_export = big_tweets_df.sample(frac=1, axis=1, random_state=529)
big_tweets_df_for_export.to_excel(
    start.ANNOTATIONS_DIR + "tweet_coherence_ratings.xlsx"
)
# %%
