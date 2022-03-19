# %%
import pandas as pd

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from NPF.teachers_and_covid import start
from NPF.library import process_text

# %%
tweets = pd.read_csv(start.CLEAN_DIR + "tweets_full.csv").set_index("unique_id")

relevance = pd.read_csv(start.TEMP_DIR + "tweets_classified.csv")
relevance["unique_id"] = pd.to_numeric(relevance.unique_id, errors="coerce")
relevance = relevance.set_index("unique_id")

annotations = pd.read_csv(start.CLEAN_DIR + "annotations_characters.csv").set_index(
    "unique_id"
)


df = tweets.merge(
    annotations[["relevant", "category", "hero", "villain", "victim"]],
    how="left",
    indicator=True,
    left_index=True,
    right_index=True,
)

df = df.merge(relevance, left_index=True, right_index=True)

# %%
df = df[(df.classification == 1) | (df._merge == "both")]

df = df.rename(
    columns={
        "text": "tweet_text",
        "created": "tweet_created",
        "likes": "tweet_likes",
        "retweets": "tweet_retweets",
        "quotes": "tweet_quotes",
        "replies": "tweet_replies",
        "geo": "tweet_geo",
        "relevant": "tweet_relevant",
        "category": "tweet_category",
        "hero": "tweet_hero",
        "villain": "tweet_villain",
        "victim": "tweet_victim",
    }
)

df.to_csv(start.CLEAN_DIR + "tweets_relevant.csv")
