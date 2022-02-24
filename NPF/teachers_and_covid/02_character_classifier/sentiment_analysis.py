# %%
from cgi import test
import re
import pandas as pd
import numpy as np

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import liwc

from NPF.teachers_and_covid import start
from NPF.library import classify

# %%
tweets = pd.read_csv(start.MAIN_DIR + "tweets_full.csv")
tweets = tweets[
    [
        "unique_id",
        "text",
        "created",
        "likes",
        "retweets",
        "quotes",
        "replies",
        "author_id",
        "geo",
        "random_set",
    ]
]
annotations = pd.read_csv(start.MAIN_DIR + "annotations_characters.csv")


df = annotations.merge(
    tweets[["unique_id", "text", "random_set"]],
    how="left",
    left_on="unique_id",
    right_on="unique_id",
)

# %%
df["polarity"] = [TextBlob(tweet).sentiment.polarity for tweet in df.text]
df["subjectivity"] = [TextBlob(tweet).sentiment.subjectivity for tweet in df.text]
# %%

analyzer = SentimentIntensityAnalyzer()

df["tone_pos"] = [analyzer.polarity_scores(tweet)["pos"] for tweet in df.text]
df["tone_neg"] = [analyzer.polarity_scores(tweet)["neg"] for tweet in df.text]
df["tone_neutral"] = [analyzer.polarity_scores(tweet)["neu"] for tweet in df.text]
df["tone_compound"] = [analyzer.polarity_scores(tweet)["compound"] for tweet in df.text]

# %%
parse, category_names = liwc.load_token_parser(
    "/Users/kla21002/LIWC2015.Dictionary.English.2021.04.12.65672.dic"
)

# %%
