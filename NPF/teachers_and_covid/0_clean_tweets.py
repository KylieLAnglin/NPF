# %%
import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from openpyxl import load_workbook
from functools import reduce


import nltk
import gensim
from nltk.tokenize import TweetTokenizer


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
