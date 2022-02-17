# %%
import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start
from NPF.library import process
from NPF.library import topic_modeling
from functools import reduce


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from NPF.library import classify


import nltk
import gensim
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
import gensim.corpora as corpora


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
tweets["text"] = tweets["text"].str.replace("&amp;", "&")
tweets["text"] = tweets["text"].str.replace("#", "")  # remove hashtag

# %%
df = tweets
list(df.head(1).text)
text_corpus = list(df.text)

annotations = pd.read_csv(start.MAIN_DIR + "annotations.csv")

# %% 5 Topics

topic_assignments5 = pd.read_csv(
    start.MAIN_DIR + "results/" + "5topics_top500words/" + "tweet_topics.csv"
)
topic_assignments5["topic5_classification"] = np.where(
    topic_assignments5["2"] > 0.5, 1, 0
)
annotations = annotations.merge(
    topic_assignments5[["unique_id", "topic5_classification"]],
    how="left",
    left_on="unique_id",
    right_on="unique_id",
)


cf_matrix = confusion_matrix(annotations.relevant, annotations.topic5_classification)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)

recall_score(annotations.relevant, annotations.topic5_classification)
precision_score(annotations.relevant, annotations.topic5_classification)

# %%
topic_assignments10 = pd.read_csv(
    start.MAIN_DIR + "results/" + "10topics_top500words/" + "tweet_topics.csv"
)
topic_assignments10["topic10_classification"] = np.where(
    topic_assignments10["6"] > 0.5, 1, 0
)
annotations = annotations.merge(
    topic_assignments10[["unique_id", "topic10_classification"]],
    how="left",
    left_on="unique_id",
    right_on="unique_id",
)


cf_matrix = confusion_matrix(annotations.relevant, annotations.topic10_classification)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)

recall_score(annotations.relevant, annotations.topic10_classification)
precision_score(annotations.relevant, annotations.topic10_classification)

# %%
topic_assignments20 = pd.read_csv(
    start.MAIN_DIR + "results/" + "20topics_top500words/" + "tweet_topics.csv"
)
topic_assignments20["topic20_classification"] = np.where(
    topic_assignments20["16"] > 0.5, 1, 0
)
annotations = annotations.merge(
    topic_assignments20[["unique_id", "topic20_classification"]],
    how="left",
    left_on="unique_id",
    right_on="unique_id",
)


cf_matrix = confusion_matrix(annotations.relevant, annotations.topic20_classification)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)

recall_score(annotations.relevant, annotations.topic20_classification)
precision_score(annotations.relevant, annotations.topic20_classification)

# %%
