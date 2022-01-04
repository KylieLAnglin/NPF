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

tweets = tweets.sample(500, random_state=352)

# %%
df = tweets
list(df.head(1).text)
text_corpus = list(df.text)

# %%
# Stop words
stoplist = set(
    "for a of the and to in are our is this that with not have be their it on they so we i you all &amp; my your who do as but how will from".split(
        " "
    )
)

num_topics = 10
num_words_to_view = 10
passes = 2

# %%
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in text_corpus
]
# %%
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda = LdaModel(
    corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=4
)
# %%
lda.print_topics(num_words=num_words_to_view)

# %%
lda.get_document_topics(corpus[0], minimum_probability=0)

# %% Create topic tables
list_of_topic_tables = []
for topic in lda.show_topics(
    num_topics=num_topics, num_words=num_words_to_view, formatted=False
):
    list_of_topic_tables.append(
        pd.DataFrame(
            topic[1],
            columns=["Word" + "_" + str(topic[0]), "Prob" + "_" + str(topic[0])],
        )
    )

bigdf = reduce(
    lambda x, y: pd.merge(x, y, left_index=True, right_index=True), list_of_topic_tables
)
# %%
bigdf.to_csv(start.MAIN_DIR + "results/topics.csv")


# %%
lda.get_document_topics(corpus[0], minimum_probability=0)
# %%

topic_probs = [
    [
        topic_prob[1]
        for topic_prob in lda.get_document_topics(doc, minimum_probability=0)
    ]
    for doc in corpus
]
# %%
topic_probs_df = pd.DataFrame(topic_probs, columns=list(np.arange(0, num_topics)))
tweets_topics = (
    tweets[["unique_id", "text"]]
    .reset_index()
    .merge(topic_probs_df, left_index=True, right_index=True)
)
tweets_topics.to_csv(start.MAIN_DIR + "results/tweet_topics.csv")
# %%
token2id_df = (
    pd.DataFrame.from_dict(dictionary.token2id, orient="index")
    .reset_index()
    .rename(columns={"index": "term", 0: "term_id"})
)
token_topics = []
for term in token2id_df.term_id:
    token_topics.append(
        topic[1] for topic in lda.get_term_topics(term, minimum_probability=0)
    )
token_topics_df = pd.DataFrame(token_topics)
token_topics_df["term"] = token2id_df.term

cols = list(token_topics_df)
cols.insert(0, cols.pop(cols.index("term")))
token_topics_df = token_topics_df[cols]

token_topics_df.to_csv(start.MAIN_DIR + "results/word_topics.csv")
# %%
