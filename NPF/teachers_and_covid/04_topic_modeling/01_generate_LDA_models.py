# %%
import os

import pandas as pd
import numpy as np

import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.hdpmodel import HdpModel
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary
from numpy import array
from tqdm import tqdm

from NPF.teachers_and_covid import start
from NPF.library import process_text
from NPF.library import topic_modeling


# %%
df = pd.read_csv(start.CLEAN_DIR + "tweets_relevant.csv")

features = pd.read_csv(start.MAIN_DIR + "data/clean/features.csv")
matrix = features[[col for col in features.columns if "term_" in col]]
matrix.columns = [col.replace("term_", "") for col in matrix.columns]
# %% Consider replacing
docs = list(df.text)


# %% Tokenize, lowercase


def make_lower(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


df["tweet_text_clean"] = df.text.apply(make_lower)
df["tweet_text_clean"] = df.text.apply(remove_punctuation)

docs = [
    process_text.process_text_nltk(
        text,
        lower_case=True,
        remove_punct=False,
        remove_stopwords=True,
        lemma=True,
        string_or_list="list",
    )
    for text in df.tweet_text_clean
]


# %%

# Compute bigrams.

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if "_" in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

# Create a dictionary representation of the documents.


# %%

grid = []
for no_below in [0, 50, 100]:
    for no_above in [1, 0.5]:
        for num_topics in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            grid.append(
                {"no_below": no_below, "no_above": no_above, "num_topics": num_topics}
            )
len(grid)
# %%

pbar = tqdm(total=len(grid))
for parameters in grid[0:1]:
    model_name = (
        "topic_"
        + str(parameters["num_topics"])
        + "_no_below"
        + str(parameters["no_below"])
        + "_no_above_"
        + str(parameters["no_above"])
    )
    newpath = start.RESULTS_DIR + "topic_models/" + model_name + "/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    pbar.update(1)
    dictionary = Dictionary(docs)

    dictionary.filter_extremes(
        no_below=parameters["no_below"], no_above=parameters["no_above"]
    )

    corpus = [dictionary.doc2bow(doc) for doc in docs]
    # TODO: Start here

    lda = gensim.models.LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=parameters["num_topics"],
        passes=1,
        random_state=44,
        per_word_topics=True,
    )

    topic_modeling.create_topic_tables(
        lda=lda,
        corpus=corpus,
        dictionary=dictionary,
        tweets_df=df,
        num_topics=parameters["num_topics"],
        folder_path=newpath,
    )
pbar.close()


# %%
# Filter out words that occur less than 20 documents, or more than 50% of the documents.

# %%

goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=2)
badLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=1, num_topics=2)

# %%
goodcm = CoherenceModel(
    model=goodLdaModel, corpus=corpus, dictionary=dictionary, coherence="u_mass"
)
badcm = CoherenceModel(
    model=badLdaModel, corpus=corpus, dictionary=dictionary, coherence="u_mass"
)

# %%
