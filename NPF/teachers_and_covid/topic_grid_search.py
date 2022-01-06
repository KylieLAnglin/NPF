# %%
import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start
from NPF.library import process
from NPF.library import topic_modeling
from functools import reduce
import os
import tqdm


# import spacy
# nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
import nltk
import gensim
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
import gensim.corpora as corpora


NUM_WORDS_TO_VIEW = 10
PASSES = 1
DROP_FREQ_BELOW = 10

NUM_TOPICS = [5, 10, 50, 100]
KEEP_TOP = [500, 1000, 100000]
DROP_FREQ_ABOVE = [0.3, 0.5]
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

# tweets = tweets.sample(1000, random_state=352)

# %%
df = tweets
list(df.head(1).text)
text_corpus = list(df.text)

# %%

pbar = tqdm.tqdm(total=len(NUM_TOPICS) * len(KEEP_TOP) * len(DROP_FREQ_ABOVE))
for i in NUM_TOPICS:
    for j in KEEP_TOP:
        for k in DROP_FREQ_ABOVE:
            model_name = "topic" + str(i) + "words" + str(j) + "freqbelow" + str(k)
            newpath = start.MAIN_DIR + "results/" + model_name + "/"
            if not os.path.exists(newpath):
                os.makedirs(newpath)

            pbar.update(1)
            dictionary, corpus = process.process_text(
                text_corpus,
                trigram=True,
                stopwords=True,
                no_below=DROP_FREQ_BELOW,
                no_above=k,
                keep_n=j,
            )
            corpus = [dictionary.doc2bow(text) for text in corpus]

            lda = gensim.models.LdaModel(
                corpus,
                id2word=dictionary,
                num_topics=i,
                passes=PASSES,
                random_state=4,
                per_word_topics=True,
            )

            topic_modeling.create_topic_tables(
                lda=lda,
                corpus=corpus,
                dictionary=dictionary,
                tweets_df=tweets,
                num_topics=i,
                folder_path=newpath,
            )
pbar.close()


# %%
