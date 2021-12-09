# %%
import re
import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from pprint import pprint
from gensim import models
from collections import defaultdict
from gensim import corpora
from gensim import similarities

import nltk
import gensim
from nltk.tokenize import TweetTokenizer

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
# %%
annotations1 = pd.read_csv(
    start.MAIN_DIR + "training_batch1_annotated.csv", encoding="utf-8’"
)
annotations2 = pd.read_csv(
    start.MAIN_DIR + "training_batch2_annotated.csv", encoding="utf-8’"
)

annotations = pd.concat([annotations1, annotations2])
annotations = annotations[(annotations.relevant == 1) | (annotations.relevant == 0)]

df = tweets.merge(
    annotations[["unique_id", "relevant", "category"]],
    left_on="unique_id",
    right_on="unique_id",
)

df = df[df.relevant == 1]
df.category.value_counts()

# %%
list(df.head(1).text)

text_corpus = list(df.text)

# Create a set of frequent words
stoplist = set(
    "for a of the and to in are our is this that with not have be their it on they so we i you all &amp; my your who do as but how will from".split(
        " "
    )
)
# Lowercase each document, split it by white space and filter out stopwords
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in text_corpus
]

# Count word frequencies

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint(processed_corpus)


for w in sorted(frequency, key=frequency.get, reverse=True):
    print(w, frequency[w])

# %%
dictionary = corpora.Dictionary(texts)
print(dictionary)

# %%
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)


corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)
# %%
lsi_model = models.LsiModel(
    corpus_tfidf, id2word=dictionary, num_topics=2
)  # initialize an LSI transformation
corpus_lsi = lsi_model[
    corpus_tfidf
]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi


# %%
for doc, as_text in zip(corpus_lsi, text_corpus):
    print(doc, as_text)
# %%

# # %%
# # %%
# def tweet_to_words(tweets):
#     for tweet in tweets:
#         yield (
#             gensim.utils.simple_preprocess(str(tweet), deacc=True)
#         )  # deacc=True removes punctuations


# data_words = list(tweet_to_words(df.text))

# # %%

# # Build the bigram and trigram models
# bigram = gensim.models.Phrases(
#     data_words, min_count=3, threshold=20
# )  # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=20)

# # Faster way to get a sentence clubbed as a trigram/bigram
# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

# # See trigram example
# print(trigram_mod[bigram_mod[data_words[6]]])

# # %%
# # Define functions for stopwords, bigrams, trigrams and lemmatization
# def remove_stopwords(texts):
#     stop_words = nltk.corpus.stopwords.words("english")
#     return [
#         [
#             word
#             for word in gensim.utils.simple_preprocess(str(doc))
#             if word not in stop_words
#         ]
#         for doc in texts
#     ]


# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]


# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]


# # Remove Stop Words
# data_words_nostops = remove_stopwords(data_words)

# # Form Bigrams
# data_words_bigrams = make_bigrams(data_words_nostops)


# print(data_words_nostops[:1])

# data = data_words_nostops
# # %%

# # Create Dictionary
# id2word = gensim.corpora.Dictionary(data)

# # Create Corpus
# texts = data

# # Term Document Frequency
# corpus = [id2word.doc2bow(text) for text in texts]

# # View
# print(corpus[:1])

# # [[(id2word[id], freq) for id, freq in cp] for cp in corpus]
# # %%
# # Build LDA model
# lda_model = gensim.models.ldamodel.LdaModel(
#     corpus=corpus,
#     id2word=id2word,
#     num_topics=7,
#     random_state=100,
#     update_every=1,
#     chunksize=100,
#     passes=10,
#     alpha="auto",
#     per_word_topics=True,
# )
# # Print the Keyword in the 10 topics
# print(lda_model.print_topics())
# doc_lda = lda_model[corpus]
# # %%
# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = gensimvis.prepare(lda_model, corpus, id2word)
# vis
# # %%
