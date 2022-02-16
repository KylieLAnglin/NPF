# %%
import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start
from functools import reduce

# import spacy
# nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
import nltk
import gensim
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
import gensim.corpora as corpora

STOPLIST = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    # "during",
    # "before",
    # "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "https",
    "amp",
    "co",
    "th",
    "&",
]


def process_text(
    text_corpus, trigram=True, stopwords=True, no_below=10, no_above=0.5, keep_n=2000
):
    texts = list(sent_to_words(text_corpus))
    if trigram:
        texts = make_trigrams(texts)
    if stopwords:
        texts = remove_stopwords(texts)
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    return dictionary, texts


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [
        [word for word in simple_preprocess(str(doc)) if word not in STOPLIST]
        for doc in texts
    ]


def make_bigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=100, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=100, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram = gensim.models.Phrases(bigram[texts], min_count=100, threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]  # %%


# def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
#     """https://spacy.io/api/annotation"""
#     texts_out = []
#     for sent in texts:
#         doc = nlp(" ".join(sent))
#         texts_out.append(
#             [token.lemma_ for token in doc if token.pos_ in allowed_postags]
#         )
#     return texts_out

# %%
