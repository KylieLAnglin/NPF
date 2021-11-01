# %%
import re
import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# import pyLDAvis.gensim
import nltk
import gensim

# %%
df = pd.read_csv(start.MAIN_DIR + "training_batch1_annotated.csv")


# %%
def tweet_to_words(tweets):
    for tweet in tweets:
        yield (
            gensim.utils.simple_preprocess(str(tweet), deacc=True)
        )  # deacc=True removes punctuations


data_words = list(tweet_to_words(df.text))

# %%

# Build the bigram and trigram models
bigram = gensim.models.Phrases(
    data_words, min_count=3, threshold=20
)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=20)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[6]]])

# %%
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    stop_words = nltk.corpus.stopwords.words("english")
    return [
        [
            word
            for word in gensim.utils.simple_preprocess(str(doc))
            if word not in stop_words
        ]
        for doc in texts
    ]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)


print(data_words_nostops[:1])

data = data_words_nostops
# %%

# Create Dictionary
id2word = gensim.corpora.Dictionary(data)

# Create Corpus
texts = data

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# [[(id2word[id], freq) for id, freq in cp] for cp in corpus]
# %%
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=7,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
    per_word_topics=True,
)
# Print the Keyword in the 10 topics
print(lda_model.print_topics())
doc_lda = lda_model[corpus]
# %%
# Visualize the topics
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, id2word)
vis
# %%
