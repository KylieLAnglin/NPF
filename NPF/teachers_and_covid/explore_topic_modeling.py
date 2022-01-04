# %%
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import pprint

# %%
num_topics = 5
num_words_to_view = 5
passes = 20

# %%
filename = "/Users/kla21002/Downloads/introSectionsToChapters.txt"

with open(filename, encoding="utf-8") as f:
    documents = f.readlines()

# %%
texts = [
    [
        word
        for word in document.lower().split()
        if word not in STOPWORDS and word.isalnum()
    ]
    for document in documents
]
# %%
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = LdaModel(
    corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=4
)
# %%

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(lda.print_topics(num_words=num_words_to_view))


# %%
newdoc = documents[0]
newcorpus = dictionary.doc2bow(
    newword
    for newword in newdoc.lower().split()
    if newword not in STOPWORDS and newword.isalnum()
)

# %%
pp.pprint(lda[newcorpus])
# %%
