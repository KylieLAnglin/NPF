# %%
import pandas as pd

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from NPF.teachers_and_covid import start
from NPF.library import process_text
import spacy
from tqdm import tqdm
from nltk.stem import PorterStemmer
import nltk
import numpy as np

nlp = spacy.load("en_core_web_sm")
import string
from nltk.tokenize import RegexpTokenizer

# %%
df = pd.read_csv(start.CLEAN_DIR + "tweets_relevant.csv")
df = df.rename(columns={"text": "tweet_text"})
df = df[["unique_id", "tweet_text"]]
df["unique_id"] = df.unique_id.astype(int)
df = df.set_index("unique_id")
# df = df.sample(200)

annotations = pd.read_csv(
    start.CLEAN_DIR + "annotations_characters.csv", index_col="unique_id"
)
annotations = annotations[
    [
        "split",
        "character",
        "character_final",
        "tweet_hero",
        "tweet_villain",
        "tweet_victim",
    ]
]
annotations = annotations.rename(
    columns={
        "character": "tweet_character",
        "character_final": "tweet_character_final",
        "split": "tweet_split",
    }
)
df = df.merge(
    annotations, how="left", left_index=True, right_index=True, indicator=True
)
df["annotated"] = np.where(df._merge == "both", 1, 0)
df = df.drop("_merge", axis=1)


# %% Make lower, remove punctuation, stem
def make_lower(text):
    return text.lower()


df["tweet_text_clean"] = df.tweet_text.apply(make_lower)
df.sample()


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


df["tweet_text_clean"] = df.tweet_text_clean.apply(remove_punctuation)
df.sample()

stemmer = PorterStemmer()


def stem_string(text):
    token_list = []
    for token in text.split(" "):
        token_list.append(stemmer.stem(token))
    return " ".join(token_list)


df["tweet_text_clean"] = df.tweet_text_clean.apply(stem_string)
df.sample()

# %% Doc-Term Matrix
term_matrix = process_text.vectorize_text(
    df=df, text_col="tweet_text_clean", remove_stopwords=False, lemma=False, min_df=35
)
term_matrix.columns = ["term_" + str(col) for col in term_matrix.columns]
term_matrix.to_csv(start.CLEAN_DIR + "matrix.csv", index=True)

matrix_annotations = term_matrix[term_matrix.index.isin(annotations.index)]
matrix_annotations.to_csv(start.CLEAN_DIR + "matrix_annotations.csv", index=True)
# %%
lsa_matrix, word_weights = process_text.create_lsa_dfs(
    matrix=term_matrix, n_components=100
)
lsa_matrix = lsa_matrix.add_prefix("lsa_")
lsa_matrix.to_csv(start.CLEAN_DIR + "lsa.csv")

lsa_annotations = lsa_matrix[lsa_matrix.index.isin(annotations.index)]
lsa_annotations.to_csv(start.CLEAN_DIR + "lsa_annotations.csv", index=True)

# %% LIWC
liwc = pd.read_csv(start.CLEAN_DIR + "liwc_original.csv")
liwc = liwc.add_prefix("liwc_")
liwc = liwc.rename(columns={"liwc_unique_id": "unique_id"})
liwc = liwc.set_index("unique_id")
liwc.to_csv(start.CLEAN_DIR + "liwc.csv", index=True)

liwc_annotations = liwc[liwc.index.isin(annotations.index)]
liwc_annotations.to_csv(start.CLEAN_DIR + "liwc_annotations.csv", index=True)

# %%
feature_df = df.merge(liwc, left_index=True, right_index=True, how="left")
feature_df = feature_df.merge(lsa_matrix, left_index=True, right_index=True)
feature_df = feature_df.merge(term_matrix, left_index=True, right_index=True)


feature_df.to_csv(start.MAIN_DIR + "data/clean/features.csv")


features_annotations = feature_df[feature_df.index.isin(annotations.index)]
features_annotations = features_annotations[
    features_annotations.tweet_split != "validation"
]
features_annotations.to_csv(
    start.MAIN_DIR + "data/clean/features_annotations.csv", index=True
)

# %%
