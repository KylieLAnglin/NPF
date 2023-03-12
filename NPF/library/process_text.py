# %%
import re
import collections

import nltk
import pandas as pd
import numpy as np
import scipy
from collections import Counter


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords


# %%
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
    "t" "d" "ll" "m" "re" "s" "ve",
]


# %%
def process_text_nltk(
    text: str,
    lower_case: bool = True,
    remove_punct: bool = True,
    remove_stopwords: bool = False,
    lemma: bool = False,
    string_or_list: str = "string",
):

    tokens = nltk.word_tokenize(text)

    if lower_case:
        tokens = [token.lower() if token.isalpha() else token for token in tokens]

    if remove_punct:
        tokens = [token for token in tokens if token.isalpha()]

    if remove_stopwords:
        tokens = [token for token in tokens if not token in STOPLIST]

    if lemma:
        tokens = [nltk.wordnet.WordNetLemmatizer().lemmatize(token) for token in tokens]

    if string_or_list != "list":
        doc = " ".join(tokens)
    else:
        doc = tokens

    return doc


# %%


def vectorize_text(
    df: pd.DataFrame,
    text_col: str,
    remove_stopwords: bool = False,
    min_df: float = 1,
    max_features: int = None,
    tfidf: bool = False,
    lemma: bool = False,
    lsa: bool = False,
    n_components: int = 100,
    n_gram_range=(1, 1),
):
    """Returns a document-term matrix

    Args:
        df (pd.DataFrame): Dataframe containing text to be vectorized in a column
        text_col (str): Name of column containing text
        remove_stopwords (bool, optional): Remove stop words?. Defaults to False.
        tfidf (bool, optional): Tf-IDF weight?. Defaults to False.
        lemma (bool, optional): Lemmatize?. Defaults to False.
        lsa (bool, optional): Perform LSA?. Defaults to False.
        n_components (int, optional): Number of LSA components to keep. Defaults to 100.
        n_gram_range (tuple, optional): Number of n-grams. Defaults to (1, 1).

    Returns:
        pd.DataFrame: Document-term matrix
    """

    docs = [
        process_text_nltk(
            text,
            lower_case=True,
            remove_punct=False,
            remove_stopwords=remove_stopwords,
            lemma=lemma,
        )
        for text in df[text_col]
    ]

    if tfidf == False:
        vec = CountVectorizer(
            ngram_range=n_gram_range, min_df=min_df, max_features=max_features
        )

    elif tfidf:
        vec = TfidfVectorizer(
            ngram_range=n_gram_range, min_df=min_df, max_features=max_features
        )

    X = vec.fit_transform(docs)
    matrix = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out(), index=df.index)

    print("Number of words: ", len(matrix.columns))

    if lsa:
        lsa_dfs = create_lsa_dfs(matrix=matrix, n_components=n_components)
        matrix = lsa_dfs.matrix
        print("Number of dimensions: ", len(matrix.columns))

    return matrix


def create_lsa_dfs(
    matrix: pd.DataFrame, n_components: int = 100, random_state: int = 100
):

    lsa = TruncatedSVD(n_components=n_components, random_state=random_state)
    lsa_fit = lsa.fit_transform(matrix)
    lsa_fit = Normalizer(copy=False).fit_transform(lsa_fit)
    print(lsa_fit.shape)

    #  Each LSA component is a linear combo of words
    word_weights = pd.DataFrame(lsa.components_, columns=matrix.columns)
    word_weights.head()
    word_weights_trans = word_weights.T

    # Each document is a linear combination of components
    matrix_lsa = pd.DataFrame(lsa_fit, index=matrix.index, columns=word_weights.index)

    word_weights = word_weights_trans.sort_values(by=[0], ascending=False)

    LSA_tuple = collections.namedtuple("LSA_tuple", ["matrix", "word_weights"])
    new = LSA_tuple(matrix_lsa, word_weights)

    return new


def create_corpus_from_series(series: pd.Series):
    text = ""
    for row in series:
        text = text + row
    return text


def remove_tags(text: str, regex_str: str):
    text = re.sub(regex_str, " ", text)
    return text


# %%


def what_words_matter(doc_term_matrix: pd.DataFrame, row1, row2, show_num: int = 5):
    """Given a two vectors in a doc-term matrix, show words /
    that discriminate between the documents.

    Args:
        doc_term_matrix (pd.DataFrame): DF with terms in columns, freq in rows
        row1 ([type]): index of first doc
        row2 ([type]): index of other doc
        show_num (int): number of words to show
    """

    new_df = doc_term_matrix.loc[[row1, row2]]

    # divide by total word count
    new_df["total"] = new_df.sum(axis=1)
    totals = list(new_df.total)
    print(totals)

    new_df = new_df.div(new_df.total, axis=0).drop(columns=["total"])

    new_df = new_df.T.reset_index()
    new_df["diff"] = new_df[row1] - new_df[row2]
    new_df["abs"] = new_df["diff"].abs()

    new_df = new_df[(new_df[row1] != 0) | (new_df[row2] != 0)]

    new_df["row1_p"] = new_df[row1].round(2)
    new_df["row2_p"] = new_df[row2].round(2)

    new_df["row1"] = new_df[row1] * totals[0]
    new_df["row2"] = new_df[row2] * totals[1]

    row1_df = new_df.sort_values(by="diff").tail(show_num)
    row1_df["type"] = "row1_distinct"

    row2_df = new_df.sort_values(by="diff").head(show_num)
    row2_df["type"] = "row2_distinct"

    sim_df = new_df.sort_values(by="abs").head(show_num)
    sim_df["type"] = "shared"

    words = (
        row1_df.append(sim_df)
        .append(row2_df)
        .set_index(["type", "index"])[["row1", "row2", "row1_p", "row2_p"]]
    )

    return words


def top_terms(doc_term_matrix: pd.DataFrame, row, show_num: int = 5):
    words = list(doc_term_matrix.columns)
    frequencies = list(doc_term_matrix.loc[row])

    new_df = pd.DataFrame(list(zip(words, frequencies)), columns=["words", "frequency"])
    new_df = new_df.reset_index()
    new_df = new_df.sort_values(by=["frequency"], ascending=False)

    return new_df.head(show_num)


def doc_matrix_with_embeddings(df: pd.DataFrame, text_col: str):
    df_index = df[[]].reset_index()
    list_arrays = [ave_word_embedding_for_doc(text) for text in df[text_col]]
    matrix = pd.concat([df_index, pd.DataFrame(list_arrays)], axis=1)
    return matrix


def weighted_doc_matrix_with_embeddings(tfidf_matrix: pd.DataFrame):
    df_index = tfidf_matrix[[]].reset_index()
    list_arrays = [
        weighted_ave_word_embedding_for_doc(tfidf_matrix=tfidf_matrix, row=row)
        for row in tfidf_matrix.index
    ]
    matrix = pd.concat([df_index, pd.DataFrame(list_arrays)], axis=1)
    return matrix


def weighted_ave_word_embedding_for_doc(tfidf_matrix: pd.DataFrame, row):
    sum_vector = np.repeat(0, 300)
    for col, weight in zip(tfidf_matrix.columns, tfidf_matrix.loc[row]):
        sum_vector = sum_vector + (tfidf_matrix.loc[row][col] * nlp(col).vector)
    return sum_vector


def ave_word_embedding_for_doc(text: str):
    sum_vector = np.repeat(0, 300)
    for token in nlp(text):
        sum_vector = sum_vector + token.vector

    return sum_vector
