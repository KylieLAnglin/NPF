# %%
import os
import string

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from nltk.stem import PorterStemmer

from NPF.teachers_and_covid import start
from NPF.library import process_text


NO_BELOW = [10, 100, 500, 1000]
NO_ABOVE = [0.25, 0.5, 1.0]
N_COMPONENTS_TO_VIEW = 25
# %%
df = pd.read_csv(start.CLEAN_DIR + "tweets_relevant.csv")
df = df.rename(columns={"text": "tweet_text"})
df = df[["unique_id", "tweet_text"]]
df["unique_id"] = df.unique_id.astype(int)
df = df.set_index("unique_id")


# %% Make lower, remove punctuation, stem
def make_lower(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


stemmer = PorterStemmer()


def stem_string(text):
    token_list = []
    for token in text.split(" "):
        token_list.append(stemmer.stem(token))
    return " ".join(token_list)


df["tweet_text_clean"] = df.tweet_text.apply(make_lower)
df["tweet_text_clean"] = df.tweet_text_clean.apply(remove_punctuation)
df["tweet_text_clean"] = df.tweet_text_clean.apply(stem_string)

docs = [
    process_text.process_text_nltk(
        text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=False,
        string_or_list="list",
    )
    for text in df.tweet_text_clean
]

# remove https
docs_clean = []
for doc in docs:
    new_doc = []
    for term in doc:
        if not "https" in term:
            new_doc.append(term)
    docs_clean.append(" ".join(new_doc))

# %%
grid = []
for no_below in NO_BELOW:
    for no_above in NO_ABOVE:
        grid.append({"no_below": no_below, "no_above": no_above})
len(grid)


# %%

pbar = tqdm(total=len(grid))
for parameters in grid:
    model_name = (
        "lsa_no_below_"
        + str(parameters["no_below"])
        + "_no_above_"
        + str(parameters["no_above"])
    )
    newpath = start.RESULTS_DIR + "LSA/" + model_name + "/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    pbar.update(1)

    vec = CountVectorizer(max_df=parameters["no_above"], min_df=parameters["no_below"])
    X = vec.fit_transform(docs_clean)
    matrix = pd.DataFrame(
        X.toarray(), columns=vec.get_feature_names_out(), index=df.index
    )
    lsa_matrix, word_weights = process_text.create_lsa_dfs(
        matrix=matrix, n_components=N_COMPONENTS_TO_VIEW
    )

    tweets_mapping = df.merge(
        lsa_matrix, left_index=True, right_index=True, how="inner"
    )
    tweets_mapping.to_csv(start.RESULTS_DIR + "LSA/" + model_name + "/lsa_tweets.csv")

    word_weights.to_csv(start.RESULTS_DIR + "LSA/" + model_name + "/lsa_words.csv")

pbar.close()

# %%
