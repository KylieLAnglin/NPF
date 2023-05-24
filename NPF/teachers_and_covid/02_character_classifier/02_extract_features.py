# %%
import pandas as pd

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from NPF.teachers_and_covid import start
from NPF.library import process_text

# %%
df = pd.read_csv(start.CLEAN_DIR + "tweets_relevant.csv")
df = df.rename(columns={"text": "tweet_text"})
df = df[["unique_id", "tweet_text"]]
df = df.set_index("unique_id")
df = df.sample(2000)
# %% Cleaned Text
df["tweet_text_clean"] = [
    process_text.process_text_nltk(
        text=text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=False,
        lemma=True,
        string_or_list="string",
    )
    for text in df.tweet_text
]


# %%
liwc = pd.read_csv(start.CLEAN_DIR + "liwc.csv")
liwc = liwc.add_prefix("liwc_")
liwc = liwc.rename(columns={"liwc_unique_id": "unique_id", "liwc_text": "tweet_text"})
liwc = liwc.drop("tweet_text", 1)
liwc = liwc.set_index("unique_id")

# %% Sentiment Analysis
df["polarity"] = [TextBlob(tweet).sentiment.polarity for tweet in df.tweet_text]
df["subjectivity"] = [TextBlob(tweet).sentiment.subjectivity for tweet in df.tweet_text]

analyzer = SentimentIntensityAnalyzer()
df["tone_pos"] = [analyzer.polarity_scores(tweet)["pos"] for tweet in df.tweet_text]
df["tone_neg"] = [analyzer.polarity_scores(tweet)["neg"] for tweet in df.tweet_text]
df["tone_neutral"] = [analyzer.polarity_scores(tweet)["neu"] for tweet in df.tweet_text]
df["tone_compound"] = [
    analyzer.polarity_scores(tweet)["compound"] for tweet in df.tweet_text
]


# %% Doc-Term Matrix
term_matrix = process_text.vectorize_text(
    df=df, text_col="tweet_text", remove_stopwords=True, lemma=True, min_df=10
)

# %% N-grams
phrases = process_text.vectorize_text(
    df=df,
    text_col="tweet_text",
    remove_stopwords=False,
    max_features=2000,
    n_gram_range=(2, 3),
)
columns = []
for phrase in phrases.columns:
    stop_phrase = False
    for stop in process_text.STOPLIST:
        if phrase.startswith(stop):
            stop_phrase = True
        if phrase.endswith(stop):
            stop_phrase = True
    if (not stop_phrase) and (phrase not in columns):
        columns.append(phrase)
phrases = phrases[columns]

term_matrix = term_matrix.merge(phrases, left_index=True, right_index=True)
term_matrix.to_csv(start.MAIN_DIR + "data/clean/matrix.csv", index=True)
# %%
lsa_matrix, word_weights = process_text.create_lsa_dfs(
    matrix=term_matrix, n_components=50
)
lsa_matrix = lsa_matrix.add_prefix("lsa_")


# %%
feature_df = df.merge(liwc, left_index=True, right_index=True, how="left")
feature_df = feature_df.merge(lsa_matrix, left_index=True, right_index=True)
feature_df = feature_df.merge(term_matrix, left_index=True, right_index=True)

# %%
feature_df.to_csv(start.MAIN_DIR + "data/clean/features.csv")

# %% Export annotations matrix
matrix = pd.read_csv(start.MAIN_DIR + "data/clean/matrix.csv", index_col="unique_id")
annotations = pd.read_csv(
    start.CLEAN_DIR + "annotations_characters.csv", index_col="unique_id"
)

df_annotations = annotations[["split", "text", "hero", "victim", "villain"]]
df_annotations = df_annotations.rename(
    columns={
        "split": "tweet_split",
        "text": "tweet_text",
        "hero": "tweet_hero",
        "villain": "tweet_villain",
        "victim": "tweet_victim",
    }
)
matrix_annotations = matrix[matrix.index.isin(annotations.index)]
matrix_annotations = matrix_annotations[matrix_annotations.tweet_split != "validation"]
matrix_annotations.to_csv(start.CLEAN_DIR + "matrix_annotations.csv", index=True)
# %%
features_annotations = feature_df[feature_df.index.isin(annotations.index)]
features_annotations = features_annotations[
    features_annotations.tweet_split != "validation"
]
features_annotations.to_csv(
    start.MAIN_DIR + "data/clean/features_annotations.csv", index=True
)

# %%
