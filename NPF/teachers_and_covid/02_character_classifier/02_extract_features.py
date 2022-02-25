# %%
import pandas as pd

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from NPF.teachers_and_covid import start
from NPF.library import process_text

# %%
df = pd.read_csv(start.MAIN_DIR + "tweets_classified.csv")
df = df[df.classification_rule == 1]
df = df.rename(
    columns={
        "text": "tweet_text",
        "created": "tweet_created",
        "likes": "tweet_likes",
        "retweets": "tweet_retweets",
        "quotes": "tweet_quotes",
        "replies": "tweet_replies",
        "geo": "tweet_geo",
    }
)
# %% Cleaned Text
df["text_clean"] = [
    process_text.process_text_nltk(
        text=text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=False,
        lemma=True,
        string_or_list="string",
    )
    for text in df.text
]

# %% Sentiment Analysis
# TODO: text sentiment analysis on cleaned text
df["polarity"] = [TextBlob(tweet).sentiment.polarity for tweet in df.text]
df["subjectivity"] = [TextBlob(tweet).sentiment.subjectivity for tweet in df.text]

analyzer = SentimentIntensityAnalyzer()
df["tone_pos"] = [analyzer.polarity_scores(tweet)["pos"] for tweet in df.text]
df["tone_neg"] = [analyzer.polarity_scores(tweet)["neg"] for tweet in df.text]
df["tone_neutral"] = [analyzer.polarity_scores(tweet)["neu"] for tweet in df.text]
df["tone_compound"] = [analyzer.polarity_scores(tweet)["compound"] for tweet in df.text]

# %% Doc-Term Matrix
term_matrix = process_text.vectorize_text(
    df=df, text_col="text", remove_stopwords=True, lemma=True, min_df=5
)
matrix = df[["unique_id"]].merge(term_matrix, left_index=True, right_index=True)

# %% N-grams
phrases = process_text.vectorize_text(
    df=df,
    text_col="text",
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

phrases = df[["unique_id"]].merge(phrases, left_index=True, right_index=True)
# %%

lsa_matrix, word_weights = process_text.create_lsa_dfs(
    matrix=term_matrix, n_components=50
)
lsa_matrix = df[["unique_id"]].merge(lsa_matrix, left_index=True, right_index=True)
# %%

feature_df = df.merge(lsa_matrix, left_on="unique_id", right_on="unique_id")
feature_df = feature_df.merge(matrix, left_on="unique_id", right_on="unique_id")
feature_df = feature_df.merge(phrases, left_on="unique_id", right_on="unique_id")
# %%
feature_df.to_csv(start.MAIN_DIR + "features.csv")

# %%
