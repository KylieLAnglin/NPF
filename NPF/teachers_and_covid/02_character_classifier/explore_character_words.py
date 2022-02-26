# %%
import pandas as pd

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from NPF.teachers_and_covid import start
from NPF.library import process_text

# %%
df = pd.read_csv(start.MAIN_DIR + "annotations_characters.csv")
df = df.rename(
    columns={
        "hero": "tweet_hero",
        "villain": "tweet_villain",
        "victim": "tweet_victim",
        "category": "tweet_category",
    }
)
# %%

term_matrix = process_text.vectorize_text(
    df=df, text_col="text", remove_stopwords=True, lemma=True, min_df=10
)
matrix = df[["unique_id", "tweet_category"]].merge(
    term_matrix, left_index=True, right_index=True
)

# %%
phrases = process_text.vectorize_text(
    df=df,
    text_col="text",
    remove_stopwords=True,
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

matrix = matrix.merge(phrases, left_index=True, right_index=True)

# %%
grouped = matrix.groupby(["tweet_category"]).agg("sum")

# %%
words = grouped.T

words = words.sort_values(by=3.0, ascending=False)

# %%
