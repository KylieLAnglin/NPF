# %%
import pandas as pd

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from NPF.teachers_and_covid import start
from NPF.library import process_text

# %%
df = pd.read_csv(start.MAIN_DIR + "annotations_character.csv")

# %%

term_matrix = process_text.vectorize_text(
    df=df, text_col="text", remove_stopwords=True, lemma=True, min_df=10
)
matrix = df[["unique_id", "hero", "villain", "victim"]].merge(
    term_matrix, left_index=True, right_index=True
)

grouped = matrix.groupby(["unique_id", "hero", "villan", "victim"]).agg("mean")
