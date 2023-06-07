# %%
import pandas as pd

from NPF.teachers_and_covid import start


# %%
df = pd.read_csv(start.CLEAN_DIR + "tweets_relevant.csv")

features = pd.read_csv(start.MAIN_DIR + "data/clean/features.csv")

# %%
