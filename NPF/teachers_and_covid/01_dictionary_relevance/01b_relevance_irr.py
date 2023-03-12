# %%

import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start
from sklearn.metrics import cohen_kappa_score
import simpledorff

# %%
df = pd.read_excel(start.ANNOTATIONS_DIR + "relevance_key_words.xlsx")

df = df.rename(columns={"Key words if relevant": "annotator1"})
df = df[["annotator1", "annotator2"]]
df = df.dropna()

# %%
df["annotator1"] = np.where(df.annotator1 == 0, 0, 1)
df = df.head(150)
# %%
df.loc[:, "agree"] = np.where(df.annotator1 == df.annotator2, 1, 0)

aggreement = df.agree.mean()
print(f"Agreement is {aggreement.round(2)}.")
# %%
df["either"] = np.where((df.annotator1 == 1) | (df.annotator2 == 1), 1, 0)
either = df.either.mean()
print(f"One of us selected yes {either.round(2)} percent of the time")

# %%
agreement_either = df[df.either == 1].agree.mean()
print(f"Of those we agreed {agreement_either.round(2)} percent of the time")

# %%
cohen = cohen_kappa_score(df.annotator1, df.annotator2)
print(f"Cohen's Kappa {cohen.round(2)} ")

# %%


# %%
