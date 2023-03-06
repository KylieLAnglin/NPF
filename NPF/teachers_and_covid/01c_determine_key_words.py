import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start


# %%
df = pd.read_excel(start.ANNOTATIONS_DIR + "relevance_key_words.xlsx")

# %%
df = df.rename(columns={"Key words if relevant": "annotation"})

df = df[~df.annotation.isnull()]

len(df)
# %%
df["relevant"] = np.where(df.annotation == 0, 0, 1)
print(df.relevant.mean())

# %%
np.random.seed(570)

df["random_order"] = np.random.randint(1, 10000, size=len(df))

df = df.sort_values(by=["random_order"])

testing = df.head(100)

df = df.merge(
    testing[["unique_id"]],
    how="left",
    left_on="unique_id",
    right_on="unique_id",
    indicator="_merge",
)

df["testing"] = np.where(df._merge == "both", 1, 0)

testing = df[df.testing == 1]
training = df[df.training == 1]

# %%

# %%
