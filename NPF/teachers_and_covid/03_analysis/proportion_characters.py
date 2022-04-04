# %%
import pandas as pd
import numpy as np

from NPF.teachers_and_covid import start
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(start.CLEAN_DIR + "tweets_final.csv")

# %%
df["classifier_hero"] = np.where(df.spacy_hero > 0.5, 1, 0)
df["classifier_villain"] = np.where(df.spacy_villain > 0.5, 1, 0)
df["classifier_victim"] = np.where(df.spacy_victim > 0.5, 1, 0)

# %%
print("Hero Proportions")
print(df.classifier_hero.mean())

print("Villain Proportions")
print(df.classifier_villain.mean())


print("Victim Proportions")
print(df.classifier_victim.mean())

# %%
