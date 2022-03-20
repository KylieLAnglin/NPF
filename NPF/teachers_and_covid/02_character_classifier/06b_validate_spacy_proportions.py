# %%
from __future__ import annotations
import pandas as pd
import numpy as np

from NPF.teachers_and_covid import start
import matplotlib.pyplot as plt

# %%
testing = pd.read_csv(start.TEMP_DIR + "testing_characters_spacy.csv")
training = pd.read_csv(start.TEMP_DIR + "training_characters_spacy.csv")

df = pd.concat([training, testing])
df = df[df.category.isin([1, 2, 3, 4])]


df["gold_hero"] = np.where(df.category == 1, 1, 0)
df["gold_villain"] = np.where(df.category == 2, 1, 0)
df["gold_victim"] = np.where(df.category == 3, 1, 0)


df["classifier_hero"] = np.where(df.spacy_hero > 0.5, 1, 0)
df["classifier_villain"] = np.where(df.spacy_villain > 0.5, 1, 0)
df["classifier_victim"] = np.where(df.spacy_victim > 0.5, 1, 0)

# %%
print("Classifier vs Gold Standard Hero Proportions")
print(df.classifier_hero.mean())
print(df.gold_hero.mean())

print("Classifier vs Gold Standard Villain Proportions")
print(df.classifier_villain.mean())
print(df.gold_villain.mean())


print("Classifier vs Gold Standard Victim Proportions")
print(df.classifier_victim.mean())
print(df.gold_victim.mean())

# %%
