# %%
from __future__ import annotations
import pandas as pd
import numpy as np

from NPF.teachers_and_covid import start
import matplotlib.pyplot as plt

# %%
annotations = pd.read_csv(start.CLEAN_DIR + "annotations_characters.csv")
annotations["gold_hero"] = np.where(annotations.category == 1, 1, 0)
annotations["gold_villain"] = np.where(annotations.category == 2, 1, 0)
annotations["gold_victim"] = np.where(annotations.category == 3, 1, 0)

# annotations = annotations[annotations.random_set == 3]
# %%
classifications = pd.read_csv(start.CLEAN_DIR + "tweets_final.csv")
# classifications = classifications[classifications.random_set == 3]

classifications["classifier_hero"] = np.where(classifications.spacy_hero > 0.5, 1, 0)
classifications["classifier_villain"] = np.where(
    classifications.spacy_villain > 0.5, 1, 0
)
classifications["classifier_victim"] = np.where(
    classifications.spacy_victim > 0.5, 1, 0
)

# %%
df = annotations.merge(
    classifications[
        ["unique_id", "classifier_hero", "classifier_villain", "classifier_victim"]
    ],
    left_on="unique_id",
    right_on="unique_id",
)
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
