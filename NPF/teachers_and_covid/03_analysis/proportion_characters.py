# %%
import pandas as pd
import numpy as np

from NPF.teachers_and_covid import start
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(start.CLEAN_DIR + "annotations_characters.csv")

df["hero"] = np.where(df.character_final == "Hero", 1, 0)
df["victim"] = np.where(df.character_final == "Victim", 1, 0)
df["villain"] = np.where(df.character_final == "Villain", 1, 0)


# %%

# %%
print("Hero Proportions")
print(df.hero.mean())

print("Victim Proportions")
print(df.victim.mean())

print("Villain Proportions")
print(df.villain.mean())


# %%
