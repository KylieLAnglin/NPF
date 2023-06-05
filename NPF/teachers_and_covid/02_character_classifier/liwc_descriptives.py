# %%
import pandas as pd


from NPF.teachers_and_covid import start  # Zach -- delete this

import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

import pandas as pd
import numpy as np
from NPF.library import classify
from NPF.teachers_and_covid import start
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
# %%
annotations = pd.read_csv(
    start.CLEAN_DIR + "annotations_characters.csv", index_col="unique_id"
)
annotations = annotations[
    (annotations.split == "training") | (annotations.split == "testing")
]
annotations["tweet_training"] = np.where(annotations.split == "training", 1, 0)
annotations["tweet_testing"] = np.where(annotations.split == "testing", 1, 0)

liwc = pd.read_csv(start.CLEAN_DIR + "liwc.csv")
liwc["unique_id"] = liwc.unique_id.astype(int)
liwc = liwc.set_index("unique_id")

# %%
df = annotations[
    ["tweet_training", "tweet_hero", "tweet_victim", "tweet_villain", "character_final"]
].merge(liwc, left_index=True, right_index=True)
df = df[df.tweet_training == 1]


# %%

# Perform one-way ANOVA

features = ["liwc_Tone", "liwc_tone_pos", "liwc_tone_neg", "liwc_emo_anger", "liwc_emo_anx", "liwc_prosocial", "liwc_work", "liwc_health", "liwc_illness", "liwc_death", "liwc_fatigue", "liwc_politic", "liwc_risk"]
hero_means = []
victim_means = []
villain_means = []
other_means = []
p_values = []
for feature in features:
    model = ols(feature + ' ~ character_final', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_value = anova_table['PR(>F)'][0].round(3)
    p_values.append(p_value)
    group_means = df.groupby('character_final')[feature].mean()
    hero_mean = group_means.loc["Hero"].round(2)
    hero_means.append(hero_mean)
    victim_mean = group_means.loc["Victim"].round(2)
    victim_means.append(victim_mean)
    villain_mean = group_means.loc["Villain"].round(2)
    villain_means.append(villain_mean)
    other_mean = group_means.loc["Other/None"].round(2)
    other_means.append(other_mean)

results = pd.DataFrame({"LIWC": features, "Hero": hero_means, "Victim": victim_means, "Villain": villain_means, "Other": other_means, "P Value": p_values} )
results.to_excel(start.RESULTS_DIR + "liwc_descriptives.xlsx")

# %%
