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

# %%

terms = [col for col in df.columns if "liwc_" in col]
scaler = MinMaxScaler()
df[terms] = scaler.fit_transform(df[terms])


# %%
victim_coefs = []
victim_pvalues = []
for term in tqdm(terms):
    mod = smf.ols(formula="tweet_victim ~ " + term, data=df[df.tweet_training == 1])
    res = mod.fit()
    victim_coefs.append(res.params[1])
    victim_pvalues.append(res.pvalues[1])

victim_coef_df = pd.DataFrame([terms, victim_coefs, victim_pvalues]).T
victim_coef_df = victim_coef_df.rename(columns={0: "term", 1: "coef", 2: "pvalue"})
len(victim_coef_df[victim_coef_df.pvalue < 0.05])
# %%
villain_coefs = []
villain_pvalues = []
for term in tqdm(terms):
    mod = smf.ols(formula="tweet_villain ~ " + term, data=df[df.tweet_training == 1])
    res = mod.fit()
    villain_coefs.append(res.params[1])
    villain_pvalues.append(res.pvalues[1])

villain_coef_df = pd.DataFrame([terms, villain_coefs, villain_pvalues]).T
villain_coef_df = villain_coef_df.rename(columns={0: "term", 1: "coef", 2: "pvalue"})
len(villain_coef_df[villain_coef_df.pvalue < 0.05])

# %%
hero_coefs = []
hero_pvalues = []
for term in tqdm(terms):
    mod = smf.ols(formula="tweet_hero ~ " + term, data=df[df.tweet_training == 1])
    res = mod.fit()
    hero_coefs.append(res.params[1])
    hero_pvalues.append(res.pvalues[1])

hero_coef_df = pd.DataFrame([terms, hero_coefs, hero_pvalues]).T
hero_coef_df = hero_coef_df.rename(columns={0: "term", 1: "coef", 2: "pvalue"})
len(hero_coef_df[hero_coef_df.pvalue < 0.05])

# %%
significant_terms = []
for term in victim_coef_df[victim_coef_df.pvalue < 0.05].term:
    if term not in significant_terms:
        significant_terms.append(term)

for term in hero_coef_df[hero_coef_df.pvalue < 0.05].term:
    if term not in significant_terms:
        significant_terms.append(term)

for term in villain_coef_df[villain_coef_df.pvalue < 0.05].term:
    if term not in significant_terms:
        significant_terms.append(term)


# %%
terms = ["liwc_tone_pos", "liwc_tone_neg", "liwc_work", "liwc_health", "liwc_illness", "liwc_death", "liwc_fatigue"]
