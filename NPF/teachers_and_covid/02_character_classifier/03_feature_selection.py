# %%
import pandas as pd

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from NPF.teachers_and_covid import start
from NPF.library import process_text
import spacy
from tqdm import tqdm
from nltk.stem import PorterStemmer
import nltk
import numpy as np

nlp = spacy.load("en_core_web_sm")
import string
from nltk.tokenize import RegexpTokenizer
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

# %%
annotations = pd.read_csv(
    start.CLEAN_DIR + "annotations_characters.csv", index_col="unique_id"
)
annotations = annotations[
    (annotations.split == "training") | (annotations.split == "testing")
]
annotations["tweet_training"] = np.where(annotations.split == "training", 1, 0)
annotations["tweet_testing"] = np.where(annotations.split == "testing", 1, 0)
matrix = pd.read_csv(start.CLEAN_DIR + "matrix_annotations.csv", index_col="unique_id")

df = annotations[
    ["tweet_training", "tweet_hero", "tweet_victim", "tweet_villain"]
].merge(matrix, left_index=True, right_index=True)
# make binary
df[df > 1] = 1

terms = [col for col in df.columns if "term_" in col]

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

# %% Export

matrix_small = matrix[significant_terms]
matrix_small.to_csv(start.CLEAN_DIR + "matrix_annotations_small.csv")

full_doc_term_matrix = pd.read_csv(start.CLEAN_DIR + "matrix.csv")
full_doc_term_matrix_small = full_doc_term_matrix[significant_terms]
full_doc_term_matrix_small.to_csv(start.CLEAN_DIR + "matrix_small.csv")

# %%
training_df = df[df.tweet_training == 1]
testing_df = df[df.tweet_training == 0]

training_matrix = training_df[significant_terms]
testing_matrix = testing_df[significant_terms]

# %%


# %% Villain
lm = LogisticRegression()
lm.fit(training_matrix, training_df.tweet_victim)
lm_predictions = lm.predict(testing_matrix)
print(classification_report(testing_df.tweet_victim, lm_predictions))

# %%
grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf"],
}

svm_cv = GridSearchCV(
    estimator=SVC(), param_grid=grid, cv=5, verbose=1, scoring="balanced_accuracy"
)
svm_cv.fit(training_matrix, training_df.tweet_victim)
print(svm_cv.best_params_)
svm_predictions = svm_cv.predict(testing_matrix)
print(classification_report(testing_df.tweet_victim, svm_predictions))

# %%
grid = {
    "n_estimators": [10, 100, 1000],
    "max_depth": [5, 10, 50],
    "max_features": [None, "sqrt"],
    "random_state": [92],
}

rf_cv = GridSearchCV(
    estimator=RandomForestClassifier(), param_grid=grid, cv=5, verbose=2
)
rf_cv.fit(training_matrix, training_df.tweet_victim)
print(rf_cv.best_params_)
rf_predictions = rf_cv.predict(testing_matrix)
print(classification_report(testing_df.tweet_victim, rf_predictions))

# %%

model = SVC()
model.fit(training_matrix, training_df.tweet_victim)
svm_predictions = model.predict(testing_matrix)
print(classification_report(testing_df.tweet_victim, svm_predictions))

# %%
