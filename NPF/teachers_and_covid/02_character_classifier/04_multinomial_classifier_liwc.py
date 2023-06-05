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

matrix = pd.read_csv(
    start.CLEAN_DIR + "matrix_annotations_small.csv", index_col="unique_id"
)

# %%

df = annotations[["tweet_training", "character", "character_final"]].merge(
    liwc, how="left", left_index=True, right_index=True
)
df = df.merge(matrix, how="left", left_index=True, right_index=True)

# make binary
for col in df[[col for col in df.columns if "term_" in col]].columns:
    df[col] = np.where(df[col] >= 1, 1, df[col])
terms = [col for col in df.columns if "liwc_" in col or "term_" in col]

scaler = MinMaxScaler()
df[terms] = scaler.fit_transform(df[terms])
# %%
training_df = df[df.tweet_training == 1]
testing_df = df[df.tweet_training == 0]

matrix = df[terms]

training_matrix = training_df[terms]
testing_matrix = testing_df[terms]


rf = RandomForestClassifier(
    max_depth=50, max_features=None, n_estimators=1000, random_state=92
)
rf.fit(training_matrix, training_df.character_final)
rf_predictions = rf.predict(testing_matrix)
print(classification_report(testing_df.character_final, rf_predictions))

# %%
grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf"],
}

svm_cv = GridSearchCV(
    estimator=SVC(), param_grid=grid, cv=5, verbose=2, scoring="f1_macro"
)
svm_cv.fit(training_matrix, training_df.character_final)
print(svm_cv.best_params_)
svm_predictions = svm_cv.predict(testing_matrix)
print(classification_report(testing_df.character_final, svm_predictions))
svm_scores = pd.DataFrame(svm_cv.cv_results_)
svm_scores["model"] = "svm"
# %%
grid = {
    "n_estimators": [10, 100, 1000],
    "max_depth": [5, 10, 50],
    "max_features": [None, "sqrt"],
    "random_state": [92],
}

rf_cv = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=grid,
    cv=5,
    verbose=2,
    scoring="f1_macro",
)
rf_cv.fit(training_matrix, training_df.character_final)
print(rf_cv.best_params_)
rf_predictions = rf_cv.predict(testing_matrix)
print(classification_report(testing_df.character_final, rf_predictions))
rf_scores = pd.DataFrame(rf_cv.cv_results_)
rf_scores["model"] = "rf"


# %% Naive Bayes
grid = {
    "class_prior": [None, [0.25, 0.25, 25.0, 25]],
}

nb_cv = GridSearchCV(
    estimator=MultinomialNB(), param_grid=grid, cv=5, verbose=2, scoring="f1_macro"
)
nb_cv.fit(training_matrix, training_df.character_final)
print(nb_cv.best_params_)
nb_predictions = nb_cv.predict(testing_matrix)
print(classification_report(testing_df.character_final, nb_predictions))
nb_scores = pd.DataFrame(nb_cv.cv_results_)
nb_scores["model"] = "nb"
# %%

# %% CNN
grid = {
    "hidden_layer_sizes": [1, 100, 500],
}

cnn_cv = GridSearchCV(
    estimator=MLPClassifier(), param_grid=grid, cv=5, verbose=2, scoring="f1_macro"
)
cnn_cv.fit(training_matrix, training_df.character_final)
print(cnn_cv.best_params_)
cnn_predictions = cnn_cv.predict(testing_matrix)
print(classification_report(testing_df.character_final, cnn_predictions))
cnn_scores = pd.DataFrame(cnn_cv.cv_results_)
cnn_scores["model"] = "cnn"


# %% Ensemble
clf1 = SVC(**svm_cv.best_params_)
clf2 = RandomForestClassifier(**rf_cv.best_params_)
clf3 = MultinomialNB(**nb_cv.best_params_)
clf4 = MLPClassifier(**cnn_cv.best_params_)


grid = {
    "voting": ["hard", "soft"],
}

ensemble_cv = GridSearchCV(
    estimator=VotingClassifier(estimators=[("svm", clf1), ("rf", clf2), ("nb", clf3)]),
    param_grid=grid,
    cv=5,
    verbose=2,
    scoring="f1_macro",
)
ensemble_cv.fit(training_matrix, training_df.character_final)
print(ensemble_cv.best_params_)
ensemble_predictions = ensemble_cv.predict(testing_matrix)
print(classification_report(testing_df.character_final, ensemble_predictions))
ensemble_scores = pd.DataFrame(ensemble_cv.cv_results_)
ensemble_scores["model"] = "ensemble"
# %%
scores = pd.concat([rf_scores, svm_scores, nb_scores, cnn_scores, ensemble_scores])
scores.to_csv(start.RESULTS_DIR + "multinomial_scores_liwc.csv")


# %%
svm_test_scores = pd.DataFrame(classification_report(testing_df.character_final, svm_predictions, output_dict=True)).reset_index().rename(columns={"index": "measure"})
svm_test_scores["model"] = "svm"

rf_test_scores = pd.DataFrame(classification_report(testing_df.character_final, rf_predictions, output_dict=True)).reset_index().rename(columns={"index": "measure"})
rf_test_scores["model"] = "rf"

nb_test_scores = pd.DataFrame(classification_report(testing_df.character_final, nb_predictions, output_dict=True)).reset_index().rename(columns={"index": "measure"})
nb_test_scores["model"] = "nb"

cnn_test_scores = pd.DataFrame(classification_report(testing_df.character_final, cnn_predictions, output_dict=True)).reset_index().rename(columns={"index": "measure"})
cnn_test_scores["model"] = "cnn"

ensemble_test_scores = pd.DataFrame(classification_report(testing_df.character_final, ensemble_predictions, output_dict=True)).reset_index().rename(columns={"index": "measure"})
ensemble_test_scores["model"] = "ensemble"


test_scores = pd.concat([svm_test_scores, rf_test_scores, nb_test_scores, cnn_test_scores, ensemble_test_scores])

test_scores = test_scores[["model", "measure", "macro avg", "weighted avg", "Hero", "Victim", "Villain", "Other/None", "accuracy"]]

test_scores.to_csv(start.RESULTS_DIR + "multinomial_liwc_testing_scores.csv")

# %%
