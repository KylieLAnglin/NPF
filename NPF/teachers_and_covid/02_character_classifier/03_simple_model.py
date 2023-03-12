# %%
import pandas as pd
import numpy as np
from NPF.library import classify
from NPF.teachers_and_covid import start
from sklearn import svm
from sklearn.linear_model import SGDClassifier
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

# %%

tweets = pd.read_csv(start.MAIN_DIR + "data/temp/features.csv")

# %%
df = pd.read_csv(start.CLEAN_DIR + "annotations_characters.csv")
df = df.set_index("unique_id")

heros = df[df.tweet_hero == 1]
victims = df[df.tweet_victim == 1]
villains = df[df.tweet_villain == 1]
# %%
character_words = classify.character_words

for column in character_words:
    df[column] = np.where(df[column] > 0, 1, 0)

features = character_words + [
    "liwc_negate",
    "liwc_negemo",
    "liwc_posemo",
    "liwc_anx",
    "liwc_anger",
    "liwc_sad",
    "liwc_health",
    "liwc_power",
    "liwc_risk",
    "liwc_work",
    "liwc_money",
    "liwc_death",
    "liwc_swear",
    "polarity",
    "subjectivity",
    "tone_pos",
    "tone_neg",
]

# %%

matrix = df[features]
matrix = (matrix - matrix.mean()) / matrix.std()

df_std = df[["tweet_hero", "tweet_victim", "tweet_villain", "random_set"]].merge(
    matrix, left_index=True, right_index=True
)
# %%
outcome = "tweet_hero"

testing = df_std[df_std.random_set == 3]
training = df_std[df_std.random_set != 3]

testing_outcome = testing[outcome]
training_outcome = training[outcome]

# features = ["liwc_posemo", "tone_pos", "tone_neg", "liwc_death", "liwc_health", "liwc_work"]
training_matrix = training[features]
testing_matrix = testing[features]

# %% SVM
clf = svm.LinearSVC()
clf = clf.fit(training_matrix, training_outcome)


testing["classification_svm"] = clf.predict(testing_matrix)
testing["score_svm"] = clf.decision_function(testing_matrix)

training["classification_svm"] = clf.predict(training_matrix)
training["score_svm"] = clf.decision_function(training_matrix)

print("")
print("SVM Classifier")
classify.print_statistics(
    classification=testing.classification_svm,
    ground_truth=testing_outcome,
    model_name="SVM Classifier",
)

# %%
clf = SGDClassifier(random_state=87)
clf = clf.fit(training_matrix, training_outcome)

testing["classification_sgd"] = clf.predict(testing_matrix)
training["classification_sgd"] = clf.predict(training_matrix)

print("")
print("SGD Classifier")
classify.print_statistics(
    classification=testing.classification_sgd,
    ground_truth=testing_outcome,
    model_name="SGD Classifier",
)
roc_auc_score(testing_outcome, clf.decision_function(testing_matrix))


# %% Ridge
clf = LogisticRegression(penalty="l2")
clf = clf.fit(training_matrix, training_outcome)


testing["classification_ridge"] = clf.predict(testing_matrix)
training["classification_ridg"] = clf.predict(training_matrix)


print("")
print("Ridge Classifier with Threshold")
classify.print_statistics(
    classification=testing.classification_ridge,
    ground_truth=testing_outcome,
    model_name="Ridge Classifier with Threshold",
)
roc_auc_score(testing_outcome, clf.decision_function(testing_matrix))

# %%
