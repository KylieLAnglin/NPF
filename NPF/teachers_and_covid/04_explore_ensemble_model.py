# %%
from re import S
import pandas as pd
from pyparsing import col
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

from NPF.teachers_and_covid import start
from NPF.library import classify

# %%
training_spacy = pd.read_csv(start.MAIN_DIR + "training_spacy.csv")
testing_spacy = pd.read_csv(start.MAIN_DIR + "testing_spacy.csv")

training_models = pd.read_csv(start.MAIN_DIR + "training_models.csv")
testing_models = pd.read_csv(start.MAIN_DIR + "testing_models.csv")

# %%

# %%
training = training_models.merge(
    training_spacy[["unique_id", "classification"]].rename(
        columns={"classification": "score_spacy"}
    ),
    left_on="unique_id",
    right_on="unique_id",
)
cols = ["unique_id", "tweet_id", "random_set", "relevant", "category", "text"]
score_cols = [col for col in training if col.startswith("score")]

# score_cols = ["score_spacy", "score_ridge"]
cols = cols + score_cols

training = training[cols]

# %%
testing = testing_models.merge(
    testing_spacy[["unique_id", "classification"]].rename(
        columns={"classification": "score_spacy"}
    ),
    left_on="unique_id",
    right_on="unique_id",
)

training = training[cols]

# %%
clf = LogisticRegression()
clf = clf.fit(training[score_cols], training.relevant)

training["classification_ensemble"] = clf.predict(training[score_cols])
training["score_ensemble"] = [
    proba[1] for proba in clf.predict_proba(training[score_cols])
]

testing["classification_ensemble"] = clf.predict(testing[score_cols])
testing["score_ensemble"] = [
    proba[1] for proba in clf.predict_proba(testing[score_cols])
]

print("")
print("Ensemble")
classify.print_statistics(
    classification=testing.classification_ensemble,
    ground_truth=testing.relevant,
    model_name="Ensemble",
)


feature_importance = {name: coef for name, coef in zip(score_cols, clf.coef_[0])}
feature_importance_df = pd.DataFrame(
    list(feature_importance.items()), columns=["term", "importance"]
)
roc_auc_score(
    testing.relevant, [proba[1] for proba in clf.predict_proba(testing[score_cols])]
)
# %%
training["classification_rule"] = np.where(
    (training.score_spacy > 0.5)
    | (training.score_ridge > 0.5)
    | (training.score_svm > 0.5)
    | (training.score_sgd > 0.5)
    | (training.score_nb > 0.5)
    | (training.score_rf > 0.5),
    1,
    0,
)

testing["classification_rule"] = np.where(
    (testing.score_spacy > 0.5)
    | (testing.score_ridge > 0.5)
    | (testing.score_svm > 0.5)
    | (testing.score_sgd > 0.5)
    | (testing.score_nb > 0.5)
    | (testing.score_rf > 0.5),
    1,
    0,
)
classify.print_statistics(
    classification=testing.classification_rule,
    ground_truth=testing.relevant,
    model_name="Any positive",
)

cf_matrix = confusion_matrix(testing.relevant, testing.classification_rule)
classify.create_plot_confusion_matrix(cf_matrix=cf_matrix)


# %%
