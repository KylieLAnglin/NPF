# %%
import functools
import pandas as pd
import numpy as np

import functools
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from zlib import crc32
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
import seaborn as sns
from pprint import pprint
from time import time
import matplotlib.pyplot as plt


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# %%
def accuracy_stats(df: pd.DataFrame, classification_col: str, ground_truth_col: str):
    df["accurate"] = np.where(df[classification_col] == df[ground_truth_col], 1, 0)
    accuracy = df.accurate.mean()
    print("Accuracy: " + str(round(accuracy, 2)))

    try:
        df["precise"] = np.where(df[classification_col] == 1, df.accurate, np.nan)
        precision = df.precise.mean()
        print("Precision: " + str(round(precision, 2)))

        df["recalled"] = np.where(df[ground_truth_col] == 1, df.accurate, np.nan)
        recall = df.recalled.mean()
        print("Recall: " + str(round(recall, 2)))

        return accuracy, precision, recall

    except:
        print("No positive classifications.")
        return accuracy




def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xFFFFFFFF < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def create_plot_confusion_matrix(cf_matrix):
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
    ]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues")


def grid_search(pipeline, parameters, X, y, n_iter=1000):
    search = RandomizedSearchCV(
        pipeline,
        parameters,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=10,
        n_iter=n_iter,
        random_state=83,
    )

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % search.best_score_)
    print("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return search


def full_grid_search(pipeline, parameters, X, y):
    search = GridSearchCV(
        pipeline,
        parameters,
        scoring="balanced_accuracy",
    )

    print("Performing fill grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % search.best_score_)
    print("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return search


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend()
    plt.xlabel("Threshold")

    plt.show()