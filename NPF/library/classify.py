# %%
import functools
import pandas as pd
import numpy as np

import functools
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from zlib import crc32
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
    ax = sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues")
    ax.set(xlabel="Model Classification", ylabel="True Classification")
    plt.show


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


def print_statistics(
    classification,
    ground_truth,
    model_name: str,
    file_name="",
):

    accuracy = accuracy_score(y_true=ground_truth, y_pred=classification)
    precision = precision_score(y_true=ground_truth, y_pred=classification)
    recall = recall_score(y_true=ground_truth, y_pred=classification)
    f1 = f1_score(y_true=ground_truth, y_pred=classification)

    print(
        "Accuracy = ",
        str(round(accuracy, 2)),
    )

    print(
        "Precision = ",
        str(round(precision, 2)),
    )

    print(
        "Recall = ",
        str(round(recall, 2)),
    )

    print(
        "F1 = ",
        str(round(f1, 2)),
    )

    if file_name != "":
        file_object = open(file_name, "a")
        file_object.write(model_name)

        file_object.write("\n")
        file_object.write("Accuracy = " + str(round(accuracy, 2)))
        file_object.write("\n")

        file_object.write("Precision = " + str(round(precision, 2)))
        file_object.write("\n")

        file_object.write(
            "Recall = " + str(round(recall, 2)),
        )
        file_object.write("\n")

        file_object.write("F1 = " + str(round(f1, 2)))
        file_object.write("\n")
        file_object.write("\n")
        file_object.write("\n")

        file_object.close()


stop_words = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    # "during",
    # "before",
    # "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "https",
    "amp",
    "co",
    "th",
    "&",
]

character_words = [
    "teacher",
    "student",
    "learning",
    "school",
    "thank",
    "time",
    "during",
    "kid",
    "working",
    "amazing",
    "work",
    "best",
    "year",
    "staff",
    "parent",
    "teaching",
    "virtual",
    "online",
    "thanks",
    "distance",
    "union",
    "back",
    "go",
    "pandemic",
    "covid",
    "get",
    "back",
    "risk",
    "life",
    "need",
    "safe",
    "going",
    "health",
    "work",
    "care",
    "die",
    "open",
    "teach",
    "sick",
]


def return_statistics(ground_truth, scores, classification):
    """Calculate and return classifier performance statistics in dictionary

    Args:
        ground_truth (_type_): _description_
        scores (_type_): _description_
        classification (_type_): _description_
        model_name (str): _description_
    """

    accuracy = accuracy_score(y_true=ground_truth, y_pred=classification)
    precision = precision_score(y_true=ground_truth, y_pred=classification)
    recall = recall_score(y_true=ground_truth, y_pred=classification)
    f1 = f1_score(y_true=ground_truth, y_pred=classification)
    auc = roc_auc_score(ground_truth, scores)
    tn, fp, fn, tp = confusion_matrix(ground_truth, classification).ravel()
    specificity = tn / (tn + fp)

    performance = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "specificity": specificity,
    }

    return performance
