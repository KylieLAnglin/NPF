# %%
import functools
import pandas as pd
import numpy as np


# @functools.singledispatch
# def predict_from_classifier(model, texts: list):
#     """Returns list of model confidence scores for whether the snippet
#         indicates the district has a racial equity policy, based on the
#         Bing search.

#     Args:
#         texts (list): Headlines concatenated with text

#     Returns:
#         list: list of model confidences
#     """
#     raise NotImplementedError


# @predict_from_classifier.register
# def predict_from_spacy(model: spacy.lang.en.English, label: str, texts: list):
#     """Returns list of model confidence scores for whether the snippet
#           indicates the district has a racial equity policy, based on the
#           Bing search.

#     Args:
#         model (spacy.lang.en.English): spacy classifier
#         label (str): text_cat label in spacy spacy.tokens.doc.Doc.cats
#         texts (list): [description]
#     """

#     p_cats = [model(text).cats["POSITIVE"] for text in tqdm.tqdm(texts)]
#     return p_cats


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


EQUITY_GRAMS = [
    "equity",
    "equitable",
    "equal",
    "diversity",
    "inclusion",
    "inclusive",
    "inclusivity",
    "equality",
    "racial",
    "race",
    "black lives matter",
    "blm",
    "harrassment",
    "harrass",
    "anti racism",
    "anti racist",
    "antiracist",
    "racism",
    "racist",
    "nondiscrimination",
    "discriminate",
    "discrimination",
    "discriminating",
    "disciminatory",
    "multi cultural",
    "multicultural",
    "equal opportunity",
    "civil rights",
    "inequitable",
    "achievement gap",
    "hate",
    "title ix",
    "solidarity",
    "justice",
    "ethnic",
    "black",
    "of color",
    "affirmative",
]


DISTRICT_DOCUMENT_GRAMS = [
    "school",
    "district",
    "board",
    "school district",
    "policy",
    "schools",
    "plan",
    "committee",
    "resolution",
    "school board",
    "board of ",
    "public",
    "public schools",
    "the district",
    "of education",
    "board of education",
    "the board",
    "high school",
    "township",
    "equity policy",
    "strategic",
    "strategic plan",
    "racism policy",
    "statement",
    "approved",
    "on equity",
    "policy and",
    "the board of",
    "commitment",
    "meeting",
    "district school board",
    "adopted",
    "equity committee",
    "board of directors'",
    "equity plan",
    "schooll committee",
    "commitment to",
    "unanimously",
]

DOCUMENT_GRAMS = [
    "policy",
    "policies",
    "plan",
    "committee",
    "resolution",
    "statement",
    "proclamation",
    "approved",
    "approve",
    "on equity",
    "commitment",
    "meeting",
    "adopt",
    "adopted",
    "equity committee",
    "board of directors'",
    "unanimously",
    "vote",
    "voted",
    "pass",
    "passed",
]
