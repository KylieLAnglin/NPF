import pandas as pd
import numpy as np
from pprint import pprint
from time import time
import pickle


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)

from NPF.teachers_and_covid import start
from NPF.library import classify

# %%
testing = pd.read_csv(start.TEMP_DIR + "testing_spacy.csv")
testing = testing.rename(columns={"classification": "score"})

testing["classification"] = np.where(testing.score > 0.5, 1, 0)
classify.create_plot_confusion_matrix(
    cf_matrix=confusion_matrix(testing.relevant, testing.classification)
)
recall_score(testing.relevant, testing.classification)
precision_score(testing.relevant, testing.classification)
accuracy_score(testing.relevant, testing.classification)

# %%
testing["classification_recall"] = np.where(testing.score > 0.25, 1, 0)
classify.create_plot_confusion_matrix(
    cf_matrix=confusion_matrix(testing.relevant, testing.classification_recall)
)
recall_score(testing.relevant, testing.classification_recall)
precision_score(testing.relevant, testing.classification_recall)
accuracy_score(testing.relevant, testing.classification_recall)

# %%
