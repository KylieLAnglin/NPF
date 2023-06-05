# %%

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score
import simpledorff

from NPF.teachers_and_covid import start

# %%

codes = pd.read_excel(
    start.ANNOTATIONS_DIR + "character_annotations_for_triple_coding_compare.xlsx"
)
codes = codes[["character_TF", "character_KLA"]].replace({"Other/None": "Other"})
codes["agree"] = np.where(codes.character_TF == codes.character_KLA, 1, 0)
codes.agree.mean()
cohen = cohen_kappa_score(codes.character_TF, codes.character_KLA)

# %%