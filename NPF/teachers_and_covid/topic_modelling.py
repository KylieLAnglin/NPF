# %%
import re
import pandas as pd
import numpy as np
from NPF.teachers_and_covid import start

import nltk
import gensim
import pprint

# %%
df = pd.read_csv(start.MAIN_DIR + "training_batch1_annotated.csv")
