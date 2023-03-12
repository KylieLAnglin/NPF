# %%

import pandas as pd
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from NPF.library import process_dictionary
from NPF.teachers_and_covid import start


# %% Read
df = pd.read_csv(start.MAIN_DIR + "temp/creating_dictionary.csv")

# %% Process

df["text_clean"] = df.text.apply(lambda x: x.lower())
df["text_clean"] = df.text_clean.apply(process_dictionary.remove_punctuation)
df["text_clean"] = df.text_clean.apply(process_dictionary.remove_non_unicode)
df["text_clean"] = df.text_clean.apply(process_dictionary.stem_string)


# %% Create working dictionary
training = df[df.group == "training"]

training["key_words"] = training.annotation.str.split(",")

original_key_words = []
for sub_list in training[training.relevant == 1].key_words:
    for word in sub_list:
        if word not in original_key_words:
            original_key_words.append(word)

cleaned_key_words = []
for word in original_key_words:
    word = word.lower()
    word = process_dictionary.remove_punctuation(word)
    word = process_dictionary.remove_non_unicode(word)
    word = process_dictionary.stem_string(word)
    cleaned_key_words.append(word)
cleaned_key_words = list(set(cleaned_key_words))

# %%
for word in cleaned_key_words:
    df[word] = np.where(df.text_clean.str.contains(word), 1, 0)

df[df.training == 1].to_excel(start.MAIN_DIR + "temp/training_key_words.xlsx")


# %% Precision in testing data
word_precision_df = df[df.testing == 1]
for word in cleaned_key_words:
    word_precision_df[word] = np.where(word_precision_df[word] == 0, np.nan, word_precision_df.relevant)
precision_scores = pd.DataFrame(word_precision_df[cleaned_key_words].mean())
precision_scores = precision_scores.reset_index()
precision_scores = precision_scores.rename(columns={0:"word_precision", "index": "word"})
precision_scores = precision_scores.sort_values(by = "word_precision", ascending=False)
precision_scores
# %%
precision_scores.to_excel(start.RESULTS_DIR + "key_word_precision.xlsx", index=False)
# %% Keep words with precision about 0.5
reduced_key_words = list(precision_scores[precision_scores.word_precision > 0.5].index)

# %% Validate

testing = df[df.validation == 1]
testing['positive'] = testing[reduced_key_words].max(axis=1)

print("Accuracy", accuracy_score(testing.relevant, testing.positive))
print("Precision:", precision_score(testing.relevant, testing.positive))
print("Recall", recall_score(testing.relevant, testing.positive))
print("F1", f1_score(testing.relevant, testing.positive))


# %%