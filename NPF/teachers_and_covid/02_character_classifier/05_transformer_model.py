# %%
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from NPF.teachers_and_covid import start
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Hyperparameters
MODEL_NAME = "distilbert-base-uncased"  # alternative: "vinai/bertweet-base"
EPOCHS = 3
BATCH_SIZE = 32  # 0.61
# BATCH_SIZE = 64  # 0.54

LEARNING_RATE = 1e-5
NUM_LABELS = 4  # Don't change
SEED = 24  # Don't change
# %%
np.random.seed(SEED)
tf.random.set_seed(SEED)
# %%
annotations = pd.read_csv(
    start.CLEAN_DIR + "annotations_characters.csv", index_col="unique_id"
)
annotations["label"] = annotations.character_final.map(
    {"Other/None": 0, "Hero": 1, "Victim": 2, "Villain": 3}
)

training_df = annotations[annotations.split == "training"]
testing_df = annotations[annotations.split == "testing"]
validation_df = annotations[annotations.split == "validation"]
# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)
def encode_text(texts):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,  # Adjust as needed
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="tf",
        )
        input_ids.append(encoded_text["input_ids"])
        attention_masks.append(encoded_text["attention_mask"])
    return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0)

# %%
train_texts = training_df["text"].tolist()
train_inputs = encode_text(train_texts)

train_labels = training_df["label"].tolist()
train_labels = tf.convert_to_tensor(train_labels)

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
# %%
model.fit(
    train_inputs,
    train_labels,
    epochs=EPOCHS,  # Adjust as needed
    batch_size=BATCH_SIZE,  # Adjust as needed
)
# %% Testing
testing_texts = testing_df["text"].tolist()
testing_inputs = encode_text(testing_texts)

testing_labels = testing_df["label"].tolist()
testing_labels = tf.convert_to_tensor(testing_labels)

testing_predictions = model.predict(testing_inputs)
testing_predicted_labels = np.argmax(testing_predictions.logits, axis=1)
print(classification_report(testing_labels, testing_predicted_labels))

transformer_test_scores = pd.DataFrame(classification_report(testing_labels, testing_predicted_labels, output_dict=True)).reset_index().rename(columns={"index": "measure"})
transformer_test_scores["model"] = "transformer"
test_scores = transformer_test_scores[["model", "measure", "macro avg", "weighted avg", "1", "2", "3", "0", "accuracy"]]

test_scores = test_scores.rename(columns = {"0":"Other/None", "1": "Hero", "2": "Victim", "3": "Villain"})
test_scores.to_csv(start.RESULTS_DIR + "multinomial_transformer_testing_scores.csv")
# %% Validation
validation_texts = validation_df["text"].tolist()
validation_inputs = encode_text(validation_texts)

validation_labels = validation_df["label"].tolist()
validation_labels = tf.convert_to_tensor(validation_labels)

validation_predictions = model.predict(validation_inputs)
validation_predicted_labels = np.argmax(validation_predictions.logits, axis=1)
print(classification_report(validation_labels, validation_predicted_labels))

transformer_validation_scores = pd.DataFrame(classification_report(validation_labels, validation_predicted_labels, output_dict=True)).reset_index().rename(columns={"index": "measure"})
transformer_validation_scores["model"] = "transformer"

validation_scores = transformer_validation_scores[["model", "measure", "macro avg", "weighted avg", "1", "2", "3", "0", "accuracy"]]

validation_scores = validation_scores.rename(columns = {"0":"Other/None", "1": "Hero", "2": "Victim", "3": "Villain"})
validation_scores.to_csv(start.RESULTS_DIR + "multinomial_transformer_validationing_scores.csv")

# %% Apply model
all_tweets = pd.read_csv(start.CLEAN_DIR + "tweets_relevant.csv")

all_texts = all_tweets["text"].tolist()
all_inputs = encode_text(all_texts)

all_predictions = model.predict(all_inputs)
all_predicted_labels = np.argmax(all_predictions.logits, axis=1)

all_tweets["label"] = all_predicted_labels

all_tweets["character_classification"] = all_tweets.label.map(
    {0: "Other/None", 1:"Hero", 2:"Victim", 3:"Villain"}
)

all_tweets.to_csv(start.CLEAN_DIR + "tweets_relevant_labeled.csv")

# %%