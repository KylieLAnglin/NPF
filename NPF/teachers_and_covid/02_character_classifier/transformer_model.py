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
BATCH_SIZE = 64  # 0.54

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

annotations["tweet_training"] = np.where(annotations.split == "training", 1, 0)
annotations["tweet_testing"] = np.where(annotations.split == "testing", 1, 0)

train_data = annotations[annotations.tweet_training == 1]
eval_data = annotations[annotations.tweet_testing == 1]

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)


# %% Fine tune the model on the training dataset
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


train_texts = train_data["text"].tolist()
train_labels = train_data["label"].tolist()
train_inputs = encode_text(train_texts)
train_labels = tf.convert_to_tensor(train_labels)

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
# %%
eval_texts = eval_data["text"].tolist()
eval_labels = eval_data["label"].tolist()
eval_inputs = encode_text(eval_texts)
eval_labels = tf.convert_to_tensor(eval_labels)

eval_loss, eval_accuracy = model.evaluate(eval_inputs, eval_labels)
print(f"Eval Loss: {eval_loss}, Eval Accuracy: {eval_accuracy}")


# %%
predictions = model.predict(eval_inputs)
predicted_labels = np.argmax(predictions.logits, axis=1)

print(classification_report(eval_labels, predicted_labels))
# %%
# Save the model architecture as JSON
model_architecture = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_architecture)

# Save the model weights
model.save_weights(start.CLEAN_DIR + "model_weights.h5")


# Load the model architecture from JSON
with open("model_architecture.json", "r") as json_file:
    model_architecture = json_file.read()

# Load the model weights
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.config.num_labels = NUM_LABELS
model.classifier = tf.keras.layers.Dense(NUM_LABELS, activation="softmax")
model.load_weights(start.CLEAN_DIR + "model_weights.h5")

# %%
predictions = model.predict(eval_inputs)
print(classification_report(eval_labels, predicted_labels))

# %%
