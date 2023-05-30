# %%

# BEST IS THREE EPOCHS
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from NPF.teachers_and_covid import start
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

MODEL_NAME = "distilbert-base-uncased"
# MODEL_NAME = "vinai/bertweet-base"
EPOCHS = 4
NUM_LABELS = 4
# %%
seed = 24
np.random.seed(seed)
tf.random.set_seed(seed)
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


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode_text(texts):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,  # Adjust as needed
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="tf",
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
    return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0)


train_texts = train_data["text"].tolist()
train_labels = train_data["label"].tolist()
train_inputs = encode_text(train_texts)
train_labels = tf.convert_to_tensor(train_labels)
# %%

EPOCHS = 2

model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model.fit(
    train_inputs,
    train_labels,
    epochs=EPOCHS,  # Adjust as needed
    batch_size=32,  # Adjust as needed
)

eval_texts = eval_data["text"].tolist()
eval_labels = eval_data["label"].tolist()
eval_inputs = encode_text(eval_texts)
eval_labels = tf.convert_to_tensor(eval_labels)

eval_loss, eval_accuracy = model.evaluate(eval_inputs, eval_labels)
print(f"Eval Loss: {eval_loss}, Eval Accuracy: {eval_accuracy}")

predictions = model.predict(eval_inputs)
predicted_labels = np.argmax(predictions.logits, axis=1)
print(classification_report(eval_labels, predicted_labels))

# model_architecture = model.to_json()
# with open("model_architecture.json", "w") as json_file:
#     json_file.write(model_architecture)

# model.save_weights(start.CLEAN_DIR + "model_weights.h5")


# with open("model_architecture.json", "r") as json_file:
#     model_architecture = json_file.read()

# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# model.config.num_labels = NUM_LABELS
# model.classifier = tf.keras.layers.Dense(NUM_LABELS, activation="softmax")
# model.load_weights(start.CLEAN_DIR + "model_weights.h5")

predictions = model.predict(eval_inputs)
print(classification_report(eval_labels, predicted_labels))

# %%

EPOCHS = 3

model3 = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model3.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model3.fit(
    train_inputs,
    train_labels,
    epochs=EPOCHS,  # Adjust as needed
    batch_size=32,  # Adjust as needed
)

eval_texts = eval_data["text"].tolist()
eval_labels = eval_data["label"].tolist()
eval_inputs = encode_text(eval_texts)
eval_labels = tf.convert_to_tensor(eval_labels)

eval_loss, eval_accuracy = model3.evaluate(eval_inputs, eval_labels)
print(f"Eval Loss: {eval_loss}, Eval Accuracy: {eval_accuracy}")

predictions = model3.predict(eval_inputs)
predicted_labels = np.argmax(predictions.logits, axis=1)
print(classification_report(eval_labels, predicted_labels))

# model3_architecture = model3.to_json()
# with open("model3_architecture.json", "w") as json_file:
#     json_file.write(model3_architecture)

# model3.save_weights(start.CLEAN_DIR + "model3_weights.h5")


# with open("model3_architecture.json", "r") as json_file:
#     model3_architecture = json_file.read()

# model3 = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# model3.config.num_labels = NUM_LABELS
# model3.classifier = tf.keras.layers.Dense(NUM_LABELS, activation="softmax")
# model3.load_weights(start.CLEAN_DIR + "model3_weights.h5")

predictions = model3.predict(eval_inputs)
print(classification_report(eval_labels, predicted_labels))
# %%

# %%

EPOCHS = 4

model4 = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model4.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model4.fit(
    train_inputs,
    train_labels,
    epochs=EPOCHS,  # Adjust as needed
    batch_size=32,  # Adjust as needed
)

eval_texts = eval_data["text"].tolist()
eval_labels = eval_data["label"].tolist()
eval_inputs = encode_text(eval_texts)
eval_labels = tf.convert_to_tensor(eval_labels)

eval_loss, eval_accuracy = model4.evaluate(eval_inputs, eval_labels)
print(f"Eval Loss: {eval_loss}, Eval Accuracy: {eval_accuracy}")

predictions = model4.predict(eval_inputs)
predicted_labels = np.argmax(predictions.logits, axis=1)
print(classification_report(eval_labels, predicted_labels))

# model4_architecture = model4.to_json()
# with open("model4_architecture.json", "w") as json_file:
#     json_file.write(model4_architecture)

# model4.save_weights(start.CLEAN_DIR + "model4_weights.h5")


# with open("model4_architecture.json", "r") as json_file:
#     model4_architecture = json_file.read()

# model4 = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# model4.config.num_labels = NUM_LABELS
# model4.classifier = tf.keras.layers.Dense(NUM_LABELS, activation="softmax")
# model4.load_weights(start.CLEAN_DIR + "model4_weights.h5")

predictions = model4.predict(eval_inputs)
print(classification_report(eval_labels, predicted_labels))

# %%

EPOCHS = 5

model5 = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model5.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model5.fit(
    train_inputs,
    train_labels,
    epochs=EPOCHS,  # Adjust as needed
    batch_size=32,  # Adjust as needed
)

eval_texts = eval_data["text"].tolist()
eval_labels = eval_data["label"].tolist()
eval_inputs = encode_text(eval_texts)
eval_labels = tf.convert_to_tensor(eval_labels)

eval_loss, eval_accuracy = model5.evaluate(eval_inputs, eval_labels)
print(f"Eval Loss: {eval_loss}, Eval Accuracy: {eval_accuracy}")

predictions = model5.predict(eval_inputs)
predicted_labels = np.argmax(predictions.logits, axis=1)
print(classification_report(eval_labels, predicted_labels))

# model5_architecture = model5.to_json()
# with open("model5_architecture.json", "w") as json_file:
#     json_file.write(model5_architecture)

# model5.save_weights(start.CLEAN_DIR + "model5_weights.h5")


# with open("model5_architecture.json", "r") as json_file:
#     model5_architecture = json_file.read()

# model5 = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# model5.config.num_labels = NUM_LABELS
# model5.classifier = tf.keras.layers.Dense(NUM_LABELS, activation="softmax")
# model5.load_weights(start.CLEAN_DIR + "model5_weights.h5")

predictions = model5.predict(eval_inputs)
print(classification_report(eval_labels, predicted_labels))

# %%
