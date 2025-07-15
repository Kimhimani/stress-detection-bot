import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_METAL_DEVICE_FORCE_SYNC"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_DEVICE_USE_SYNC_EXECUTION"] = "1"
os.environ["APPLE_DISABLE_METAL"] = "1"

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load and clean CSV data
data_file = pd.read_csv('final_corrected_first.csv', on_bad_lines='skip')

# Ensure all entries in 'Questions' and 'Answers' are strings
data_file['Questions'] = data_file['Questions'].fillna("").astype(str)
data_file['Answers'] = data_file['Answers'].fillna("").astype(str)

# Extract questions and answers
patterns = data_file['Questions'].tolist()
tags = data_file['Answers'].tolist()

# Initialize variables
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Preprocess data
for pattern, tag in zip(patterns, tags):
    w = nltk.word_tokenize(pattern)
    words.extend(w)
    documents.append((w, tag))
    if tag not in classes:
        classes.append(tag)

# Lemmatize and sort words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    bag = [1 if w in pattern_words else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  # ✅ Safe input_shape
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# ✅ Use Adam optimizer (safe on M1 with Metal)
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Show final accuracy
print(f"Final Training Accuracy: {hist.history['accuracy'][-1]:.4f}")

# Save model
model.save('model.h5')
print(" Model created and saved as 'model.h5'")
