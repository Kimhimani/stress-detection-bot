import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
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
    w = nltk.word_tokenize(pattern)  # Tokenize each question
    words.extend(w)
    documents.append((w, tag))
    if tag not in classes:
        classes.append(tag)

# Lemmatize and sort words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes to pickle files
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]

    # Create the bag of words array with 1 if word exists, 0 otherwise
    bag = [1 if w in pattern_words else 0 for w in words]

    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Convert to numpy array
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Define model structure
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Print final accuracy
print(f"Final Training Accuracy: {hist.history['accuracy'][-1]}")

# Save model
model.save('model.h5', hist)
print("Model created and saved as 'model.h5'")
