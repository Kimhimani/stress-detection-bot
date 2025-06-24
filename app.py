import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import random
from flask import Flask, render_template, request

# Load CSV data
data_file = pd.read_csv('final_corrected_first.csv', on_bad_lines='skip')



# Extract questions and answers
patterns = data_file['Questions'].tolist()
tags = data_file['Answers'].tolist()

# Load words, classes, and model
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))
model = load_model('model.h5')

# Define functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(ints):
    if not ints:  # If the intents list is empty
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    tag = ints[0]['intent']
    for pattern, response in zip(patterns, tags):
        if response == tag:
            return response
    return "I'm sorry, I didn't understand that. Could you please rephrase?"  # Fallback response


# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    print(f"User Input: {user_text}")  # Debug user input
    ints = predict_class(user_text, model)
    print(f"Predicted Intents: {ints}")  # Debug prediction
    res = get_response(ints)
    print(f"Response: {res}")  # Debug response
    return res


if __name__ == "__main__":
    app.run()
