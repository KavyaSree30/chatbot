from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle

app = Flask(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('chatbot_model.h5')

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Get input shape for padding
input_shape = len(words)

# Define the global graph and session
global graph
graph = tf.compat.v1.get_default_graph()
global sess
sess = tf.compat.v1.Session()

with graph.as_default():
    tf.compat.v1.keras.backend.set_session(sess)
    sess.run(tf.compat.v1.global_variables_initializer())

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                break
    return np.array(bag)

def predict_class(sentence, model, graph, sess):
    with graph.as_default():
        with sess.as_default():
            bow_data = bow(sentence, words)
            bow_data = np.pad(bow_data, (0, input_shape - len(bow_data)), 'constant')
            res = model.predict(np.array([bow_data]))[0]
            ERROR_THRESHOLD = 0.25
            results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
            return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return random.choice(intents_json['intents'][-1]['responses'])

def chatbot_response(text):
    try:
        ints = predict_class(text, model, graph, sess)
        if not ints:
            return get_response([{"intent": "default"}], intents)
        res = get_response(ints, intents)
        return res
    except Exception as e:
        print(f"Error generating chatbot response: {e}")
        return "Sorry, I'm having trouble processing your request right now."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chatbot_response_endpoint():
    user_text = request.args.get('msg')
    if user_text is None:
        return jsonify({"response": "Error: No message provided."}), 400
    try:
        response = chatbot_response(user_text)
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
