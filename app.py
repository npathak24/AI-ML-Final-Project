from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import pickle

app = Flask(__name__)

# Make sure to download the required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

stemmer = LancasterStemmer()
model = None  # Will be initialized during app startup
words = []
labels = []
data = {}

with open('intents.json') as f:
    data = json.load(f)

if model is None:
    tf.compat.v1.reset_default_graph()

# Load the preprocessed words and labels
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

net = tflearn.input_data(shape=[None, len(words)])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(labels), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        if se in words:
            bag[words.index(se)] = 1

    return np.array(bag)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    results = model.predict([bag_of_words(user_input, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            response = responses[0]
            return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
