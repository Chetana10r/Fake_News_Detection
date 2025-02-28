#pip install flask

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model & vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    transformed_text = vectorizer.transform([data])
    prediction = model.predict(transformed_text)
    return jsonify({'prediction': 'Fake' if prediction[0] == 0 else 'Real'})

if __name__ == '__main__':
    app.run(debug=True)

import pickle
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
