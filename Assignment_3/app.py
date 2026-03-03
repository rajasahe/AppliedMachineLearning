"""
app.py - Flask application for SMS Spam Classification.

Provides a /score endpoint that accepts a POST request with text input
and returns a JSON response with prediction and propensity score.
"""

import os
import pickle

from flask import Flask, request, jsonify
from sklearn.pipeline import Pipeline

from score import score

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'best_model')

with open(os.path.join(MODEL_DIR, 'linear_svc_model.pkl'), 'rb') as f:
    classifier = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

MODEL = Pipeline([
    ('tfidf', vectorizer),
    ('classifier', classifier)
])

THRESHOLD = 0.5  # Default threshold


@app.route('/score', methods=['POST'])
def score_endpoint():
    """
    Score a text input for spam classification.

    Expects a JSON POST body:
        { "text": "some SMS message" }

    Optionally accepts a threshold:
        { "text": "some SMS message", "threshold": 0.7 }

    Returns JSON:
        { "prediction": true/false, "propensity": 0.0-1.0 }
    """
    data = request.get_json()

    if data is None or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text = data['text']
    threshold = data.get('threshold', THRESHOLD)

    try:
        prediction, propensity = score(text, MODEL, threshold)
        return jsonify({
            'prediction': bool(prediction),
            'propensity': float(propensity)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
