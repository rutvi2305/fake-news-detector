from flask import Flask, request, jsonify, render_template
# NEW - replace with this
from fake_news_detection import (
    FakeNewsDetector,
    TextPreprocessor,
    TFIDFVectorizer,
    LogisticRegressionScratch,
    NaiveBayesScratch,
    RandomForestScratch,
    VotingEnsemble
)
import pickle
import os

app = Flask(__name__)

# Load the trained model
print("Loading trained model...")
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_news_model.pkl')

try:
    with open(model_path, 'rb') as f:
        detector = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("ERROR: fake_news_model.pkl not found!")
    print("Please run fake_news_detection.py first to train and save the model.")
    detector = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if detector is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    data = request.get_json()
    text = data.get('text', '').strip()

    if not text or len(text) < 20:
        return jsonify({'error': 'Please enter at least 20 characters.'}), 400

    result = detector.predict_single(text)

    return jsonify({
        'label':      result['label'],
        'confidence': round(result['confidence'] * 100, 1),
        'real_prob':  round(result['real_prob'] * 100, 1),
        'fake_prob':  round(result['fake_prob'] * 100, 1),
    })

if __name__ == '__main__':
    app.run(debug=False)