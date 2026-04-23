# =============================================================
# app.py
# Flask web application for Fake News Detection
# Run with: python app.py
# Then open http://127.0.0.1:5000 in your browser
#
# IMPORTANT: Run "python src/train.py" FIRST to generate models!
# =============================================================

import os
import sys
from flask import Flask, request, render_template, jsonify

# Add src/ to path so we can import from src/predict.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from predict import load_model, predict_news

# Create Flask app
app = Flask(__name__)

# -----------------------------------------------------------------
# Load the saved model once when Flask starts (not on every request)
# This is more efficient — model stays in memory
# -----------------------------------------------------------------
print("[INFO] Loading model...")
try:
    # Paths are relative to where you run app.py (project root)
    model, vectorizer = load_model(
        model_path=os.path.join('models', 'best_model.pkl'),
        vectorizer_path=os.path.join('models', 'tfidf_vectorizer.pkl')
    )
    MODEL_LOADED = True
    print("[INFO] Model loaded successfully!")
except FileNotFoundError as e:
    print(f"[ERROR] Could not load model: {e}")
    print("[ERROR] Please run:  python src/train.py  first!")
    model, vectorizer = None, None
    MODEL_LOADED = False


@app.route('/')
def home():
    """Serve the main HTML page. Passes model status to template."""
    return render_template('index.html', model_loaded=MODEL_LOADED)


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST endpoint — accepts JSON body: {"text": "news article here"}
    Returns JSON: {"prediction": "FAKE"/"REAL", "confidence": 95.4}
    """
    # Check model is ready
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded. Please run: python src/train.py first!'
        }), 503   # 503 = Service Unavailable

    # ---- Read input ----
    if request.is_json:
        data = request.get_json()
        news_text = data.get('text', '').strip()
    else:
        news_text = request.form.get('news_text', '').strip()

    # ---- Validate ----
    if not news_text:
        return jsonify({'error': 'No text provided. Please enter a news article.'}), 400

    if len(news_text) < 20:
        return jsonify({'error': 'Text too short. Please enter a longer article.'}), 400

    # ---- Predict ----
    try:
        label, confidence = predict_news(news_text, model, vectorizer)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

    # ---- Return result ----
    print(f"[RESULT] Prediction={label} | Confidence={confidence}% | Text='{news_text[:60]}...'")
    return jsonify({
        'prediction': label,           # 'FAKE' or 'REAL'
        'confidence': confidence,      # float like 97.4, or None for SVM
        'text_preview': news_text[:200] + ('...' if len(news_text) > 200 else '')
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
