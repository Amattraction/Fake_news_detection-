import os
import sys
import pickle

_this_dir   = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)

if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

try:
    from preprocess import clean_text
except ImportError:
    from src.preprocess import clean_text


def load_model(model_path='models/best_model.pkl',
               vectorizer_path='models/tfidf_vectorizer.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_news(text, model=None, vectorizer=None):
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    cleaned = clean_text(text)
    print(f"[DEBUG] Cleaned text (first 80 chars): {cleaned[:80]}")

    features   = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    print(f"[DEBUG] Raw prediction value: {prediction}")

    confidence = None
    if hasattr(model, 'predict_proba'):
        proba      = model.predict_proba(features)[0]
        confidence = round(float(max(proba)) * 100, 2)
        print(f"[DEBUG] Probabilities — Fake: {proba[0]:.3f} | Real: {proba[1]:.3f}")

    label = 'REAL' if prediction == 1 else 'FAKE'
    return label, confidence


if __name__ == '__main__':
    sample_fake = (
        "SHOCKING: Government scientists confirmed 5G towers are used for mind control. "
        "Whistleblowers have come forward with proof. Share before deleted!"
    )
    sample_real = (
        "The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday, "
        "as policymakers continued efforts to bring inflation to the 2 percent target. "
        "The decision was unanimous among committee members."
    )

    print("\n--- Test ---")
    l, c = predict_news(sample_fake)
    print(f"Sample 1 | Expected FAKE → Got: {l} | Confidence: {c}%\n")

    l, c = predict_news(sample_real)
    print(f"Sample 2 | Expected REAL → Got: {l} | Confidence: {c}%")