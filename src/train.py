import os
import sys
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_prepare_data


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    metrics = {
        'Model':     model_name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1 Score':  round(f1_score(y_test, y_pred), 4),
    }
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {metrics['Accuracy']}")
    print(f"  Precision: {metrics['Precision']}")
    print(f"  Recall   : {metrics['Recall']}")
    print(f"  F1 Score : {metrics['F1 Score']}")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    return metrics


def train():
    # Change working directory to project root so paths work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    fake_path = os.path.join('data', 'Fake.csv')
    true_path = os.path.join('data', 'True.csv')

    df = load_and_prepare_data(fake_path, true_path)

    X = df['cleaned']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # Check label distribution — important for debugging
    print(f"[INFO] Train label counts:\n{y_train.value_counts()}")

    print("\n[INFO] Fitting TF-IDF...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    print("[INFO] Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_model.fit(X_train_tfidf, y_train)

    print("[INFO] Training Linear SVM...")
    svm_model = LinearSVC(max_iter=2000, random_state=42, C=1.0)
    svm_model.fit(X_train_tfidf, y_train)

    lr_metrics  = evaluate_model(lr_model,  X_test_tfidf, y_test, "Logistic Regression")
    svm_metrics = evaluate_model(svm_model, X_test_tfidf, y_test, "Linear SVM")

    print("\n" + "="*55)
    print("  COMPARISON")
    print("="*55)
    results_df = pd.DataFrame([lr_metrics, svm_metrics]).set_index('Model')
    print(results_df.to_string())

    # Save best model by F1
    if lr_metrics['F1 Score'] >= svm_metrics['F1 Score']:
        best_model, best_name = lr_model, "Logistic Regression"
    else:
        best_model, best_name = svm_model, "Linear SVM"

    print(f"\n[INFO] Best model: {best_name}")

    os.makedirs('models', exist_ok=True)

    with open(os.path.join('models', 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    with open(os.path.join('models', 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join('models', 'logistic_regression.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
    with open(os.path.join('models', 'linear_svm.pkl'), 'wb') as f:
        pickle.dump(svm_model, f)

    print("[INFO] All models saved to models/")
    print("\n✅ Training complete!")


if __name__ == '__main__':
    train()
