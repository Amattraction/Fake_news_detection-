# 📰 FakeCheck — Fake News Detection Web App

FakeCheck is a simple and interactive web application that detects whether a news article is **Real** or **Fake** using Machine Learning.

The project is built using **Flask (backend)** and **scikit-learn (ML model)**, with a clean frontend for user interaction.

---

## 🚀 Features

* Detects fake vs real news articles
* Displays prediction with confidence score
* Simple and user-friendly interface
* Works with real-time user input
* Includes sample test cases in UI

---

## 🧠 How It Works

1. User enters a news article in the web interface
2. Text is preprocessed (cleaning, normalization, etc.)
3. TF-IDF vectorizer converts text into numerical form
4. Trained ML model predicts:

   * **FAKE**
   * **REAL**
5. Result is displayed along with confidence score

---

## 🛠️ Tech Stack

* Python
* Flask
* Scikit-learn
* HTML, CSS, JavaScript

---

## 📂 Project Structure

```
fake-news-project/
│
├── app.py                  # Main Flask application
├── src/
│   ├── train.py            # Model training script
│   ├── predict.py          # Prediction logic
│   └── preprocess.py       # Text preprocessing
│
├── models/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── templates/
│   └── index.html          # Frontend UI
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Train the model (IMPORTANT)

```bash
python src/train.py
```

This will generate:

* `models/best_model.pkl`
* `models/tfidf_vectorizer.pkl`

---

### 3. Run the application

```bash
python app.py
```

Then open in browser:

```
http://127.0.0.1:5000
```

---

## ⚠️ Important Note About Dataset

The dataset files (**Fake.csv** and **True.csv**) are **NOT included in this repository**.

### Why?

* Large file size
* GitHub upload limits
* To keep repository lightweight

### What to do?

Download the dataset manually (ISOT dataset) and place it inside:

```
data/
├── Fake.csv
└── True.csv
```

Only after adding the dataset, run:

```bash
python src/train.py
```

---

## 🧪 Testing

You can test the model using:

```bash
python test.py
```

Or directly use the web interface.

---

## ⚡ API Endpoint

### POST `/predict`

**Request:**

```json
{
  "text": "Your news article here"
}
```

**Response:**

```json
{
  "prediction": "FAKE",
  "confidence": 95.4
}
```

---

## 📌 Notes

* If the model is not trained, the app will show a warning
* Minimum input length is required for prediction
* Confidence may be unavailable for some models (e.g., SVM)

---

## 👤 Author

**Kashish Jain**

---

## 💡 Future Improvements

* Better dataset handling
* Deep learning models (LSTM / BERT)
* Improved UI/UX
* News source verification

---

✨ This project is created for learning and academic purposes.
