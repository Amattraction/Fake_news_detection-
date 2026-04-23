import re
import string
import nltk
import pandas as pd

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    # Remove Reuters-style location prefix: "WASHINGTON (Reuters) -"
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', str(text))
    # Also remove standalone city headers like "WASHINGTON -"
    text = re.sub(r'^[A-Z\s]+\s*-\s*', '', text)

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and word.isalpha()
    ]
    return ' '.join(cleaned_words)


def load_and_prepare_data(fake_path, true_path):
    print("[INFO] Loading datasets...")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'] + ' ' + df['text']
    elif 'text' in df.columns:
        df['content'] = df['text']
    else:
        raise ValueError("Dataset must have a 'text' column!")

    df = df.dropna(subset=['content'])

    print(f"[INFO] Total articles: {len(df)}")
    print(f"[INFO] Fake: {len(df[df['label']==0])} | Real: {len(df[df['label']==1])}")
    print("[INFO] Cleaning text...")

    df['cleaned'] = df['content'].apply(clean_text)

    print("[INFO] Done cleaning!")
    return df[['cleaned', 'label']]