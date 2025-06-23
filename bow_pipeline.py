# bow_pipeline.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def load_dataset(file_path="data/spam.csv"):
    df = pd.read_csv(file_path)
    if 'label' in df.columns and 'text' in df.columns:
        return df[['label', 'text']].dropna()
    elif 'sms' in df.columns and 'label' in df.columns:
        return df.rename(columns={"sms": "text"})[['label', 'text']].dropna()
    else:
        raise ValueError("Dataset must contain 'label' and 'text' or 'sms' columns.")

def get_vectorizer_and_data(df, max_features=30):
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english', binary=True)
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['label'].values
    feature_names = vectorizer.get_feature_names_out()
    return X, y, vectorizer, feature_names

def get_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Optional test run if called directly
if __name__ == "__main__":
    df = load_dataset()
    X, y, vectorizer, features = get_vectorizer_and_data(df)
    X_train, X_test, y_train, y_test = get_train_test(X, y)

    print("✅ Vectorizer features:", features)
    print("🧪 Sample vector:", X_train[0])
    print("🏷️ Label:", y_train[0])
