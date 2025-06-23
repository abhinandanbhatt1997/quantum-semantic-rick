# benchmark.py

import time
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from qiskit import QuantumCircuit

# === Load Clean Dataset ===
ds = load_dataset("ucirvine/sms_spam", split="train")
df = pd.DataFrame(ds)

print("📄 Original Columns:", df.columns.tolist())
print("🔍 Raw label values:", df['label'].unique())

# Only map if needed
if df["label"].dtype == object:
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Safety check
if df["label"].isnull().any():
    raise ValueError("❌ Found NaN in labels after mapping.")

print("✅ Final unique labels:", df["label"].unique())
print("🧾 Label distribution:\n", df["label"].value_counts())

# === Vectorize ===
vectorizer = CountVectorizer(max_features=30, stop_words="english", binary=True)
X = vectorizer.fit_transform(df["sms"]).toarray()
y = df["label"].values

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sanity check
if len(np.unique(y_train)) < 2:
    raise ValueError("🚫 y_train contains only one class. Check dataset balance.")

# === Classical Model Benchmark ===
start = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
end = time.time()

print("\n🧠 Classical BoW + Logistic Regression")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("⏱️ Time taken: {:.4f} sec".format(end - start))

# === Quantum-Inspired Encoding ===
def build_quantum_circuit(bow_vector):
    qc = QuantumCircuit(len(bow_vector))
    for i, val in enumerate(bow_vector):
        angle = np.pi / 2 if val == 1 else 0
        qc.ry(angle, i)
    return qc

print("\n⚛️ Quantum Encoding Examples")
for i in range(3):
    sample_vec = X_test[i]
    qc = build_quantum_circuit(sample_vec)
    print(f"\n📧 Email #{i+1} - Label: {y_test[i]}")
    print("🧾 Vector:", sample_vec)
    print(qc.draw())
