# main.py

from qiskit import QuantumCircuit
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Read and prepare email text
with open("example.txt", "r") as f:
    email_text = f.read()
print("📨 Email text:\n", email_text.strip())

# Load dataset and train vectorizer on full corpus
df = pd.read_csv("data/spam.csv")
vectorizer = CountVectorizer(max_features=30, stop_words="english", binary=True)
vectorizer.fit(df["text"])

# Vectorize the example email
X = vectorizer.transform([email_text]).toarray()[0]
print("\n📦 Email BoW Vector:", X.tolist())
print("🧠 Vocabulary:", vectorizer.get_feature_names_out().tolist())

# Convert to quantum circuit
def encode_bow_vector(vec):
    qc = QuantumCircuit(len(vec))
    for i, bit in enumerate(vec):
        if bit:
            qc.ry(np.pi / 2, i)
        else:
            qc.ry(0, i)
    return qc

qc = encode_bow_vector(X)
print("\n⚛️ Quantum Circuit:")
print(qc.draw())
