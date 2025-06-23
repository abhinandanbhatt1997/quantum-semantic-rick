# quantum_batch_processor.py

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset
from collections import defaultdict

# === CONFIG ===
NUM_QUBITS = 8
NUM_EMAILS = 5000  # You can increase this!
SHOTS = 1024

# === Load Dataset ===
# === Load Dataset ===
print("📩 Loading dataset...")
ds = load_dataset("ucirvine/sms_spam", split="train")
df = pd.DataFrame(ds)

# ✅ FIX: Only map if needed
if df['label'].dtype == object or isinstance(df['label'].iloc[0], str):
    df['label'] = df['label'].map({"ham": 0, "spam": 1})
elif df['label'].dtype != int:
    print("⚠️ Unknown label format:", df['label'].unique())

# ✅ Show label distribution
print("✅ Final Label Distribution:\n", df['label'].value_counts())

# === Sentence Embedding ===
print("📡 Embedding sentences...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_and_scale(sentence):
    emb = model.encode(sentence)
    scaled = MinMaxScaler((0, np.pi/2)).fit_transform(emb.reshape(-1, 1)).flatten()
    return scaled[:NUM_QUBITS]

# === Build Circuit ===
def build_quantum_circuit(vec):
    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    for i, angle in enumerate(vec):
        qc.ry(angle, i)
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    return qc

# === Quantum Backend ===
backend = Aer.get_backend("qasm_simulator")

# === Process Batch ===
results = []

print("⚛️ Running quantum encoding and measurement...")
for i in range(NUM_EMAILS):
    row = df.iloc[i]
    text, label = row["sms"], row["label"]
    vec = embed_and_scale(text)
    qc = build_quantum_circuit(vec)

    job = backend.run(qc, shots=SHOTS)
    counts = job.result().get_counts()

    # Store top N outcomes
    top_counts = dict(sorted(counts.items(), key=lambda x: -x[1])[:5])
    results.append({
        "text": text,
        "label": label,
        "top_counts": top_counts,
        "most_likely": max(top_counts, key=top_counts.get)
    })

# === Export to CSV ===
print("💾 Saving quantum measurement results...")
output = pd.DataFrame(results)
output.to_csv("quantum_data_export.csv", index=False)

print("✅ Done! Saved to 'quantum_data_export.csv'")
