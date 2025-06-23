# quantum_measurement.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# === Load and Prepare Data ===
print("🚀 Loading dataset...")
ds = load_dataset("ucirvine/sms_spam", split="train")
df = pd.DataFrame(ds)

# Convert labels if needed
if df["label"].dtype == object:
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Vectorize the SMS text
print("🧠 Vectorizing text...")
vectorizer = CountVectorizer(max_features=6, stop_words='english', binary=True)
X = vectorizer.fit_transform(df["sms"]).toarray()
y = df["label"].values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# === Define Quantum Circuit ===
def build_quantum_circuit(vec):
    n_qubits = len(vec)
    qc = QuantumCircuit(n_qubits, n_qubits)
    for i, bit in enumerate(vec):
        angle = np.pi / 2 if bit == 1 else 0
        qc.ry(angle, i)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

# === Run Quantum Simulation ===
backend = Aer.get_backend("qasm_simulator")
num_samples = 10  # Feel free to increase

print("⚛️ Running quantum simulations...\n")
for i in range(num_samples):
    vec = X_test[i]
    label = y_test[i]

    qc = build_quantum_circuit(vec)
    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=1024)
    result = job.result()
    counts = result.get_counts()

    print(f"📩 Email #{i+1} - Label: {'SPAM' if label == 1 else 'HAM'}")
    print("🧃 BoW vector:", vec)
    print("📊 Quantum measurement counts:", counts)

    # Plot histogram
    plt.figure(figsize=(6, 3))
    plt.bar(counts.keys(), counts.values(), color="violet")
    plt.title(f"Quantum Measurement - Email #{i+1}")
    plt.xlabel("Bitstring Outcome")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
