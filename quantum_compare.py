import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from sklearn.preprocessing import MinMaxScaler

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Define two test sentences ===
sentence_1 = "Congratulations! You've won a free prize. Click to claim now."  # SPAM
sentence_2 = "Let's schedule our team sync meeting for Monday morning."       # HAM

def embed_and_encode(text, num_qubits=8):
    embedding = model.encode(text)
    scaler = MinMaxScaler(feature_range=(0, np.pi / 2))
    scaled = scaler.fit_transform(embedding.reshape(-1, 1)).flatten()
    return scaled[:num_qubits]  # Reduce to number of qubits

def build_quantum_circuit(vec):
    qc = QuantumCircuit(len(vec), len(vec))
    for i, val in enumerate(vec):
        qc.ry(val, i)
    qc.measure(range(len(vec)), range(len(vec)))
    return qc

def simulate(qc):
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(qc, shots=1024)
    return job.result().get_counts()

# === Run both ===
vec1 = embed_and_encode(sentence_1)
vec2 = embed_and_encode(sentence_2)

qc1 = build_quantum_circuit(vec1)
qc2 = build_quantum_circuit(vec2)

counts1 = simulate(qc1)
counts2 = simulate(qc2)

# === Plot ===
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.bar(counts1.keys(), counts1.values(), color='tomato')
plt.title("SPAM Sentence\n" + sentence_1[:40] + "...")
plt.xlabel("Bitstring Outcome")
plt.ylabel("Frequency")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(counts2.keys(), counts2.values(), color='skyblue')
plt.title("HAM Sentence\n" + sentence_2[:40] + "...")
plt.xlabel("Bitstring Outcome")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
