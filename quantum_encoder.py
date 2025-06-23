# quantum_encoder.py

from qiskit import QuantumCircuit
import numpy as np

def encode_bow_vector(bow_vector):
    num_qubits = len(bow_vector)
    qc = QuantumCircuit(num_qubits)
    for i, bit in enumerate(bow_vector):
        angle = np.pi / 2 if bit == 1 else 0
        qc.ry(angle, i)
    return qc
