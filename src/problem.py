"""Small 4-qubit VQC and synthetic classification dataset."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

SEED = 42
N_QUBITS = 4

def make_data():
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=3, n_redundant=0,
        n_repeated=0, n_classes=2, random_state=SEED
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train.astype(int), y_test.astype(int), scaler

def vqc_circuit(x, theta, measured=False):
    """4-qubit angle encoding + trainable RY/RZ layer + ring entanglement.

    theta has 8 parameters. Readout is qubit 0 in the Z basis.
    """
    qc = QuantumCircuit(N_QUBITS, 1 if measured else 0)
    for q in range(N_QUBITS):
        qc.ry(float(x[q]), q)
        qc.rz(float(x[q]) / 2.0, q)
    for q in range(N_QUBITS):
        qc.ry(float(theta[2*q]), q)
        qc.rz(float(theta[2*q+1]), q)
    for q in range(N_QUBITS - 1):
        qc.cx(q, q+1)
    qc.cx(N_QUBITS-1, 0)
    if measured:
        qc.measure(0, 0)
    return qc

def ideal_p1(x, theta):
    sv = Statevector.from_instruction(vqc_circuit(x, theta))
    # Qiskit indexes qubit 0 as the least-significant bit.
    probs = sv.probabilities()
    return sum(float(p) for idx, p in enumerate(probs) if idx & 1)
