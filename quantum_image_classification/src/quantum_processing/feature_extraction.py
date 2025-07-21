from qiskit_aer import AerSimulator
import numpy as np
from qiskit import QuantumCircuit, transpile
# src/quantum_processing/feature_extraction.py
def extract_features_from_circuit(circuit, shots=1024):
    """Handle empty measurement results"""
    simulator = AerSimulator()
    
    # Ensure circuit has measurements
    if not circuit.clbits:
        circuit.measure_all()
    
    # Transpile for backend compatibility
    transpiled = transpile(circuit, simulator)
    
    job = simulator.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Handle empty results
    if not counts:
        return np.zeros(32)  # Return zero features if no counts
    
    # Convert to feature vector
    features = []
    for i in range(2**6):
        key = bin(i)[2:].zfill(6)
        features.append(counts.get(key, 0)/shots)
    
    return np.array(features[:32])
