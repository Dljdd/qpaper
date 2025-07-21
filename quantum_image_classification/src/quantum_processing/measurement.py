from qiskit_aer import AerSimulator
import numpy as np

def measure_quantum_state(circuit, shots=1024):
    """Measure quantum state and return counts"""
    # Add measurement if not already present
    if not circuit.data or circuit.data[-1][0].name != 'measure':
        circuit.measure_all()
    
    # Execute the circuit on a simulator
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    
    return counts

def extract_features_from_measurement(counts, num_features=32):
    """Extract features from measurement results"""
    # Calculate total shots
    total_shots = sum(counts.values())
    
    # Convert counts to probabilities
    features = []
    for i in range(2**6):  # For 6 qubits (5 position + 1 color)
        key = format(i, '06b')
        features.append(counts.get(key, 0) / total_shots if total_shots > 0 else 0)
    
    # Return first num_features features
    return np.array(features[:num_features])
