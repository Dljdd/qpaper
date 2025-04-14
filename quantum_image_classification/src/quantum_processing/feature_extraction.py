from qiskit_aer import AerSimulator
import numpy as np

def extract_features(circuit, shots=1024):
    """Extract simplified features from quantum circuit"""
    simulator = AerSimulator()
    
    # Add measurement
    circuit.measure_all()
    
    # Execute simulation
    job = simulator.run(circuit, shots=shots)
    counts = job.result().get_counts()
    
    # Convert to 6-qubit feature vector (5 position + 1 color)
    return [counts.get(bin(i)[2:].zfill(6), 0) for i in range(64)]
