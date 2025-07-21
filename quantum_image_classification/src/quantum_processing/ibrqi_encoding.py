# src/quantum_processing/ibrqi.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler

class ImprovedBRQI:
    def __init__(self, n_components=4, reps=2):
        self.n_components = n_components
        self.reps = reps
        self.sampler = Sampler()
        
        # Quantum feature map and ansatz
        self.feature_map = ZZFeatureMap(n_components, reps=1)
        self.ansatz = RealAmplitudes(n_components, reps=reps)
        
        # Observable for measurement
        self.observable = SparsePauliOp("Z" * n_components)
        
    def encode(self, image):
        """Improved BRQI encoding with dimension reduction"""
        # Downsample and preprocess
        img = self._preprocess(image)
        
        # Create parameterized circuit
        circuit = self.feature_map.compose(self.ansatz)
        circuit.measure_all()
        
        return circuit.assign_parameters(img)
    
    def _preprocess(self, image):
        """Downsample and normalize image"""
        # Reshape to 28x28 if needed
        img = image.reshape(28, 28)
        
        # Downsample to 14x14
        downsampled = img[::2, ::2]
        
        # Flatten and normalize
        return (downsampled.flatten() / 255.0)[:self.n_components]
    
    def extract_features(self, circuit):
        """Extract features using expectation values"""
        # Calculate expectation value
        result = self.sampler.run(circuit, self.observable).result()
        return np.array([result.values[0]])
