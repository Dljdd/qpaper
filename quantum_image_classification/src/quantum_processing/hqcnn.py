# src/quantum_processing/hqcnn.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap
from sklearn.decomposition import PCA

class HQCNNFeatureExtractor:
    def __init__(self, n_components=4):
        self.pca = PCA(n_components=n_components)
        self.feature_map = ZZFeatureMap(n_components, reps=1).decompose()
        
    def extract_features(self, images):
        processed = [self._preprocess(img) for img in images]
        reduced = self.pca.fit_transform(processed)
        
        features = []
        for vec in reduced:
            circuit = self._build_quantum_circuit(vec)
            counts = self._execute_circuit(circuit)
            features.append(self._process_counts(counts))
            
        return np.array(features)
    
    def _preprocess(self, image):
        img_2d = image.reshape(28, 28)
        downsampled = img_2d[::2, ::2]
        return (downsampled / 255.0).flatten()

    def _build_quantum_circuit(self, features):
        param_dict = {param: val for param, val in 
                     zip(self.feature_map.parameters, features)}
        return self.feature_map.assign_parameters(param_dict)

    def _execute_circuit(self, circuit):
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        transpiled = transpile(circuit, simulator)
        job = simulator.run(transpiled, shots=1024)
        return job.result().get_counts()

    def _process_counts(self, counts):
        return [counts.get(bin(i)[2:].zfill(4), 0) for i in range(16)]
