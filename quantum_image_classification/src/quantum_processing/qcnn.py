from qiskit import QuantumCircuit
import numpy as np

class QCNNFeatureExtractor:
    def __init__(self, n_qubits=8, n_layers=2):
        """Initialize QCNN feature extractor"""
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
    def extract_features(self, circuit, shots=1024):
        """Apply QCNN layers and extract features"""
        # Add convolutional layers
        for i in range(self.n_layers):
            circuit = self._add_conv_layer(circuit, i)
            circuit = self._add_pooling_layer(circuit, i)
        
        # Add final measurement
        circuit.measure_all()
        
        # Simulate and get counts
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=shots)
        counts = job.result().get_counts()
        
        # Extract features from different measurement bases
        features = self._extract_features_from_counts(counts)
        return features
        
    def _add_conv_layer(self, circuit, layer_idx):
        """Add quantum convolutional layer"""
        # Hardware-efficient ansatz for convolution
        n_qubits = circuit.num_qubits
        
        # Apply parameterized rotations
        for i in range(n_qubits):
            circuit.ry(np.pi/4, i)
            circuit.rz(np.pi/4, i)
        
        # Apply entangling gates (convolution kernel)
        for i in range(0, n_qubits-1, 2):
            circuit.cx(i, i+1)
            
        for i in range(1, n_qubits-1, 2):
            circuit.cx(i, i+1)
            
        # Add non-linearity
        for i in range(n_qubits):
            circuit.ry(np.pi/2, i)
            
        return circuit
        
    def _add_pooling_layer(self, circuit, layer_idx):
        """Add quantum pooling layer"""
        # Skip pooling for last layer to maintain sufficient qubits
        if layer_idx == self.n_layers - 1:
            return circuit
            
        n_qubits = circuit.num_qubits
        
        # Implement quantum pooling via controlled operations
        for i in range(0, n_qubits-1, 2):
            circuit.cx(i, i+1)
            circuit.ry(np.pi/2, i)
            
        return circuit
        
    def _extract_features_from_counts(self, counts):
        """Extract features from measurement counts using multiple bases"""
        # Calculate probabilities
        total_shots = sum(counts.values())
        probabilities = {k: v / total_shots for k, v in counts.items()}
        
        # Extract features based on bit patterns
        features = []
        
        # Feature set 1: Individual qubit probabilities
        for i in range(min(8, len(next(iter(counts.keys()))))):
            # Probability of qubit i being 1
            prob_1 = sum(v for k, v in probabilities.items() if k[i] == '1')
            features.append(prob_1)
            
        # Feature set 2: Parity features (simulating Z-basis measurements)
        for i in range(min(7, len(next(iter(counts.keys()))))):
            # Parity of adjacent qubits
            parity = sum(v for k, v in probabilities.items() 
                       if (int(k[i]) + int(k[i+1])) % 2 == 0)
            features.append(parity)
            
        # Feature set 3: Entropy-inspired features
        entropy = -sum(v * np.log(v) for v in probabilities.values() if v > 0)
        features.append(entropy)
        
        # Feature set 4: Hamming weight features
        for w in range(1, 5):
            # Probability of Hamming weight w
            prob_w = sum(v for k, v in probabilities.items() 
                       if k.count('1') == w)
            features.append(prob_w)
            
        return np.array(features)
