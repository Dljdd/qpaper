from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

def create_brqi_circuit(image):
    """Simplified BRQI circuit for 28x28 MNIST images"""
    # Input validation
    assert image.shape == (28, 28), "Image must be 28x28"
    
    # Quantum resources (5 position qubits + 1 color qubit)
    qr = QuantumRegister(6, 'q')  # 5 position + 1 color
    circuit = QuantumCircuit(qr)
    
    # Initialize position qubits (0-4) in superposition
    circuit.h(range(5))
    
    # Encode pixel values
    for i in range(28):
        for j in range(28):
            # Convert position to 5-bit string (pad to 5 bits)
            pos_bin = f"{i:05b}"[:3] + f"{j:05b}"[:2]
            
            # Apply X gates to create control condition
            controls = []
            for q, bit in enumerate(pos_bin):
                if bit == '0':
                    circuit.x(q)
                    controls.append(q)
            
            # Encode pixel intensity to color qubit (index 5)
            if image[i,j] > 0.5:  # Threshold for binary encoding
                circuit.cx(controls, 5)  # Multi-controlled NOT
            
            # Reset control qubits
            for q in controls:
                circuit.x(q)
    
    return circuit
