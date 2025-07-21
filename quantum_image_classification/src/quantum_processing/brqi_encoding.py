from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

def decimal_to_binary(decimal_value, num_bits=8):
    """Convert decimal value to binary representation"""
    return bin(int(decimal_value * 255))[2:].zfill(num_bits)

def create_brqi_state(image, num_bits=8):
    """Create BRQI encoding for a single image"""
    height, width = image.shape
    bitplanes = []
    
    # Create bitplanes
    for bit_pos in range(num_bits):
        bitplane = np.zeros((height, width), dtype=np.int8)
        
        for i in range(height):
            for j in range(width):
                pixel_value = image[i, j]
                binary = decimal_to_binary(pixel_value, num_bits)
                bitplane[i, j] = int(binary[-(bit_pos+1)])
        
        bitplanes.append(bitplane)
    
    return bitplanes

def create_brqi_circuit(image):
    """Create a quantum circuit for BRQI encoding"""
    # Ensure image is 32x32
    assert image.shape == (32, 32), "Image must be 32x32"
    
    # Create quantum registers (5 position qubits + 1 color qubit)
    qr = QuantumRegister(6, 'q')
    cr = ClassicalRegister(6, 'c')
    circuit = QuantumCircuit(6, 6) 
    
    # Initialize position qubits (0-4) in superposition
    circuit.h(range(5))
    
    # Encode pixel values
    for i in range(32):
        for j in range(32):
            # Convert position to 5-bit string (3 bits for i, 2 for j)
            i_bin = format(i, '03b')  # 3 bits for row (0-7)
            j_bin = format(j, '02b')  # 2 bits for column (0-3)
            pos_bin = i_bin + j_bin  # 5 bits total
            
            # Apply X gates to create control condition
            controls = []
            for q, bit in enumerate(pos_bin):
                if bit == '0' and q < 5:  # Ensure q is within range
                    circuit.x(q)
                    controls.append(q)
            
            # Encode pixel intensity to color qubit (index 5)
            if image[i, j] > 0.5:  # Threshold for binary encoding
                # Apply controlled-X from all position qubits to color qubit
                if len(controls) == 1:
                    circuit.cx(controls[0], 5)
                elif len(controls) == 2:
                    circuit.ccx(controls[0], controls[1], 5)
                elif len(controls) > 0:
                    # For more controls, use multiple gates
                    for control in controls:
                        circuit.cx(control, 5)
            
            # Reset control qubits
            for q in controls:
                circuit.x(q)
    circuit.measure(range(6), range(6))

    return circuit
