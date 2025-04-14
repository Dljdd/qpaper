import numpy as np
from qiskit.quantum_info import Pauli, Statevector
from qiskit.quantum_info.states.measures import entropy

def measure_entanglement(circuit):
    state = Statevector.from_instruction(circuit)
    entropy_value = entropy(state)
    return entropy_value

def get_quantum_metrics(circuit):
    metrics = {
        'entanglement_entropy': measure_entanglement(circuit),
        'quantum_volume': 2**circuit.num_qubits
    }
    return metrics
