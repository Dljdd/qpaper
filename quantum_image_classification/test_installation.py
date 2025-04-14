# test_installation.py
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Create circuit WITH MEASUREMENT
qc = QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits
qc.h(0)
qc.cx(0, 1)
qc.measure_all()  # <-- Critical addition

# Run simulation
simulator = AerSimulator()
job = simulator.run(qc, shots=1024)
result = job.result()
print(result.get_counts())
