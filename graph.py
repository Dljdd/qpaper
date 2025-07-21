from graphviz import Digraph

dot = Digraph(comment='Workflow Diagram')

# 1) Define all nodes
dot.node('A',  'Input: MNIST Images')
dot.node('B1', 'Downsampling\n(16×16 / 8×8)')
dot.node('B2', 'PCA / Autoencoder\nCompression')
dot.node('C',  'BRQI Encoding\n(Bit-Plane Quantum Encoding)')
dot.node('D1', 'QCNN\n(Quantum ConvNet)')
dot.node('D2', 'PQC\n(Hardware-Efficient Ansatz)')
dot.node('E1', 'Pauli-Basis\n(X, Y, Z)')
dot.node('E2', 'Classical Shadows\nTomography')
dot.node('F',  'Classical Classifier\n(SVM / RF)')

# 2) Add edges explicitly
dot.edge('A',  'B1')
dot.edge('A',  'B2')
dot.edge('B1', 'C')
dot.edge('B2', 'C')
dot.edge('C',  'D1')
dot.edge('C',  'D2')
dot.edge('D1', 'E1')
dot.edge('D2', 'E2')
dot.edge('E1', 'F')
dot.edge('E2', 'F')

# 3) Render to file (requires the Graphviz system package installed)
dot.render('workflow_diagram', format='png', cleanup=False)
