from qiskit import QuantumCircuit
from numpy import pi

qc = QuantumCircuit(1)






qc.x(0)

# ------------------------------------------------------------------------------

from qiskit import transpile

# Transpile
qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id'], optimization_level=0)

# Draw
