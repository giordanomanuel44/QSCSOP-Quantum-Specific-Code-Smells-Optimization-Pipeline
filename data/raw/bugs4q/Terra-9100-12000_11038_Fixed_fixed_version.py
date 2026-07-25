qc = QuantumCircuit(3)
qc.cx(0, 1)
qc.cx(1, 2)
trans_circ = transpile(qc, backend,
                       basis_gates=['rz', 'sx', 'cx'],
                       optimization_level=1)
trans_circ.draw('mpl')