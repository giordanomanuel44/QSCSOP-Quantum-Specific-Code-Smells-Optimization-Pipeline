"""
Script diagnostico: verifica quali cartelle di Bugs4Q-NA contengono realmente
codice Qiskit (cerca la stringa 'qiskit' nel testo dei file .py).

Non modifica nulla, serve solo a decidere quali fonti includere in
fetch_datasets.py.

Uso:
    python check_qiskit_sources.py
"""

import pathlib

root = pathlib.Path(r"C:\Users\giord\Desktop\DatasetTesi\Bugs4Q-NA")

for folder in sorted(root.iterdir()):
    if not folder.is_dir():
        continue
    py_files = list(folder.rglob("*.py"))
    if not py_files:
        continue
    qiskit_count = sum(
        1 for f in py_files
        if "qiskit" in f.read_text(errors="ignore").lower()
    )
    print(f"{folder.name}: {len(py_files)} file .py totali, {qiskit_count} contengono 'qiskit'")
