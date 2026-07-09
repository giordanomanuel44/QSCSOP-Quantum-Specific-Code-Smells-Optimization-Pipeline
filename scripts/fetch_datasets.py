"""
Script di preparazione dei dataset grezzi per QCEP.

Filtra le repository clonate localmente (TheSmellyEight, Bugs4Q-NA) e copia
esclusivamente i file .py Qiskit rilevanti dentro data/raw/, organizzati
per fonte.

Uso:
    1. Clona le due repo originali in una cartella separata, fuori dal progetto:
       git clone https://github.com/QBugs/qsmells-study-data.git
       git clone https://github.com/Z-928/Bugs4Q-NA.git

    2. Aggiorna le costanti SMELLY_EIGHT_SRC e BUGS4Q_SRC qui sotto con i
       percorsi locali corretti.

    3. Esegui questo script dalla root del progetto:
       python scripts/fetch_datasets.py
"""

import shutil
from pathlib import Path

# --- Configurazione: aggiorna questi percorsi in base a dove hai clonato le repo ---
SMELLY_EIGHT_SRC = Path(r"C:\Users\giord\Desktop\DatasetTesi\qsmells-study-data")
# Bugs4Q-NA: versione estesa di Bugs4Q ("New Additions"), superset del dataset originale
BUGS4Q_SRC = Path(r"C:\Users\giord\Desktop\DatasetTesi\Bugs4Q-NA")

# Root del progetto (data/raw/ relativo a questo file, che sta in scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# Smell in scope per questa tesi (Long Circuit, Idle Qubits)
TARGET_SMELLS = ["lc", "idq"]

# Nomi (case-insensitive, senza estensione) che identificano la versione
# "risolta" di un bug, indipendentemente dalla sotto-collezione del dataset.
# Scoperti analizzando l'intero dataset: alcune sotto-collezioni usano
# fixed.py, altre fix.py, altre fixed_version.py per lo stesso ruolo.
# Tutto ciò che non rientra in questo insieme (buggy.py, bug.py, bug_version.py,
# mod.py, modify.py, mod9.py, o casi anomali come times.py/times_shell.py)
# viene automaticamente ignorato, senza bisogno di eccezioni esplicite.
FIXED_STEMS = {"fixed", "fix", "fixed_version"}


def is_fixed_candidate(path: Path) -> bool:
    """True se il file rappresenta la versione corretta/risolta di un bug."""
    return path.stem.lower() in FIXED_STEMS


# Fonti Bugs4Q-NA che sono codice sorgente interno di Qiskit stesso (Aer, Terra):
# includiamo tutti i file "fixed-like" senza filtrare per la stringa "qiskit",
# perché un bugfix interno al framework non contiene necessariamente "import qiskit".
BUGS4Q_TRUSTED_SOURCES = [
    "Aer",
    "Terra-0-4000",
    "Terra-4001-6000",
    "Terra-6000-7100",
    "Terra-7100-9100",
    "Terra-9100-12000",
    "Terra-11000-15000",
]

# Fonti community (StackExchange/StackOverflow): possono riguardare framework
# diversi da Qiskit, quindi filtriamo file per file cercando "qiskit" nel testo.
BUGS4Q_MIXED_SOURCES = [
    "StackExchange",
    "StackExchange-page-1-25",
    "StackExchange_2",
    "stackoverflow-1-5",
    "stackoverflow-6-10",
]

# "Program" ha file sciolti senza distinzione buggy/fixed: per scelta
# metodologica li trattiamo come già "fixed" e li includiamo interamente.
BUGS4Q_PROGRAM_SOURCE = "Program"

# Escluse per definizione: Cirq (framework diverso), Q# (linguaggio diverso, non Python)


def fetch_thesmellyeight() -> None:
    dest = DATA_RAW / "thesmellyeight"
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for smell in TARGET_SMELLS:
        smell_dir = SMELLY_EIGHT_SRC / "samples" / smell
        if not smell_dir.exists():
            print(f"[WARN] Cartella non trovata: {smell_dir}")
            continue

        smell_dest = dest / smell
        smell_dest.mkdir(exist_ok=True)

        for py_file in smell_dir.glob("*.py"):
            shutil.copy2(py_file, smell_dest / py_file.name)
            copied += 1

    print(f"TheSmellyEight: copiati {copied} file .py in {dest}")


def fetch_bugs4q() -> None:
    dest = DATA_RAW / "bugs4q"
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0

    # 1. Fonti fidate (Aer, Terra-*): copia ogni file "fixed-like", nessun filtro testuale
    for source_name in BUGS4Q_TRUSTED_SOURCES:
        source_dir = BUGS4Q_SRC / source_name
        if not source_dir.exists():
            print(f"[WARN] Cartella non trovata: {source_dir}")
            continue

        for py_file in source_dir.rglob("*.py"):
            if not is_fixed_candidate(py_file):
                continue
            bug_id = py_file.parent.name
            new_name = f"{source_name}_{bug_id}_{py_file.name}"
            shutil.copy2(py_file, dest / new_name)
            copied += 1

    # 2. Fonti community (StackExchange/StackOverflow): copia solo i file
    #    "fixed-like" che contengono effettivamente "qiskit" nel testo
    for source_name in BUGS4Q_MIXED_SOURCES:
        source_dir = BUGS4Q_SRC / source_name
        if not source_dir.exists():
            print(f"[WARN] Cartella non trovata: {source_dir}")
            continue

        for py_file in source_dir.rglob("*.py"):
            if not is_fixed_candidate(py_file):
                continue
            if "qiskit" not in py_file.read_text(errors="ignore").lower():
                continue
            bug_id = py_file.parent.name
            safe_source = source_name.replace("-", "_")
            new_name = f"{safe_source}_{bug_id}_{py_file.name}"
            shutil.copy2(py_file, dest / new_name)
            copied += 1

    # 3. Program: file sciolti senza distinzione buggy/fixed, trattati come
    #    già "fixed" per scelta metodologica
    program_dir = BUGS4Q_SRC / BUGS4Q_PROGRAM_SOURCE
    if program_dir.exists():
        for py_file in program_dir.glob("*.py"):
            new_name = f"Program_{py_file.stem}_fixed.py"
            shutil.copy2(py_file, dest / new_name)
            copied += 1
    else:
        print(f"[WARN] Cartella non trovata: {program_dir}")

    print(f"Bugs4Q-NA: copiati {copied} file in {dest}")


if __name__ == "__main__":
    if not SMELLY_EIGHT_SRC.exists():
        raise SystemExit(f"Percorso non trovato: {SMELLY_EIGHT_SRC}")
    if not BUGS4Q_SRC.exists():
        raise SystemExit(f"Percorso non trovato: {BUGS4Q_SRC}")

    fetch_thesmellyeight()
    fetch_bugs4q()