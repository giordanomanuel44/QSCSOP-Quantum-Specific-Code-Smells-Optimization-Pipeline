"""Filtra synthetic_ground_truth.jsonl tenendo solo i record verificati e affidabili."""
import json
from pathlib import Path

# Stesso pattern di generate.py: ancora il percorso al file stesso, non alla working
# directory del processo (che PyCharm puo' impostare diversamente da dove ti aspetti).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "synthetic_ground_truth.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "synthetic_ground_truth_f.jsonl"


def is_trustworthy(record: dict) -> bool:
    if not record.get("structural_check_passed"):
        return False
    if record.get("duplicate_of") is not None:
        return False
    if not record.get("theme_consistent", True):
        return False
    smells = record.get("intended_smells", [])
    if "long_circuit" in smells and record.get("simplification_verified") is not True:
        return False
    if "idle_qubits" in smells and record.get("idle_qubits_verified") is not True:
        return False
    return True

seen_ids: dict[str, int] = {}

with INPUT_PATH.open(encoding="utf-8") as infile, OUTPUT_PATH.open("w", encoding="utf-8") as outfile:
    kept, total = 0, 0
    for line in infile:
        total += 1
        record = json.loads(line)
        if is_trustworthy(record):
            original_id = record["circuit_id"]
            seen_ids[original_id] = seen_ids.get(original_id, 0) + 1
            if seen_ids[original_id] > 1:
                record["circuit_id"] = f"{original_id}_dup{seen_ids[original_id]}"
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

print(f"Tenuti {kept}/{total} record in {OUTPUT_PATH}")