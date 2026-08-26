"""Filtra synthetic_ground_truth.jsonl tenendo solo i record affidabili.

I criteri sono cambiati insieme al contratto di generazione: prima si scartava un record quando
la DICHIARAZIONE del generatore non reggeva alla verifica (simplification_verified,
idle_qubits_verified). Ora il generatore non dichiara nulla e l'etichetta e' misurata dalla
facade, quindi non c'e' piu' niente da smentire: restano solo i motivi per cui un circuito non
e' utilizzabile come dato: non compila, e' degenere, e' un duplicato.

E' CADUTO anche il criterio "forma fuori bersaglio" (theme_consistent). I lotti non chiedono
piu' una forma metrica -- chiedono una struttura di codice, e l'etichetta esce dalla misura --
quindi non esiste piu' un bersaglio da mancare. Nel primo giro di generazione quel criterio da
solo scartava 50 circuiti compilanti su 50 fuori dal lotto pulito: erano circuiti validi con
etichetta corretta, buttati perche' misuravano numeri diversi da quelli ordinati. Ora si tengono
e la copertura si legge a posteriori dalla distribuzione delle etichette misurate.
"""

import json
from pathlib import Path

# Stesso pattern di generate.py: ancora il percorso al file stesso, non alla working
# directory del processo (che PyCharm puo' impostare diversamente da dove ti aspetti).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "synthetic_ground_truth.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "synthetic_ground_truth_f.jsonl"


def discard_reason(record: dict) -> str | None:
    """Perche' il record e' stato scartato, o None se e' utilizzabile.

    is_trustworthy risponde si'/no; qui si dice QUALE controllo ha ceduto. La differenza conta
    al controllo a vista: un lotto che rende male per "duplicato" e' un problema di diversita' --
    prompt o temperatura -- mentre uno che rende male per "non compila" e' un problema di
    modello, e i due si correggono in modo opposto.
    """
    if not record.get("structural_check_passed"):
        if record.get("compile_error"):
            return "non compila"
        if record.get("assigned_circuits") not in (None, 1):
            return f"{record['assigned_circuits']} circuiti assegnati"
        return "scartato in verifica"
    if record.get("duplicate_of") is not None:
        return f"duplicato ({record['duplicate_of']})"
    if record.get("measured_shape") is None:
        return "mai misurato"
    return None


def is_trustworthy(record: dict) -> bool:
    """True se il record e' utilizzabile come dato del dataset sintetico."""
    return discard_reason(record) is None


def _format_row(record: dict, reason: str | None) -> str:
    """Una riga per circuito: le metriche gia' misurate da generate.py, non ricalcolate qui."""
    shape = record.get("measured_shape")
    if shape is None:
        measures = f"{'-':>4} {'-':>3} {'-':>4} {'-':>4}"
        label = "-"
    else:
        measures = f"{shape['l']:>4} {shape['c']:>3} {shape['lc_product']:>4} {shape['idq']:>4}"
        label = ", ".join(record.get("measured_smells") or []) or "pulito"
    return (
        f"{record['circuit_id'][:38]:38} {record.get('generation_batch', '?')[:16]:16} "
        f"{measures}  {label:26} {reason or 'OK'}"
    )


def _print_yield(per_batch: dict[str, dict[str, int]]) -> None:
    """Resa per lotto e motivo dei fallimenti: la sola domanda che conta dopo una generazione."""
    print("\n  -- resa per lotto")
    for batch in sorted(per_batch):
        counters = per_batch[batch]
        kept, total = counters["kept"], counters["total"]
        percent = 100 * kept / total if total else 0
        failures = ", ".join(
            f"{count}x {reason}"
            for reason, count in sorted(counters.items(), key=lambda kv: -kv[1])
            if reason not in ("kept", "total")
        )
        print(f"     {batch:18} {kept:>3}/{total:<3} ({percent:3.0f}%)  {failures}")


def _print_labels(by_label: dict[str, int], kept: int) -> None:
    """Composizione finale del dataset: e' cio' che il filtro deve preservare."""
    print("\n  -- composizione (record tenuti)")
    for label, count in sorted(by_label.items(), key=lambda kv: -kv[1]):
        percent = 100 * count / kept if kept else 0
        print(f"     {label:32} {count:>3}  ({percent:3.0f}%)")


if __name__ == "__main__":
    seen_ids: dict[str, int] = {}
    # Solo contatori: il file si scorre una riga alla volta e non se ne accumula mai il
    # contenuto (vincolo O(1) di CLAUDE.md).
    by_label: dict[str, int] = {}
    per_batch: dict[str, dict[str, int]] = {}
    kept = total = 0

    print(
        f"{'circuito':38} {'lotto':16} {'l':>4} {'c':>3} {'l*c':>4} {'IdQ':>4}  {'etichetta':26} esito"
    )

    with (
        INPUT_PATH.open(encoding="utf-8") as infile,
        OUTPUT_PATH.open("w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            total += 1
            record = json.loads(line)
            reason = discard_reason(record)
            print(_format_row(record, reason))

            batch = record.get("generation_batch", "?")
            counters = per_batch.setdefault(batch, {"kept": 0, "total": 0})
            counters["total"] += 1

            if reason is not None:
                counters[reason] = counters.get(reason, 0) + 1
                continue

            original_id = record["circuit_id"]
            seen_ids[original_id] = seen_ids.get(original_id, 0) + 1
            if seen_ids[original_id] > 1:
                record["circuit_id"] = f"{original_id}_dup{seen_ids[original_id]}"

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            counters["kept"] += 1
            label = ", ".join(record.get("measured_smells") or []) or "pulito"
            by_label[label] = by_label.get(label, 0) + 1

    print(f"\nTenuti {kept}/{total} record in {OUTPUT_PATH}")
    _print_yield(per_batch)
    _print_labels(by_label, kept)
