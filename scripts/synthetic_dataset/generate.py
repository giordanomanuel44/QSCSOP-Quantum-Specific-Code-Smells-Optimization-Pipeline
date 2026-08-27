"""
Entry point per la generazione di un dataset sintetico di circuiti Qiskit (smelly + puliti).

Tooling di preparazione dati, stesso ruolo di scripts/fetch_datasets.py -- NON e' un modulo
dell'architettura QCEP/QSCSOP/Analytics, vive fuori da src/qscsop_pipeline/.

Per ogni lotto in prompts.BATCH_THEMES: costruisce il prompt (prompts.build_batch_prompt),
chiama l'LLM per un GenerationBatch strutturato (generate_batch), verifica ogni circuito
strutturalmente (verification.verify_generated_circuit), scrive i circuiti non scartati in
data/raw/synthetic/ e un record di
metadati per OGNI circuito (scartati inclusi, per tracciabilita') in
data/interim/synthetic_ground_truth.jsonl.

Uso (dalla root del progetto, con il venv attivo e Ollama in esecuzione):
    python -m scripts.synthetic_dataset.generate
    python -m scripts.synthetic_dataset.generate --model "anthropic/claude-sonnet-5"

Non esegue nulla automaticamente all'import: la generazione vera parte solo da
`if __name__ == "__main__"`, e chiede conferma testuale prima di consumare chiamate LLM.
"""

import argparse
import json
import re
from pathlib import Path

from crewai import LLM
from pydantic import BaseModel

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from scripts.synthetic_dataset.prompts import (
    BATCH_THEMES,
    BatchTheme,
    GenerationBatch,
    build_batch_prompt,
)
from scripts.synthetic_dataset.verification import (
    is_near_duplicate,
    verify_generated_circuit,
)

# scripts/synthetic_dataset/generate.py -> risale alla root del progetto: un .parent in piu'
# di scripts/fetch_datasets.py, perche' questo file vive un livello piu' in profondita'.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CIRCUITS_DIR = PROJECT_ROOT / "data" / "raw" / "synthetic"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "interim" / "synthetic_ground_truth.jsonl"

# Modello di default: sostituibile a riga di comando (--model)
DEFAULT_MODEL = "ollama/qwen3-coder:30b"

_JSON_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_json_fence(text: str) -> str:
    """Rimuove un blocco di fence markdown (```json ... ``` / ``` ... ```) se presente.

    Fallback difensivo, non il meccanismo primario: response_model (vedi generate_batch) gia'
    produce JSON pulito o un'istanza Pydantic in ogni ramo osservato nel sorgente installato di
    crewai. Questo helper copre solo il caso limite in cui un provider ignori response_model e
    restituisca comunque testo con fence markdown attorno al JSON.
    """
    stripped = text.strip()
    match = _JSON_FENCE_PATTERN.match(stripped)
    return match.group(1) if match else stripped


def generate_batch(llm: LLM, prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Chiama llm.call() con response_model=schema e normalizza il risultato in un'istanza schema.

    crewai.LLM.call(response_model=...) ha comportamento diverso per provider (verificato
    leggendo crewai/llm.py e crewai/llms/providers/anthropic/completion.py, versione 1.15.2
    installata): se instradato su LiteLLM (es. "ollama/...", non tra i provider nativi) ritorna
    una stringa JSON; se instradato su una classe nativa (es. "anthropic/<modello-noto>") ritorna
    gia' l'istanza Pydantic parsata. crewai.agents.crew_agent_executor gestisce la stessa
    ambiguita' con un isinstance(answer, BaseModel) prima di validare la stringa -- stesso
    pattern replicato qui, cosi' generate_batch funziona identico con qualunque provider LiteLLM
    supporti, senza bisogno di sapere in anticipo quale ramo verra' preso.
    """
    result = llm.call(prompt, response_model=schema)
    if isinstance(result, schema):
        return result

    text = result if isinstance(result, str) else str(result)
    text = _strip_json_fence(text)
    return schema.model_validate_json(text)


def _circuit_id(theme: BatchTheme, index: int) -> str:
    """Nome canonico del circuito, corrispondente al nome del file .py (se scritto su disco)."""
    return f"synthetic_{theme.theme}_{index}"


def _write_circuit_file(circuit_id: str, source_code: str) -> None:
    OUTPUT_CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_CIRCUITS_DIR / f"{circuit_id}.py").write_text(source_code, encoding="utf-8")


def _process_circuit(circuit, theme, index, facade, accepted_in_batch: list):
    circuit_id = _circuit_id(theme, index)
    verification = verify_generated_circuit(circuit, facade)
    structural_check_passed = not verification["discarded"]

    # La forma MISURATA e' presente solo se il circuito ha superato la verifica strutturale: su
    # codice che non compila, o che assegna piu' di un circuito, non c'e' nulla da
    # misurare, e riportare una forma fittizia falserebbe le statistiche del dataset.
    shape = None
    if structural_check_passed:
        shape = {
            key: verification[key]
            for key in ("l", "c", "lc_product", "idq", "idq_worst_qubit", "measured_smells")
        }

    duplicate_of = (
        is_near_duplicate(circuit, accepted_in_batch) if structural_check_passed else None
    )

    if structural_check_passed and duplicate_of is None:
        _write_circuit_file(circuit_id, circuit.source_code)
        accepted_in_batch.append(circuit)

    return {
        "circuit_id": circuit_id,
        "source_code": circuit.source_code,
        "structural_check_passed": structural_check_passed,
        "compile_error": verification.get("compile_error"),
        "assigned_circuits": verification.get("assigned_circuits"),
        "dead_circuit": verification.get("dead_circuit"),
        # Punto cieco noto di QSMELL, registrato come metadato: un qubit allocato e mai toccato
        # ha IdQ = 0 e la metrica lo considera pulito (vedi verification.structural_idle_check).
        "untouched_qubit_indices": verification.get("untouched_qubit_indices"),
        "measured_shape": shape,
        "measured_smells": shape["measured_smells"] if shape else None,
        "duplicate_of": duplicate_of,
        "generation_batch": theme.theme,
    }


def run_generation(model: str) -> None:
    """Orchestra la generazione, verifica e scrittura su disco per tutti i BATCH_THEMES."""
    llm = LLM(model=model, temperature=0.8)
    facade = QiskitFacade()

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)

    with GROUND_TRUTH_PATH.open("a", encoding="utf-8") as ground_truth_file:
        for theme in BATCH_THEMES:
            print(f"[BATCH] {theme.theme} -- richiesti {theme.count} circuiti")
            prompt = build_batch_prompt(theme)
            batch: GenerationBatch = generate_batch(llm, prompt, GenerationBatch)

            accepted_in_batch = []

            for index, circuit in enumerate(batch.circuits, start=1):
                record = _process_circuit(
                    circuit=circuit,
                    theme=theme,
                    index=index,
                    facade=facade,
                    accepted_in_batch=accepted_in_batch,
                )
                ground_truth_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                ground_truth_file.flush()

    print(
        f"Generazione completata. Circuiti in {OUTPUT_CIRCUITS_DIR}, metadati in {GROUND_TRUTH_PATH}"
    )


def _print_summary(model: str) -> None:
    total = sum(theme.count for theme in BATCH_THEMES)
    print("Riepilogo generazione dataset sintetico")
    print(f"  Modello:              {model}")
    print(f"  Lotti:                {len(BATCH_THEMES)}")
    # Nessuna colonna "atteso": il lotto non prevede piu' un'etichetta. Descrive una struttura
    # di codice e l'etichetta esce dalla misura a valle, quindi qui c'e' solo cosa si chiede.
    for theme in BATCH_THEMES:
        print(
            f"    - {theme.theme}: {theme.count} circuiti "
            f"(qubit {theme.qubit_range[0]}-{theme.qubit_range[1]})"
        )
    print(f"  Totale circuiti:      {total}")
    print(f"  Output circuiti:      {OUTPUT_CIRCUITS_DIR}")
    print(f"  Output metadati:      {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Stringa modello CrewAI/LiteLLM per la generazione (default: {DEFAULT_MODEL}).",
    )
    args = parser.parse_args()

    _print_summary(args.model)
    confirmation = input("Procedere con la generazione (chiamate LLM reali)? [y/N] ")
    if confirmation.strip().lower() != "y":
        print("Generazione annullata.")
    else:
        run_generation(model=args.model)
