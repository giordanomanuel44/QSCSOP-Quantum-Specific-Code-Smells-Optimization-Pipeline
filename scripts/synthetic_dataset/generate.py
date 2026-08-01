"""
Entry point per la generazione di un dataset sintetico di circuiti Qiskit (smelly + puliti).

Tooling di preparazione dati, stesso ruolo di scripts/fetch_datasets.py -- NON e' un modulo
dell'architettura QCEP/QSCSOP/Analytics, vive fuori da src/qscsop_pipeline/.

Per ogni lotto in prompts.BATCH_THEMES: costruisce il prompt (prompts.build_batch_prompt),
chiama l'LLM per un GenerationBatch strutturato (generate_batch), verifica ogni circuito
strutturalmente (verification.verify_generated_circuit), campiona un sottoinsieme per il
DetectorAgent reale, scrive i circuiti non scartati in data/raw/synthetic/ e un record di
metadati per OGNI circuito (scartati inclusi, per tracciabilita') in
data/interim/synthetic_ground_truth.jsonl.

Uso (dalla root del progetto, con il venv attivo e Ollama in esecuzione):
    python -m scripts.synthetic_dataset.generate
    python -m scripts.synthetic_dataset.generate --model "anthropic/claude-sonnet-5" --sample-every 3

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
from qscsop_pipeline.qscsop.mas.agents.detector_agent import DetectorAgent
from qscsop_pipeline.qscsop.mas.llm_config import DETECTOR_MODEL
from scripts.synthetic_dataset.prompts import (
    BATCH_THEMES,
    BatchTheme,
    GeneratedCircuit,
    GenerationBatch,
    build_batch_prompt,
)
from scripts.synthetic_dataset.verification import verify_generated_circuit

# scripts/synthetic_dataset/generate.py -> risale alla root del progetto: un .parent in piu'
# di scripts/fetch_datasets.py, perche' questo file vive un livello piu' in profondita'.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CIRCUITS_DIR = PROJECT_ROOT / "data" / "raw" / "synthetic"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "interim" / "synthetic_ground_truth.jsonl"

# Modello di default: sostituibile a riga di comando (--model) con un modello Claude via
# LiteLLM/CrewAI (es. "anthropic/claude-sonnet-5") quando/se si decidera' di passare a quello.
DEFAULT_MODEL = "ollama/qwen2.5-coder:14b"

# Campiona 1 circuito ogni N per il confronto col DetectorAgent reale (Parte 5), non ogni
# circuito: il DetectorAgent e' costoso (LLM reale), la verifica strutturale gia' copre gran
# parte della diagnostica per Idle Qubits.
DEFAULT_SAMPLE_EVERY = 5

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


def _sample_with_detector_agent(
    detector_agent: DetectorAgent, circuit: GeneratedCircuit, circuit_id: str
) -> bool:
    """Confronta detected_smells del DetectorAgent con intended_smells; ritorna l'accordo.

    Non scarta nulla in caso di disaccordo: stampa un blocco leggibile per revisione manuale,
    coerente con la debolezza strutturale della verifica automatica per Long Circuit (vedi
    verification.py).
    """
    report = detector_agent.detect_smell(circuit.source_code)
    detected = set(report.get_detected_smells())
    intended = {smell.value for smell in circuit.intended_smells}
    agrees = detected == intended

    if not agrees:
        print(
            f"[DETECTOR MISMATCH] circuit_id={circuit_id}\n"
            f"  intended: {sorted(intended)}\n"
            f"  detected: {sorted(detected)}\n"
            f"  detector report_details: {report.get_report_details()}"
        )

    return agrees


def _process_circuit(
    circuit: GeneratedCircuit,
    theme: BatchTheme,
    index: int,
    facade: QiskitFacade,
    detector_agent: DetectorAgent,
    sample_every: int,
    processed_count: int,
) -> dict:
    """Verifica, eventualmente scrive su disco e campiona un singolo circuito generato."""
    circuit_id = _circuit_id(theme, index)
    verification = verify_generated_circuit(circuit, facade)
    structural_check_passed = not verification["discarded"]

    detector_agreement: bool | None = None
    if structural_check_passed:
        _write_circuit_file(circuit_id, circuit.source_code)
        if processed_count % sample_every == 0:
            detector_agreement = _sample_with_detector_agent(detector_agent, circuit, circuit_id)

    return {
        "circuit_id": circuit_id,
        "source_code": circuit.source_code,
        "intended_smells": [smell.value for smell in circuit.intended_smells],
        "line_by_line_expansion": circuit.line_by_line_expansion,
        "qubit_operation_analysis": circuit.qubit_operation_analysis,
        "reasoning": circuit.reasoning,
        "structural_check_passed": structural_check_passed,
        "detector_agreement": detector_agreement,
        "generation_batch": theme.theme,
    }


def run_generation(model: str, sample_every: int) -> None:
    """Orchestra la generazione, verifica e scrittura su disco per tutti i BATCH_THEMES."""
    llm = LLM(model=model, temperature=0.9)
    detector_llm = LLM(model=DETECTOR_MODEL, temperature=0)
    detector_agent = DetectorAgent(llm=detector_llm)
    facade = QiskitFacade()

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    with GROUND_TRUTH_PATH.open("a", encoding="utf-8") as ground_truth_file:
        for theme in BATCH_THEMES:
            print(f"[BATCH] {theme.theme} -- richiesti {theme.total_circuits} circuiti")
            prompt = build_batch_prompt(theme)
            batch: GenerationBatch = generate_batch(llm, prompt, GenerationBatch)

            for index, circuit in enumerate(batch.circuits, start=1):
                processed_count += 1
                record = _process_circuit(
                    circuit=circuit,
                    theme=theme,
                    index=index,
                    facade=facade,
                    detector_agent=detector_agent,
                    sample_every=sample_every,
                    processed_count=processed_count,
                )
                ground_truth_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                ground_truth_file.flush()

    print(f"Generazione completata. Circuiti in {OUTPUT_CIRCUITS_DIR}, metadati in {GROUND_TRUTH_PATH}")


def _print_summary(model: str, sample_every: int) -> None:
    total_smelly = sum(theme.count_smelly for theme in BATCH_THEMES)
    total_clean = sum(theme.count_clean for theme in BATCH_THEMES)
    print("Riepilogo generazione dataset sintetico")
    print(f"  Modello:              {model}")
    print(f"  Modello DetectorAgent (campionamento): {DETECTOR_MODEL}")
    print(f"  Campionamento:        1 ogni {sample_every} circuiti")
    print(f"  Lotti:                {len(BATCH_THEMES)}")
    for theme in BATCH_THEMES:
        print(
            f"    - {theme.theme}: {theme.count_smelly} smelly + {theme.count_clean} puliti "
            f"(qubit {theme.qubit_range[0]}-{theme.qubit_range[1]}, focus={theme.smell_focus.value})"
        )
    print(f"  Totale circuiti:      {total_smelly + total_clean} ({total_smelly} smelly, {total_clean} puliti)")
    print(f"  Output circuiti:      {OUTPUT_CIRCUITS_DIR}")
    print(f"  Output metadati:      {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Stringa modello CrewAI/LiteLLM per la generazione (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=DEFAULT_SAMPLE_EVERY,
        help=f"Campiona 1 circuito ogni N per il DetectorAgent (default: {DEFAULT_SAMPLE_EVERY}).",
    )
    args = parser.parse_args()

    _print_summary(args.model, args.sample_every)
    confirmation = input("Procedere con la generazione (chiamate LLM reali)? [y/N] ")
    if confirmation.strip().lower() != "y":
        print("Generazione annullata.")
    else:
        run_generation(model=args.model, sample_every=args.sample_every)
