"""
Script di esecuzione della pipeline QSCSOP sul dataset arricchito prodotto da QCEP.

Istanzia le dipendenze concrete (QiskitFacade, i tre agenti LLM, ValidationService,
MASEngine, JsonlDatasetAdapter, JsonlDataSink) e le inietta in PipelineCoordinator, poi
lancia il ciclo detect-refactor-validate-review end-to-end su ogni circuito del dataset
di input.

Uso (dalla root del progetto, con il venv attivo):
    python scripts/run_qscsop.py                    # sintetico -> risultati_dataset_synthetic.jsonl
    python scripts/run_qscsop.py --real             # dataset_pulito.jsonl -> risultati_dataset.jsonl
    python scripts/run_qscsop.py --trace log.txt    # con la traccia completa del ciclo su file

IL DEFAULT E' IL SINTETICO, come in run_qcep.py: i due script vanno tenuti allineati, altrimenti
senza flag lavorerebbero su dataset diversi e i risultati non corrisponderebbero all'input che si
crede di aver processato. Ogni ramo va usato sull'output del ramo corrispondente di run_qcep.py.

I due esperimenti restano su file distinti, cosi' che i risultati sui circuiti sintetici (di cui
esiste la ground truth, joinabile su circuitId con synthetic_ground_truth_f.jsonl) non si
mescolino a quelli sui dataset reali.
"""

import argparse
import logging
from pathlib import Path

from crewai import LLM

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.adapters.jsonl_dataset_adapter import JsonlDatasetAdapter
from qscsop_pipeline.qscsop.coordinator.pipeline_coordinator import PipelineCoordinator
from qscsop_pipeline.qscsop.mas.agents.detector_agent import DetectorAgent
from qscsop_pipeline.qscsop.mas.agents.refactorer_agent import RefactorerAgent
from qscsop_pipeline.qscsop.mas.agents.reviewer_agent import ReviewerAgent
from qscsop_pipeline.qscsop.mas.llm_config import DEFAULT_AGENT_MODEL, DETECTOR_MODEL
from qscsop_pipeline.qscsop.mas.mas_engine import MASEngine
from qscsop_pipeline.qscsop.mas.validation.validation_service import ValidationService
from qscsop_pipeline.qscsop.sinks.jsonl_data_sink import JsonlDataSink

# scripts/run_qscsop.py -> risale alla root del progetto, stesso pattern di run_qcep.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INTERIM = PROJECT_ROOT / "data" / "interim"

INPUT_PATH = _INTERIM / "dataset_pulito.jsonl"
OUTPUT_PATH = _INTERIM / "risultati_dataset.jsonl"
SYNTHETIC_INPUT_PATH = _INTERIM / "dataset_pulito_synthetic.jsonl"
SYNTHETIC_OUTPUT_PATH = _INTERIM / "risultati_dataset_synthetic.jsonl"

# Budget di iterazioni del ciclo refactor-validate-review: stesso valore gia' usato nei test e2e
# (test_mas_engine_e2e.py, test_iterative_refactor_review_loop_e2e.py). Default ragionevole: da'
# al ciclo di feedback una possibilita' realistica di convergere (il primo tentativo a volte e'
# gia' valido, altre volte serve un secondo giro) senza un costo in tempo illimitato per circuito.
MAX_ITERATIONS = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        action="store_true",
        help="Elabora il dataset reale prodotto da run_qcep.py --real.",
    )
    parser.add_argument(
        "--trace",
        metavar="FILE",
        help=(
            "Scrive su FILE la tracciatura completa del ciclo per ogni circuito: prescrizione "
            "del Detector, codice prodotto a ogni tentativo, esito della validazione e feedback "
            "del Reviewer. Nessuno di questi testi finisce nel record dei risultati, quindi "
            "senza questo file un fallimento NOT_EQUIVALENT non e' distinguibile fra "
            "prescrizione sbagliata ed esecuzione sbagliata."
        ),
    )
    args = parser.parse_args()

    input_path = INPUT_PATH if args.real else SYNTHETIC_INPUT_PATH
    output_path = OUTPUT_PATH if args.real else SYNTHETIC_OUTPUT_PATH

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.trace:
        # La traccia va su FILE a DEBUG, mentre la console resta a INFO: i testi sono lunghi
        # (un codice refactored per tentativo) e renderebbero illeggibile l'avanzamento.
        trace_handler = logging.FileHandler(args.trace, mode="w", encoding="utf-8")
        trace_handler.setLevel(logging.DEBUG)
        trace_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        mas_logger = logging.getLogger("qscsop_pipeline.qscsop.mas.mas_engine")
        mas_logger.setLevel(logging.DEBUG)
        mas_logger.addHandler(trace_handler)

    facade = QiskitFacade()

    # DetectorAgent su un modello piu' grande (DETECTOR_MODEL): non classifica piu' -- lo fanno
    # la facade e le soglie -- ma deve PRESCRIVERE quali operazioni cambiare, che e' il compito
    # di lettura del codice piu' difficile dei tre. RefactorerAgent e ReviewerAgent tollerano
    # meglio l'approssimazione, essendo corretti dal ciclo iterativo di validazione/review. Vedi
    # qscsop_pipeline/qscsop/mas/llm_config.py
    detector_llm = LLM(model=DETECTOR_MODEL, temperature=0.6)
    # Una singola istanza per ReviewerAgent.
    agent_llm = LLM(model=DEFAULT_AGENT_MODEL, temperature=0.6)

    detector_agent = DetectorAgent(llm=detector_llm, facade=facade)
    refactorer_agent = RefactorerAgent(llm=detector_llm)
    reviewer_agent = ReviewerAgent(llm=agent_llm)
    validation_service = ValidationService(facade=facade)

    mas_engine = MASEngine(
        max_iterations=MAX_ITERATIONS,
        detector_agent=detector_agent,
        refactorer_agent=refactorer_agent,
        validation_service=validation_service,
        reviewer_agent=reviewer_agent,
    )

    dataset_adapter = JsonlDatasetAdapter(filepath=str(input_path))
    data_sink = JsonlDataSink(filepath=str(output_path))

    pipeline_coordinator = PipelineCoordinator(
        dataset_adapter=dataset_adapter,
        mas_engine=mas_engine,
        data_sink=data_sink,
    )
    pipeline_coordinator.run()

    print(f"QSCSOP completato. Output scritto in {output_path}")
