"""
Script di esecuzione della pipeline QSCSOP sul dataset arricchito prodotto da QCEP.

Istanzia le dipendenze concrete (QiskitFacade, i tre agenti LLM, ValidationService,
MASEngine, JsonlDatasetAdapter, JsonlDataSink) e le inietta in PipelineCoordinator, poi
lancia il ciclo detect-refactor-validate-review end-to-end su ogni circuito di
data/interim/dataset_pulito.jsonl. Produce data/interim/risultati_dataset.jsonl.

Uso (dalla root del progetto, con il venv attivo):
    python scripts/run_qscsop.py
"""

import logging
from pathlib import Path

from crewai import LLM

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.adapters.jsonl_dataset_adapter import JsonlDatasetAdapter
from qscsop_pipeline.qscsop.mas.agents.detector_agent import DetectorAgent
from qscsop_pipeline.qscsop.mas.agents.refactorer_agent import RefactorerAgent
from qscsop_pipeline.qscsop.mas.agents.reviewer_agent import ReviewerAgent
from qscsop_pipeline.qscsop.mas.llm_config import DEFAULT_AGENT_MODEL, DETECTOR_MODEL
from qscsop_pipeline.qscsop.mas.mas_engine import MASEngine
from qscsop_pipeline.qscsop.mas.validation.validation_service import ValidationService
from qscsop_pipeline.qscsop.pipeline_coordinator import PipelineCoordinator
from qscsop_pipeline.qscsop.sinks.jsonl_data_sink import JsonlDataSink

# scripts/run_qscsop.py -> risale alla root del progetto, stesso pattern di run_qcep.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "dataset_pulito.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "risultati_dataset.jsonl"

# Budget di iterazioni del ciclo refactor-validate-review: stesso valore gia' usato nei test e2e
# (test_mas_engine_e2e.py, test_iterative_refactor_review_loop_e2e.py). Default ragionevole: da'
# al ciclo di feedback una possibilita' realistica di convergere (il primo tentativo a volte e'
# gia' valido, altre volte serve un secondo giro) senza un costo in tempo illimitato per circuito.
MAX_ITERATIONS = 3


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    facade = QiskitFacade()

    # DetectorAgent su un modello piu' grande (DETECTOR_MODEL): richiede un giudizio di
    # classificazione corretto al primo colpo, dove RefactorerAgent/ReviewerAgent tollerano
    # meglio l'approssimazione essendo corretti dal ciclo iterativo di validazione/review. Vedi
    # qscsop_pipeline/qscsop/mas/llm_config.py
    detector_llm = LLM(model=DETECTOR_MODEL, temperature=0.6)
    # Una singola istanza per ReviewerAgent.
    agent_llm = LLM(model=DEFAULT_AGENT_MODEL, temperature=0.6)

    detector_agent = DetectorAgent(llm=detector_llm)
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

    dataset_adapter = JsonlDatasetAdapter(filepath=str(INPUT_PATH))
    data_sink = JsonlDataSink(filepath=str(OUTPUT_PATH))

    pipeline_coordinator = PipelineCoordinator(
        dataset_adapter=dataset_adapter,
        mas_engine=mas_engine,
        data_sink=data_sink,
    )
    pipeline_coordinator.run()

    print(f"QSCSOP completato. Output scritto in {OUTPUT_PATH}")
