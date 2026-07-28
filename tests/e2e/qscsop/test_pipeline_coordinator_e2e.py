
"""E2E BOUNDED di PipelineCoordinator con tutti i collaboratori reali. MAI in CI.

A differenza di test_mas_engine_e2e.py (che esercita MASEngine.process_entity() su una singola
entita' costruita a mano), qui e' l'intero PipelineCoordinator a orchestrare stream_programs ->
process_entity -> save_program end-to-end, con la stessa composizione di collaboratori dello
script di produzione (scripts/run_qscsop.py).

BOUNDED: usa solo i primi 2 record REALI di data/interim/dataset_pulito.jsonl (letti e riscritti
in un file temporaneo), non l'intero dataset da 13 record ne' circuiti finti -- un ciclo completo
per record (detect, fino a max_iterations tentativi di refactor-validate-review) ha gia' un costo
in tempo non trascurabile con un vero LLM, dimostrato da test_mas_engine_e2e.py.

NATURA DEL TEST: verifica l'infrastruttura di orchestrazione (il file di output esiste, contiene
esattamente il numero di righe atteso, ogni riga e' JSON valido con i campi previsti, lo stato
finale e' sempre uno dei tre terminali validi). Non rivaluta la qualita' del MAS (quali smell,
quali metriche, quale esito) -- quello e' gia' coperto da test_mas_engine_e2e.py -- quindi nessun
assert sul contenuto specifico degli esiti.
"""

import json
from pathlib import Path

import pytest
from crewai import LLM

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.adapters.jsonl_dataset_adapter import JsonlDatasetAdapter
from qscsop_pipeline.qscsop.entities.evaluation_status import EvaluationStatus
from qscsop_pipeline.qscsop.mas.agents.detector_agent import DetectorAgent
from qscsop_pipeline.qscsop.mas.agents.refactorer_agent import RefactorerAgent
from qscsop_pipeline.qscsop.mas.agents.reviewer_agent import ReviewerAgent
from qscsop_pipeline.qscsop.mas.llm_config import DEFAULT_AGENT_MODEL, DETECTOR_MODEL
from qscsop_pipeline.qscsop.mas.mas_engine import MASEngine
from qscsop_pipeline.qscsop.mas.validation.validation_service import ValidationService
from qscsop_pipeline.qscsop.pipeline_coordinator import PipelineCoordinator
from qscsop_pipeline.qscsop.sinks.jsonl_data_sink import JsonlDataSink

# tests/e2e/qscsop/ -> risali a root repo, poi al dataset reale prodotto da QCEP.
_REAL_DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "interim" / "dataset_pulito.jsonl"
)
_BOUNDED_RECORD_COUNT = 2

_TAG = "[PipelineCoordinator E2E]"

_VALID_TERMINAL_STATUSES = {
    EvaluationStatus.SMELL_FREE.value,
    EvaluationStatus.OPTIMIZED.value,
    EvaluationStatus.OPT_FAILED.value,
}


@pytest.mark.e2e
def test_pipeline_coordinator_processes_two_real_records_end_to_end(tmp_path: Path) -> None:
    # Primi 2 record REALI del dataset gia' prodotto da QCEP, non circuiti fabbricati.
    real_lines = _REAL_DATASET_PATH.read_text(encoding="utf-8").splitlines()[
        :_BOUNDED_RECORD_COUNT
    ]
    input_path = tmp_path / "dataset_pulito_bounded.jsonl"
    input_path.write_text("\n".join(real_lines) + "\n", encoding="utf-8")
    output_path = tmp_path / "risultati_dataset_bounded.jsonl"

    facade = QiskitFacade()
    detector_llm = LLM(model=DETECTOR_MODEL, temperature=0)
    agent_llm = LLM(model=DEFAULT_AGENT_MODEL, temperature=0)
    detector_agent = DetectorAgent(llm=detector_llm)
    refactorer_agent = RefactorerAgent(llm=agent_llm)
    reviewer_agent = ReviewerAgent(llm=agent_llm)
    validation_service = ValidationService(facade=facade)
    mas_engine = MASEngine(
        max_iterations=3,
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

    assert output_path.exists()
    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(output_lines) == _BOUNDED_RECORD_COUNT

    print(f"\n{_TAG} --- RIEPILOGO ({_BOUNDED_RECORD_COUNT} record) ---")
    for line in output_lines:
        record = json.loads(line)

        assert "circuitId" in record
        assert "datasetSource" in record
        assert "baseline" in record
        assert "evaluation" in record

        status = record["evaluation"]["status"]
        print(f"{_TAG} circuitId={record['circuitId']}: status={status}")

        assert status in _VALID_TERMINAL_STATUSES
