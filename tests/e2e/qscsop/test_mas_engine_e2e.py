"""E2E DIAGNOSTICO del MASEngine reale (sezione 1.4.3, Activity Diagram Fig. 1.6). MAI in CI.

A differenza di test_iterative_refactor_review_loop_e2e.py (orchestrazione manuale del ciclo),
qui e' il MASEngine stesso a coordinare i quattro collaboratori reali (DetectorAgent,
RefactorerAgent, ValidationService, ReviewerAgent). Caso di studio: idq-smelly.py, stesso smell
noto (Idle Qubits) gia' usato nel test del ciclo iterativo.

Il DetectorAgent usa un modello diverso (piu' grande) da RefactorerAgent/ReviewerAgent: vedi
qscsop_pipeline.qscsop.mas.llm_config per la diagnosi che ha motivato la scelta.

NATURA DEL TEST: diagnostico, non pass/fail sull'esito del ciclo. La convergenza entro
max_iterations=3 non e' garantita; l'informazione utile sta nell'output stampato (da leggere con
`pytest -s`). Solo le garanzie strutturali dell'entita' (stato terminale valido, coerenza interna
refactored/is_functionally_equivalent, identita' dell'istanza) sono asserite.
"""

from pathlib import Path

import pytest
from crewai import LLM

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.entities.circuit_version import CircuitVersion
from qscsop_pipeline.qscsop.entities.smell_metrics import SmellMetrics
from qscsop_pipeline.qscsop.entities.evaluation_status import EvaluationStatus
from qscsop_pipeline.qscsop.entities.quantum_program_entity import QuantumProgramEntity
from qscsop_pipeline.qscsop.mas.agents.detector_agent import DetectorAgent
from qscsop_pipeline.qscsop.mas.agents.refactorer_agent import RefactorerAgent
from qscsop_pipeline.qscsop.mas.agents.reviewer_agent import ReviewerAgent
from qscsop_pipeline.qscsop.mas.llm_config import DEFAULT_AGENT_MODEL, DETECTOR_MODEL
from qscsop_pipeline.qscsop.mas.mas_engine import MASEngine
from qscsop_pipeline.qscsop.mas.validation.validation_service import ValidationService

# tests/e2e/qscsop/ -> risali a root repo, poi al file smelly usato come baseline reale.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "thesmellyeight"
_IDQ_SMELLY_PATH = _DATA_DIR / "idq" / "idq-smelly.py"

_TAG = "[MASEngine E2E]"


def _build_baseline_entity(facade: QiskitFacade) -> QuantumProgramEntity:
    """Costruisce la QuantumProgramEntity di ingresso misurando il baseline con la facade.

    Nessun numero inventato: l, c e IdQ vengono da calculate_smell_metrics, la stessa misura che
    QCEP persiste nel dataset e su cui ValidationService da' il verdetto.
    """
    source_code = _IDQ_SMELLY_PATH.read_text(encoding="utf-8")

    smell_metrics = facade.calculate_smell_metrics(source_code)

    baseline = CircuitVersion(
        source_code=source_code,
        smell_metrics=SmellMetrics(
            max_ops_per_qubit=smell_metrics["longCircuit"]["maxOpsPerQubit"],
            max_parallel_ops=smell_metrics["longCircuit"]["maxParallelOps"],
            idle_qubits=smell_metrics["idleQubits"]["value"],
        ),
    )
    return QuantumProgramEntity(
        circuit_id="idq-smelly-e2e", dataset_source="TheSmellyEight", baseline=baseline
    )


@pytest.mark.e2e
def test_mas_engine_processes_idle_qubits_circuit_end_to_end() -> None:
    # DetectorAgent su un modello piu' grande (DETECTOR_MODEL): richiede piu' rigore analitico,
    # vedi qscsop_pipeline.qscsop.mas.llm_config per la diagnosi che ha motivato la scelta.
    # RefactorerAgent e ReviewerAgent restano sul modello piu' piccolo e veloce
    # (DEFAULT_AGENT_MODEL).
    detector_llm = LLM(model=DETECTOR_MODEL, temperature=0)
    agent_llm = LLM(model=DEFAULT_AGENT_MODEL, temperature=0)
    facade = QiskitFacade()
    detector = DetectorAgent(llm=detector_llm)
    refactorer = RefactorerAgent(llm=agent_llm)
    reviewer = ReviewerAgent(llm=agent_llm)
    validation_service = ValidationService(facade=facade)
    # max_iterations=3: budget ragionevole per dare al ciclo di feedback una possibilita'
    # realistica di convergere, non 1 (il primo tentativo a volte e' gia' valido, altre volte
    # serve un secondo giro, vedi test_iterative_refactor_review_loop_e2e.py).
    mas_engine = MASEngine(
        max_iterations=3,
        detector_agent=detector,
        refactorer_agent=refactorer,
        validation_service=validation_service,
        reviewer_agent=reviewer,
    )

    entity = _build_baseline_entity(facade)
    baseline = entity.get_baseline()

    print(
        f"\n{_TAG} baseline smellMetrics: "
        f"l={baseline.get_smell_metrics().get_max_ops_per_qubit()}, "
        f"c={baseline.get_smell_metrics().get_max_parallel_ops()}, "
        f"l*c={baseline.get_smell_metrics().long_circuit}, "
        f"IdQ={baseline.get_smell_metrics().get_idle_qubits()}"
    )

    result = mas_engine.process_entity(entity)

    evaluation = result.get_evaluation()
    status = evaluation.get_status()
    iteration_count = evaluation.get_iteration_count()
    detected_smells = evaluation.get_detected_smells()
    is_functionally_equivalent = evaluation.get_is_functionally_equivalent()

    print(f"\n{_TAG} --- ESITO ---")
    print(f"{_TAG} status={status.value}")
    print(f"{_TAG} iteration_count={iteration_count}")
    print(f"{_TAG} detected_smells={detected_smells}")
    print(f"{_TAG} is_functionally_equivalent={is_functionally_equivalent}")

    if status == EvaluationStatus.OPTIMIZED:
        refactored = result.get_refactored()
        print(f"\n{_TAG} --- REFACTORED (OPTIMIZED) ---")
        print(f"{_TAG} codice prodotto:\n{refactored.get_source_code()}")
        print(
            f"{_TAG} l*c: baseline={baseline.get_smell_metrics().long_circuit} -> "
            f"refactored={refactored.get_smell_metrics().long_circuit}"
        )
        print(
            f"{_TAG} IdQ: baseline={baseline.get_smell_metrics().get_idle_qubits()} -> "
            f"refactored={refactored.get_smell_metrics().get_idle_qubits()}"
        )
    elif status == EvaluationStatus.OPT_FAILED:
        print(
            f"\n{_TAG} ATTENZIONE: il ciclo non e' converso entro max_iterations=3. Non e' un "
            "fallimento del test: indica un limite di autocorrezione del ciclo "
            "Refactorer/Reviewer (modello 7B locale) su questo tentativo specifico, non un bug "
            "dell'infrastruttura (agenti, ValidationService, QiskitFacade, MASEngine)."
        )

    # --- VERIFICHE STRUTTURALI (invarianti indipendenti dall'esito del ciclo) ---
    assert status in (EvaluationStatus.OPTIMIZED, EvaluationStatus.OPT_FAILED)
    assert detected_smells != []
    assert 1 <= iteration_count <= 3

    if status == EvaluationStatus.OPTIMIZED:
        assert result.get_refactored() is not None
        assert is_functionally_equivalent is True
    else:
        assert result.get_refactored() is None
        assert is_functionally_equivalent is False

    # process_entity muta in place: stessa istanza, non una copia.
    assert result is entity
