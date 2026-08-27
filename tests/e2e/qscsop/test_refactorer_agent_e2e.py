"""E2E del RefactorerAgent con LLM reale (Ollama) e vera QiskitFacade. MAI in CI.

Il controllo piu' importante non e' che il codice refattorizzato compili, ma che preservi la
semantica del circuito originale (check_equivalence): un refactoring che rompe l'equivalenza e'
un fallimento reale, non da mascherare aggiustando il test.
"""

from pathlib import Path

import pytest
from crewai import LLM

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.mas.agents.detector_agent import DetectorAgent
from qscsop_pipeline.qscsop.mas.agents.refactorer_agent import RefactorerAgent
from qscsop_pipeline.qscsop.mas.llm_config import DEFAULT_AGENT_MODEL, DETECTOR_MODEL

# tests/e2e/qscsop/ -> risali a root repo, poi ai file smelly usati come baseline reale.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "thesmellyeight"
_LC_SMELLY_PATH = _DATA_DIR / "lc" / "lc-smelly.py"
_IDQ_SMELLY_PATH = _DATA_DIR / "idq" / "idq-smelly.py"


@pytest.mark.e2e
def test_refactorer_agent_fixes_long_circuit_preserving_equivalence() -> None:
    # DetectorAgent su DETECTOR_MODEL, RefactorerAgent su DEFAULT_AGENT_MODEL: vedi
    # qscsop_pipeline.qscsop.mas.llm_config per la diagnosi che ha motivato modelli diversi
    # per-agente.
    detector = DetectorAgent(llm=LLM(model=DETECTOR_MODEL, temperature=0, facade=QiskitFacade()))
    refactorer = RefactorerAgent(llm=LLM(model=DEFAULT_AGENT_MODEL, temperature=0))
    facade = QiskitFacade()

    baseline_code = _LC_SMELLY_PATH.read_text(encoding="utf-8")
    smell_report = detector.detect_smell(baseline_code)
    refactored_code = refactorer.refactor(baseline_code, smell_report, review_feedback="")

    print("\n[RefactorerAgent E2E LC] refactored_code:\n" + refactored_code)

    compiled, error = facade.compile_circuit(refactored_code)
    assert (compiled, error) == (True, None)

    # Controllo centrale: il refactoring deve preservare la semantica del circuito originale.
    assert facade.check_equivalence(baseline_code, refactored_code) is True


@pytest.mark.e2e
@pytest.mark.xfail(
    reason=(
        "Tentativo singolo (senza review loop) del modello 7B locale perde gate legittimi "
        "durante il refactoring Idle Qubits — comportamento noto, vedi "
        "docs/report_qscsop_refactoring_equivalence.md sezione 5. Il ciclo iterativo con "
        "ReviewerAgent è progettato per correggere esattamente questo caso: vedi "
        "test_iterative_refactor_review_loop_e2e."
    ),
    strict=False,
)
def test_refactorer_agent_fixes_idle_qubits_preserving_equivalence() -> None:
    # DetectorAgent su DETECTOR_MODEL, RefactorerAgent su DEFAULT_AGENT_MODEL: vedi
    # qscsop_pipeline.qscsop.mas.llm_config per la diagnosi che ha motivato modelli diversi
    # per-agente.
    detector = DetectorAgent(llm=LLM(model=DETECTOR_MODEL, temperature=0, facade=QiskitFacade()))
    refactorer = RefactorerAgent(llm=LLM(model=DEFAULT_AGENT_MODEL, temperature=0))
    facade = QiskitFacade()

    baseline_code = _IDQ_SMELLY_PATH.read_text(encoding="utf-8")
    smell_report = detector.detect_smell(baseline_code)
    refactored_code = refactorer.refactor(baseline_code, smell_report, review_feedback="")

    print("\n[RefactorerAgent E2E IDQ] refactored_code:\n" + refactored_code)

    compiled, error = facade.compile_circuit(refactored_code)
    assert (compiled, error) == (True, None)

    # Ispezione: la strategia corretta per Idle Qubits RIDUCE i qubit logici. Il baseline ne ha 3;
    # ci si aspetta che il refattorizzato ne abbia meno. Stampato per verifica manuale.
    baseline_qubits = facade.isolate_circuit(baseline_code).num_qubits
    refactored_qubits = facade.isolate_circuit(refactored_code).num_qubits
    print(
        f"[RefactorerAgent E2E IDQ] qubit baseline={baseline_qubits} refactored={refactored_qubits}"
    )

    # Controllo centrale: il refactoring deve preservare la semantica del circuito originale.
    assert facade.check_equivalence(baseline_code, refactored_code) is True
