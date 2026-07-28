"""E2E del ReviewerAgent con LLM reale (Ollama) e vera QiskitFacade. MAI in CI.

Caso di studio reale gia' noto: il refactoring Idle Qubits di idq-smelly.py prodotto dal modello
locale rimuove correttamente il qubit inerte ma perde gate legittimi, quindi non e' funzionalmente
equivalente al baseline. Si verifica che il ReviewerAgent produca un feedback non vuoto a partire
da quel fallimento; il contenuto e' materiale di ISPEZIONE manuale, non di asserzione semantica.
Il ciclo completo (RefactorerAgent che si corregge usando questo feedback) e' invece coperto da
test_iterative_refactor_review_loop_e2e.
"""

from pathlib import Path

import pytest
from crewai import LLM

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.mas.agents.detector_agent import DetectorAgent
from qscsop_pipeline.qscsop.mas.agents.refactorer_agent import RefactorerAgent
from qscsop_pipeline.qscsop.mas.agents.reviewer_agent import ReviewerAgent
from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO
from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO
from qscsop_pipeline.qscsop.mas.llm_config import DEFAULT_AGENT_MODEL, DETECTOR_MODEL
from qscsop_pipeline.qscsop.mas.validation.validation_service import ValidationService

# tests/e2e/qscsop/ -> risali a root repo, poi al file smelly usato come baseline reale.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "thesmellyeight"
_IDQ_SMELLY_PATH = _DATA_DIR / "idq" / "idq-smelly.py"


@pytest.mark.e2e
def test_reviewer_agent_contextualizes_a_real_validation_failure() -> None:
    # DetectorAgent su DETECTOR_MODEL, RefactorerAgent/ReviewerAgent su DEFAULT_AGENT_MODEL:
    # vedi qscsop_pipeline.qscsop.mas.llm_config per la diagnosi che ha motivato modelli diversi
    # per-agente.
    detector_llm = LLM(model=DETECTOR_MODEL, temperature=0)
    agent_llm = LLM(model=DEFAULT_AGENT_MODEL, temperature=0)
    detector = DetectorAgent(llm=detector_llm)
    refactorer = RefactorerAgent(llm=agent_llm)
    reviewer = ReviewerAgent(llm=agent_llm)
    facade = QiskitFacade()
    validation_service = ValidationService(facade=facade)

    baseline_code = _IDQ_SMELLY_PATH.read_text(encoding="utf-8")
    smell_report = detector.detect_smell(baseline_code)
    refactored_code = refactorer.refactor(baseline_code, smell_report, review_feedback="")

    # ValidationResultDTO reale, prodotto dal ValidationService vero, come farebbe il MASEngine.
    validation_result = validation_service.validate(baseline_code, refactored_code)
    print(f"\n[ReviewerAgent E2E] is_valid={validation_result.get_is_valid()}")
    print(f"[ReviewerAgent E2E] raw_error_data: {validation_result.get_raw_error_data()}")

    if validation_result.get_is_valid():
        print(
            "[ReviewerAgent E2E] Il refactoring e' gia' valido: nessun errore da contestualizzare "
            "in questa esecuzione."
        )
        return

    feedback = reviewer.review(validation_result, smell_report)

    print("[ReviewerAgent E2E] feedback:\n" + feedback)

    # Ispezione qualitativa: si verifica solo che il feedback esista e sia sostanziale, non il
    # suo contenuto semantico (non deterministico a livello di formulazione).
    assert isinstance(feedback, str)
    assert len(feedback.strip()) > 20


# Errore di natura completamente diversa dall'equivalenza funzionale: un fallimento di
# compilazione. Fabbricato (non prodotto da una run reale) perche' serve un caso deterministico
# del secondo ramo di _run_review_crew, quello in cui l'hint sui gate persi NON viene iniettato.
_SYNTAX_ERROR_DETAILS = (
    'Traceback (most recent call last):\n  File "<string>", line 3\n    qc.h(0\n'
    "         ^\nSyntaxError: '(' was never closed"
)


@pytest.mark.e2e
def test_reviewer_agent_contextualizes_a_compilation_failure() -> None:
    llm = LLM(model=DEFAULT_AGENT_MODEL, temperature=0)
    reviewer = ReviewerAgent(llm=llm)

    smell_report = SmellReportDTO(
        has_smells=True, report_details="Qubit 2 non contribuisce al risultato: Idle Qubits."
    )
    # Fallimento di compilazione fabbricato (ValidationService non e' invocato qui apposta: serve
    # un new_code sintatticamente rotto, che compile_circuit intercetterebbe prima di arrivare a
    # questo punto): new_metrics=None per costruzione, come per un vero fallimento di compilazione.
    validation_result = ValidationResultDTO(
        is_valid=False, raw_error_data=_SYNTAX_ERROR_DETAILS, new_metrics=None
    )

    feedback = reviewer.review(validation_result, smell_report)

    print("\n[ReviewerAgent E2E syntax] raw_error_details:\n" + _SYNTAX_ERROR_DETAILS)
    print("[ReviewerAgent E2E syntax] feedback:\n" + feedback)

    # Stesso livello di rigore del test sopra: si verifica solo che il feedback sia sostanziale.
    # Lo scopo e' l'ispezione manuale dell'output, per vedere se il modello riconosce di trovarsi
    # davanti a un errore di sintassi e non a una perdita di equivalenza funzionale.
    assert isinstance(feedback, str)
    assert len(feedback.strip()) > 20
