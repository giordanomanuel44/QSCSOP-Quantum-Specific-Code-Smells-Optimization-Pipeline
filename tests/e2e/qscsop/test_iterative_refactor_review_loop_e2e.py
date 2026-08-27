"""E2E DIAGNOSTICO del ciclo Refactorer -> Validation -> Reviewer -> Refactorer. MAI in CI.

Dimostra il concetto del ciclo di correzione iterativa descritto nella sezione 1.4.3 della tesi
("Ciclo Iterativo e Gestione della Convergenza") usando i componenti reali gia' costruiti. Il
MASEngine non esiste ancora: l'orchestrazione dei due tentativi e' fatta a mano qui nel test,
esattamente nell'ordine in cui il motore la eseguira'.

Caso di studio: idq-smelly.py. Il primo tentativo del modello 7B locale e' gia' documentato come
non equivalente (perde gate legittimi mentre rimuove il qubit inerte, vedi
docs/report_qscsop_refactoring_equivalence.md sezione 5); il secondo tentativo riceve il feedback
contestualizzato del ReviewerAgent e ha la possibilita' di correggersi.

NATURA DEL TEST: diagnostico, non pass/fail. Vedi il commento esteso a fine file sul perche' non
esiste alcun assert sull'esito del secondo tentativo.
"""

from pathlib import Path

import pytest
from crewai import LLM

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.mas.agents.detector_agent import DetectorAgent
from qscsop_pipeline.qscsop.mas.agents.refactorer_agent import RefactorerAgent
from qscsop_pipeline.qscsop.mas.agents.reviewer_agent import ReviewerAgent
from qscsop_pipeline.qscsop.mas.llm_config import DEFAULT_AGENT_MODEL, DETECTOR_MODEL
from qscsop_pipeline.qscsop.mas.validation.validation_service import ValidationService

# tests/e2e/qscsop/ -> risali a root repo, poi al file smelly usato come baseline reale.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "thesmellyeight"
_IDQ_SMELLY_PATH = _DATA_DIR / "idq" / "idq-smelly.py"

_TAG = "[Refactor/Review loop E2E]"


@pytest.mark.e2e
def test_review_feedback_guides_a_second_refactoring_attempt() -> None:
    # DetectorAgent su un modello piu' grande (DETECTOR_MODEL), RefactorerAgent/ReviewerAgent sul
    # modello piu' piccolo e veloce (DEFAULT_AGENT_MODEL): vedi qscsop.mas.llm_config per la
    # diagnosi.
    detector_llm = LLM(model=DETECTOR_MODEL, temperature=0)
    agent_llm = LLM(model=DEFAULT_AGENT_MODEL, temperature=0)
    facade = QiskitFacade()
    detector = DetectorAgent(llm=detector_llm, facade=QiskitFacade())
    refactorer = RefactorerAgent(llm=agent_llm)
    reviewer = ReviewerAgent(llm=agent_llm)
    validation_service = ValidationService(facade=facade)

    baseline_code = _IDQ_SMELLY_PATH.read_text(encoding="utf-8")

    smell_report = detector.detect_smell(baseline_code)
    print(f"\n{_TAG} has_smells={smell_report.get_has_smells()}")
    print(f"{_TAG} report_details:\n{smell_report.get_report_details()}")

    # --- TENTATIVO 1: nessun feedback disponibile (prima iterazione del ciclo). ---
    attempt_1 = refactorer.refactor(baseline_code, smell_report, review_feedback="")
    validation_1 = validation_service.validate(baseline_code, attempt_1)

    print(f"\n{_TAG} --- TENTATIVO 1 ---")
    print(f"{_TAG} codice prodotto:\n{attempt_1}")
    print(f"{_TAG} is_valid={validation_1.get_is_valid()}")
    print(f"{_TAG} raw_error_data={validation_1.get_raw_error_data()!r}")

    # Esito atteso False sulla base di quanto gia' documentato, ma NON assunto: se il modello
    # azzecca il refactoring al primo colpo il ciclo di review semplicemente non serve.
    if validation_1.get_is_valid():
        print(
            f"{_TAG} Il primo tentativo e' gia' valido: il ciclo di feedback non e' stato "
            "necessario in questa esecuzione."
        )
        return

    # --- REVIEW: l'intero ValidationResultDTO diventa feedback azionabile. ---
    feedback = reviewer.review(validation_1, smell_report, attempt_1)
    print(f"\n{_TAG} --- FEEDBACK DEL REVIEWER ---")
    print(feedback)

    # --- TENTATIVO 2: stesso baseline e stesso smell report, ma con il feedback iniettato. ---
    attempt_2 = refactorer.refactor(baseline_code, smell_report, review_feedback=feedback)
    validation_2 = validation_service.validate(baseline_code, attempt_2)

    print(f"\n{_TAG} --- TENTATIVO 2 ---")
    if validation_2.get_is_valid():
        print(f"{_TAG} Il ciclo di feedback ha funzionato: il secondo tentativo e' valido.")
    else:
        print(
            f"{_TAG} Il secondo tentativo non e' ancora valido dopo il feedback. "
            f"raw_error_data: {validation_2.get_raw_error_data()!r}"
        )
    print(f"{_TAG} codice prodotto:\n{attempt_2}")

    # NESSUN ASSERT sull'esito di validation_2, deliberatamente.
    #
    # Un fallimento del secondo tentativo NON indica un bug del sistema: indica un limite della
    # capacita' di autocorrezione del modello 7B locale su questo tentativo specifico, con un solo
    # giro di feedback e temperatura 0. Trattarlo come regressione renderebbe rosso un test la cui
    # infrastruttura (agenti, ValidationService, QiskitFacade) sta funzionando correttamente, e
    # spingerebbe ad "aggiustare" il test invece di leggerne l'esito.
    #
    # L'informazione utile di questo test sta interamente nell'output stampato (feedback prodotto
    # dal ReviewerAgent e codice del secondo tentativo), da leggere con `pytest -s`: e' materiale
    # di ispezione per il capitolo sperimentale, non un cancello pass/fail.
