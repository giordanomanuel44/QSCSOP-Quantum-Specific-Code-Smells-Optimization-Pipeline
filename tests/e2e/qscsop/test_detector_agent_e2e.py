"""E2E del DetectorAgent con LLM reale (Ollama). MAI in CI: richiede il modello caricato in locale."""

from pathlib import Path

import pytest
from crewai import LLM

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.mas.agents.detector_agent import DetectorAgent
from qscsop_pipeline.qscsop.mas.llm_config import DETECTOR_MODEL

# tests/e2e/qscsop/ -> risali a root repo, poi ai file di esempio (smell noto / circuito corretto).
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "thesmellyeight"
_LC_SMELLY_PATH = _DATA_DIR / "lc" / "lc-smelly.py"
_LC_FIXED_PATH = _DATA_DIR / "lc" / "lc-fixed.py"
_IDQ_SMELLY_PATH = _DATA_DIR / "idq" / "idq-smelly.py"
_IDQ_FIXED_PATH = _DATA_DIR / "idq" / "idq-fixed.py"


@pytest.mark.e2e
def test_detector_agent_detects_smell_on_real_example() -> None:
    llm = LLM(model=DETECTOR_MODEL, temperature=0)
    agent = DetectorAgent(llm=llm, facade=QiskitFacade())

    code = _LC_SMELLY_PATH.read_text(encoding="utf-8")
    result = agent.detect_smell(code)

    print("\n[DetectorAgent E2E] report_details:\n" + result.get_report_details())
    print("[DetectorAgent E2E] detected_smells: " + str(result.get_detected_smells()))

    # Il modello dovrebbe riconoscere lo smell nel suo stesso esempio few-shot; se fallisce,
    # e' un segnale utile sulla qualita' del prompt/modello da rivedere.
    assert result.get_has_smells() is True
    assert "long_circuit" in result.get_detected_smells()


@pytest.mark.e2e
def test_detector_agent_detects_idle_qubits_on_real_example() -> None:
    llm = LLM(model=DETECTOR_MODEL, temperature=0)
    agent = DetectorAgent(llm=llm, facade=QiskitFacade())

    code = _IDQ_SMELLY_PATH.read_text(encoding="utf-8")

    # Accesso diretto al metodo privato SOLO per diagnosi: espone qubit_operation_analysis, il
    # campo di ragionamento interno che detect_smell() non propaga mai in SmellReportDTO (vedi
    # docstring di _SmellDetectionSchema in detector_agent.py). Una singola chiamata al Crew,
    # riusata sia per il ragionamento sia per il verdetto finale, invece di duplicare la
    # chiamata LLM richiamando anche detect_smell() separatamente.
    schema = agent._run_detection_crew(code)
    has_smells = bool(schema.detected_smell_types)
    detected_smells = [smell_type.value for smell_type in schema.detected_smell_types]

    print(
        "\n[DetectorAgent E2E idle_qubits] line_by_line_expansion:\n"
        + schema.line_by_line_expansion
    )
    print(
        "[DetectorAgent E2E idle_qubits] qubit_operation_analysis:\n"
        + schema.qubit_operation_analysis
    )
    print("[DetectorAgent E2E idle_qubits] report_details:\n" + schema.report_details)
    print("[DetectorAgent E2E idle_qubits] has_smells: " + str(has_smells))
    print("[DetectorAgent E2E idle_qubits] detected_smells: " + str(detected_smells))

    # Simmetrico a test_detector_agent_detects_smell_on_real_example (caso Long Circuit): finora
    # idq-smelly.py era toccato solo da test diagnostici senza assert dedicato (il loop
    # iterativo Refactorer/Reviewer, ora anche il MASEngine e2e) -- mai da un'asserzione vera qui.
    assert has_smells is True
    assert "idle_qubits" in detected_smells


@pytest.mark.e2e
def test_detector_agent_reports_clean_on_fixed_circuit() -> None:
    llm = LLM(model=DETECTOR_MODEL, temperature=0)
    agent = DetectorAgent(llm=llm, facade=QiskitFacade())

    code = _LC_FIXED_PATH.read_text(encoding="utf-8")
    result = agent.detect_smell(code)

    print("\n[DetectorAgent E2E clean] report_details:\n" + result.get_report_details())
    print("[DetectorAgent E2E clean] detected_smells: " + str(result.get_detected_smells()))

    # Circuito gia' corretto (lo smell H-Z-H e' stato sostituito da X): non deve piu' essere
    # segnalato. Se qui esce True, e' un segnale reale sul bias/qualita' del prompt da discutere,
    # NON da nascondere aggiustando il test.
    assert result.get_has_smells() is False
    assert result.get_detected_smells() == []


@pytest.mark.e2e
def test_detector_agent_reports_clean_on_fixed_idle_qubits_circuit() -> None:
    llm = LLM(model=DETECTOR_MODEL, temperature=0)
    agent = DetectorAgent(llm=llm, facade=QiskitFacade())

    code = _IDQ_FIXED_PATH.read_text(encoding="utf-8")
    result = agent.detect_smell(code)

    print("\n[DetectorAgent E2E clean IDQ] report_details:\n" + result.get_report_details())
    print("[DetectorAgent E2E clean IDQ] detected_smells: " + str(result.get_detected_smells()))

    # Versione corretta del caso Idle Qubits: ogni qubit e' ora effettivamente misurato e
    # contribuisce al risultato, quindi non deve piu' essere segnalato. Simmetrico al test su
    # lc-fixed. Se esce True, e' un segnale reale su bias/qualita' da discutere, NON da mascherare.
    assert result.get_has_smells() is False
