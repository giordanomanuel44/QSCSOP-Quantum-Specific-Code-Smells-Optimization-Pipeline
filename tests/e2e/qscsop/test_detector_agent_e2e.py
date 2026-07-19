"""E2E del DetectorAgent con LLM reale (Ollama). MAI in CI: richiede il modello caricato in locale."""

from pathlib import Path

import pytest
from crewai import LLM

from qscsop_pipeline.qscsop.mas.agents.detector_agent import DetectorAgent

# tests/e2e/qscsop/ -> risali a root repo, poi ai file di esempio (smell noto / circuito corretto).
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "thesmellyeight"
_LC_SMELLY_PATH = _DATA_DIR / "lc" / "lc-smelly.py"
_LC_FIXED_PATH = _DATA_DIR / "lc" / "lc-fixed.py"
_IDQ_FIXED_PATH = _DATA_DIR / "idq" / "idq-fixed.py"


@pytest.mark.e2e
def test_detector_agent_detects_smell_on_real_example() -> None:
    llm = LLM(model="ollama/qwen2.5-coder:7b", temperature=0)
    agent = DetectorAgent(llm=llm)

    code = _LC_SMELLY_PATH.read_text(encoding="utf-8")
    result = agent.detect_smell(code)

    print("\n[DetectorAgent E2E] report_details:\n" + result.get_report_details())

    # Il modello dovrebbe riconoscere lo smell nel suo stesso esempio few-shot; se fallisce,
    # e' un segnale utile sulla qualita' del prompt/modello da rivedere.
    assert result.get_has_smells() is True


@pytest.mark.e2e
def test_detector_agent_reports_clean_on_fixed_circuit() -> None:
    llm = LLM(model="ollama/qwen2.5-coder:7b", temperature=0)
    agent = DetectorAgent(llm=llm)

    code = _LC_FIXED_PATH.read_text(encoding="utf-8")
    result = agent.detect_smell(code)

    print("\n[DetectorAgent E2E clean] report_details:\n" + result.get_report_details())

    # Circuito gia' corretto (lo smell H-Z-H e' stato sostituito da X): non deve piu' essere
    # segnalato. Se qui esce True, e' un segnale reale sul bias/qualita' del prompt da discutere,
    # NON da nascondere aggiustando il test.
    assert result.get_has_smells() is False


@pytest.mark.e2e
def test_detector_agent_reports_clean_on_fixed_idle_qubits_circuit() -> None:
    llm = LLM(model="ollama/qwen2.5-coder:7b", temperature=0)
    agent = DetectorAgent(llm=llm)

    code = _IDQ_FIXED_PATH.read_text(encoding="utf-8")
    result = agent.detect_smell(code)

    print("\n[DetectorAgent E2E clean IDQ] report_details:\n" + result.get_report_details())

    # Versione corretta del caso Idle Qubits: ogni qubit e' ora effettivamente misurato e
    # contribuisce al risultato, quindi non deve piu' essere segnalato. Simmetrico al test su
    # lc-fixed. Se esce True, e' un segnale reale su bias/qualita' da discutere, NON da mascherare.
    assert result.get_has_smells() is False
