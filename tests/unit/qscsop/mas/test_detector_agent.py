"""Unit test del DetectorAgent: mappatura output strutturato -> SmellReportDTO, con mock del Crew."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from crewai import BaseLLM

from qscsop_pipeline.qscsop.mas.agents.detector_agent import (
    DetectorAgent,
    _SmellDetectionSchema,
)
from qscsop_pipeline.qscsop.mas.dto.quantum_smell_type import QuantumSmellType
from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO

_SAMPLE_CODE = "qc.h(0)\nqc.z(0)\nqc.h(0)\n"

# tests/unit/qscsop/mas/ -> risali a root repo, poi al file dell'esempio negativo (circuito corretto).
_LC_FIXED_PATH = (
    Path(__file__).resolve().parents[4] / "data" / "raw" / "thesmellyeight" / "lc" / "lc-fixed.py"
)


def _make_agent() -> DetectorAgent:
    """Costruisce un DetectorAgent con un LLM finto (l'LLM reale non viene mai usato nei mock)."""
    return DetectorAgent(llm=Mock(spec=BaseLLM))


@pytest.mark.unit
def test_detect_smell_maps_positive_detection(mocker) -> None:
    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_detection_crew",
        return_value=_SmellDetectionSchema(
            qubit_operation_analysis="q0: h, z, h",
            detected_smell_types=[QuantumSmellType.LONG_CIRCUIT],
            report_details="H-Z-H equivale a X: Long Circuit.",
        ),
    )

    result = agent.detect_smell(_SAMPLE_CODE)

    assert isinstance(result, SmellReportDTO)
    assert result.get_has_smells() is True
    assert result.get_detected_smells() == ["long_circuit"]
    assert result.get_report_details() == "H-Z-H equivale a X: Long Circuit."


@pytest.mark.unit
def test_detect_smell_maps_negative_detection(mocker) -> None:
    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_detection_crew",
        return_value=_SmellDetectionSchema(
            qubit_operation_analysis="q0: h, z, h",
            detected_smell_types=[],
            report_details="",
        ),
    )

    result = agent.detect_smell(_SAMPLE_CODE)

    assert isinstance(result, SmellReportDTO)
    assert result.get_has_smells() is False
    assert result.get_detected_smells() == []
    assert result.get_report_details() == ""


@pytest.mark.unit
def test_detect_smell_maps_both_smell_types(mocker) -> None:
    # Il modello puo' segnalare entrambi gli smell contemporaneamente: la lista li deve riportare
    # tutti come stringhe semplici.
    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_detection_crew",
        return_value=_SmellDetectionSchema(
            qubit_operation_analysis="q0..q2 analizzati",
            detected_smell_types=[
                QuantumSmellType.LONG_CIRCUIT,
                QuantumSmellType.IDLE_QUBITS,
            ],
            report_details="Entrambi gli smell presenti.",
        ),
    )

    result = agent.detect_smell(_SAMPLE_CODE)

    assert result.get_has_smells() is True
    detected = result.get_detected_smells()
    assert "long_circuit" in detected
    assert "idle_qubits" in detected


@pytest.mark.unit
def test_detect_smell_propagates_parsing_failure(mocker) -> None:
    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_detection_crew",
        side_effect=RuntimeError("output non conforme allo schema"),
    )

    with pytest.raises(RuntimeError, match="output non conforme allo schema"):
        agent.detect_smell(_SAMPLE_CODE)


@pytest.mark.unit
def test_detect_smell_invokes_crew_once_with_code(mocker) -> None:
    agent = _make_agent()
    run_mock = mocker.patch.object(
        agent,
        "_run_detection_crew",
        return_value=_SmellDetectionSchema(
            qubit_operation_analysis="q0: h, z, h",
            detected_smell_types=[],
            report_details="",
        ),
    )

    agent.detect_smell(_SAMPLE_CODE)

    run_mock.assert_called_once_with(_SAMPLE_CODE)


@pytest.mark.unit
def test_task_prompt_includes_negative_lc_fixed_example(mocker) -> None:
    # Verifica che il Task realmente costruito da _run_detection_crew contenga l'esempio
    # negativo (lc-fixed.py): rende esplicito che il few-shot non e' piu' solo positivo.
    agent = _make_agent()

    captured: dict = {}

    class _FakeCrew:
        def __init__(self, agents, tasks, process) -> None:
            captured["task"] = tasks[0]

        def kickoff(self):
            return SimpleNamespace(
                pydantic=_SmellDetectionSchema(
                    qubit_operation_analysis="q0: h, z, h",
                    detected_smell_types=[],
                    report_details="",
                ),
                raw="",
            )

    # Intercetta la creazione del Crew per catturare il Task senza chiamare un LLM reale.
    mocker.patch("qscsop_pipeline.qscsop.mas.agents.detector_agent.Crew", _FakeCrew)

    agent.detect_smell(_SAMPLE_CODE)

    lc_fixed_content = _LC_FIXED_PATH.read_text(encoding="utf-8")
    assert lc_fixed_content in captured["task"].description


@pytest.mark.unit
def test_reasoning_field_stays_internal_and_is_not_exposed_in_dto(mocker) -> None:
    # Il chain-of-thought (qubit_operation_analysis) migliora il ragionamento interno del modello
    # ma non deve trapelare nel DTO pubblico, che resta limitato a has_smells + report_details.
    agent = _make_agent()
    reasoning = "q0: h, p, z, s, measure -> usato; nessun qubit idle"
    mocker.patch.object(
        agent,
        "_run_detection_crew",
        return_value=_SmellDetectionSchema(
            qubit_operation_analysis=reasoning,
            detected_smell_types=[],
            report_details="Circuito pulito.",
        ),
    )

    result = agent.detect_smell(_SAMPLE_CODE)

    # Il DTO non deve avere alcun attributo/campo che contenga il ragionamento interno.
    assert not hasattr(result, "qubit_operation_analysis")
    assert not hasattr(result, "get_qubit_operation_analysis")
    assert reasoning not in result.get_report_details()
    assert result.get_report_details() == "Circuito pulito."
    assert result.get_has_smells() is False
