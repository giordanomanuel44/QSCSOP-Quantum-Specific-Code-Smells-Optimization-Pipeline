"""Unit test del RefactorerAgent: refactor() ritorna il codice dello schema, con mock del Crew."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from crewai import BaseLLM

from qscsop_pipeline.qscsop.mas.agents.refactorer_agent import (
    _IDQ_FIXED_EXAMPLE,
    _IDQ_SMELLY_EXAMPLE,
    RefactorerAgent,
    _RefactorSchema,
)
from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO

_SAMPLE_CODE = "qc.h(0)\nqc.z(0)\nqc.h(0)\n"
_REFACTORED_CODE = "qc.x(0)\n"


def _make_agent() -> RefactorerAgent:
    """Costruisce un RefactorerAgent con un LLM finto (l'LLM reale non viene mai usato nei mock)."""
    return RefactorerAgent(llm=Mock(spec=BaseLLM))


def _smell_report() -> SmellReportDTO:
    return SmellReportDTO(has_smells=True, report_details="H-Z-H equivale a X: Long Circuit.")


@pytest.mark.unit
def test_refactor_returns_refactored_code(mocker) -> None:
    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_refactor_crew",
        return_value=_RefactorSchema(refactored_code=_REFACTORED_CODE),
    )

    result = agent.refactor(_SAMPLE_CODE, _smell_report(), review_feedback="")

    assert result == _REFACTORED_CODE


@pytest.mark.unit
def test_refactor_propagates_parsing_failure(mocker) -> None:
    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_refactor_crew",
        side_effect=RuntimeError("output non conforme allo schema"),
    )

    with pytest.raises(RuntimeError, match="output non conforme allo schema"):
        agent.refactor(_SAMPLE_CODE, _smell_report(), review_feedback="")


@pytest.mark.unit
def test_refactor_invokes_crew_once_with_all_arguments(mocker) -> None:
    agent = _make_agent()
    report = _smell_report()
    run_mock = mocker.patch.object(
        agent,
        "_run_refactor_crew",
        return_value=_RefactorSchema(refactored_code=_REFACTORED_CODE),
    )

    agent.refactor(_SAMPLE_CODE, report, review_feedback="qualcosa non va")

    run_mock.assert_called_once_with(_SAMPLE_CODE, report, "qualcosa non va")


@pytest.mark.unit
def test_refactor_passes_empty_feedback_on_first_iteration(mocker) -> None:
    # Prima iterazione: review_feedback deve arrivare come stringa vuota, non None, non omesso.
    agent = _make_agent()
    report = _smell_report()
    run_mock = mocker.patch.object(
        agent,
        "_run_refactor_crew",
        return_value=_RefactorSchema(refactored_code=_REFACTORED_CODE),
    )

    agent.refactor(_SAMPLE_CODE, report, review_feedback="")

    _, kwargs = run_mock.call_args
    passed_feedback = run_mock.call_args.args[2] if not kwargs else kwargs["review_feedback"]
    assert passed_feedback == ""
    assert passed_feedback is not None


@pytest.mark.unit
def test_task_prompt_makes_the_agent_an_executor_not_an_analyst(mocker) -> None:
    """Il RefactorerAgent non ri-diagnostica: applica la prescrizione del Detector.

    Il suo prompt conteneva il ragionamento sulla ridondanza semantica, che ora vive nel
    DetectorAgent: lasciarlo qui significherebbe due agenti che fanno la stessa analisi e
    possono contraddirsi. Sono spariti anche tre few-shot su quattro, perche' misurati con la
    facade insegnavano a riparare NON-smell (la coppia Long Circuit parte da l*c = 3, quella
    illustrativa di Idle Qubits da IdQ = 0).
    """
    agent = _make_agent()

    captured: dict = {}

    class _FakeCrew:
        def __init__(self, agents, tasks, process) -> None:
            captured["task"] = tasks[0]

        def kickoff(self):
            return SimpleNamespace(
                pydantic=_RefactorSchema(refactored_code=_REFACTORED_CODE), raw=""
            )

    mocker.patch("qscsop_pipeline.qscsop.mas.agents.refactorer_agent.Crew", _FakeCrew)

    agent.refactor(_SAMPLE_CODE, _smell_report(), review_feedback="")

    description = captured["task"].description
    # Resta l'unica coppia valida: stesse operazioni riordinate, IdQ 7 -> 0 misurato.
    assert _IDQ_SMELLY_EXAMPLE in description
    assert _IDQ_FIXED_EXAMPLE in description
    # Il permesso di non fare nulla: senza, l'agente inventa una modifica per sembrare produttivo
    # e il circuito smette di essere equivalente.
    assert "return the circuit UNCHANGED" in description
    # Rimuovere un qubit non e' mai il fix per un'attesa: un qubit mai usato ha IdQ = 0.
    assert "Removing a qubit is NEVER the fix" in description


@pytest.mark.unit
def test_refactor_passes_nonempty_feedback_intact(mocker) -> None:
    # Iterazione successiva: il feedback specifico deve arrivare intatto a _run_refactor_crew.
    agent = _make_agent()
    report = _smell_report()
    feedback = "Il circuito refattorizzato non e' equivalente all'originale."
    run_mock = mocker.patch.object(
        agent,
        "_run_refactor_crew",
        return_value=_RefactorSchema(refactored_code=_REFACTORED_CODE),
    )

    agent.refactor(_SAMPLE_CODE, report, review_feedback=feedback)

    assert run_mock.call_args.args[2] == feedback
