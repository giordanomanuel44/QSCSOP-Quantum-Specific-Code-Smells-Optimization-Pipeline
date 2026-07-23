"""Unit test del ReviewerAgent: review() ritorna il feedback dello schema, con mock del Crew."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from crewai import BaseLLM

from qscsop_pipeline.qscsop.mas.agents.reviewer_agent import (
    ReviewerAgent,
    _ReviewSchema,
)
from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO

_RAW_ERROR = "Il circuito refattorizzato non e' funzionalmente equivalente al baseline."
_FEEDBACK = (
    "Il tentativo precedente ha rimosso il qubit inerte ma ha anche cancellato gate legittimi "
    "sugli altri qubit: rimuovi solo il qubit idle e lascia intatto il resto."
)


def _make_agent() -> ReviewerAgent:
    """Costruisce un ReviewerAgent con un LLM finto (l'LLM reale non viene mai usato nei mock)."""
    return ReviewerAgent(llm=Mock(spec=BaseLLM))


def _smell_report() -> SmellReportDTO:
    return SmellReportDTO(
        has_smells=True, report_details="Qubit 2 non contribuisce al risultato: Idle Qubits."
    )


@pytest.mark.unit
def test_review_returns_contextualized_feedback(mocker) -> None:
    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_review_crew",
        return_value=_ReviewSchema(contextualized_feedback=_FEEDBACK),
    )

    result = agent.review(_RAW_ERROR, _smell_report())

    assert result == _FEEDBACK


@pytest.mark.unit
def test_review_propagates_parsing_failure(mocker) -> None:
    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_review_crew",
        side_effect=RuntimeError("output non conforme allo schema"),
    )

    with pytest.raises(RuntimeError, match="output non conforme allo schema"):
        agent.review(_RAW_ERROR, _smell_report())


@pytest.mark.unit
def test_review_invokes_crew_once_with_all_arguments(mocker) -> None:
    agent = _make_agent()
    report = _smell_report()
    run_mock = mocker.patch.object(
        agent,
        "_run_review_crew",
        return_value=_ReviewSchema(contextualized_feedback=_FEEDBACK),
    )

    agent.review(_RAW_ERROR, report)

    run_mock.assert_called_once_with(_RAW_ERROR, report)


@pytest.mark.unit
def test_review_returns_long_feedback_without_truncation(mocker) -> None:
    # Il feedback finisce verbatim nel prompt del tentativo successivo: un traceback lungo deve
    # arrivare intero, senza che review() lo accorci o lo riassuma.
    long_feedback = (
        "Il tentativo precedente ha sollevato la seguente eccezione durante la validazione:\n"
        + "\n".join(
            f'  File "<string>", line {line}, in <module>\n    qc.cx(qreg_q[0], qreg_q[{line}])'
            for line in range(1, 15)
        )
        + "\nIndexError: list index out of range\n"
        "Rimuovi solo il qubit inerte e mantieni intatti tutti i gate degli altri qubit, "
        "aggiornando gli indici usati dalle operazioni a due qubit."
    )
    assert len(long_feedback) > 500

    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_review_crew",
        return_value=_ReviewSchema(contextualized_feedback=long_feedback),
    )

    result = agent.review(_RAW_ERROR, _smell_report())

    assert result == long_feedback


@pytest.mark.unit
def test_review_returns_empty_feedback_verbatim(mocker) -> None:
    # Nessun placeholder di comodo e nessuna eccezione: il chiamante (futuro MASEngine) deve
    # poter osservare esattamente cio' che il modello ha prodotto, stringa vuota inclusa.
    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_review_crew",
        return_value=_ReviewSchema(contextualized_feedback=""),
    )

    result = agent.review(_RAW_ERROR, _smell_report())

    assert result == ""


@pytest.mark.unit
def test_task_prompt_includes_equivalence_hint_only_on_equivalence_errors(mocker) -> None:
    # Il suggerimento sui gate persi per errore deve comparire solo quando l'errore riguarda
    # davvero l'equivalenza funzionale: su un errore di compilazione sarebbe fuorviante.
    agent = _make_agent()
    captured: dict = {}

    class _FakeCrew:
        def __init__(self, agents, tasks, process) -> None:
            captured["task"] = tasks[0]

        def kickoff(self):
            return SimpleNamespace(
                pydantic=_ReviewSchema(contextualized_feedback=_FEEDBACK), raw=""
            )

    mocker.patch("qscsop_pipeline.qscsop.mas.agents.reviewer_agent.Crew", _FakeCrew)

    agent.review(_RAW_ERROR, _smell_report())
    assert "ADDITIONAL HINT FOR THIS SPECIFIC FAILURE" in captured["task"].description

    agent.review("SyntaxError: invalid syntax", _smell_report())
    assert "ADDITIONAL HINT FOR THIS SPECIFIC FAILURE" not in captured["task"].description


@pytest.mark.unit
def test_task_prompt_carries_raw_error_and_original_smell(mocker) -> None:
    # Il Task deve contenere entrambi gli input: l'errore grezzo verbatim e il contesto dello
    # smell che il refactoring stava cercando di correggere.
    agent = _make_agent()
    report = _smell_report()
    captured: dict = {}

    class _FakeCrew:
        def __init__(self, agents, tasks, process) -> None:
            captured["task"] = tasks[0]

        def kickoff(self):
            return SimpleNamespace(
                pydantic=_ReviewSchema(contextualized_feedback=_FEEDBACK), raw=""
            )

    mocker.patch("qscsop_pipeline.qscsop.mas.agents.reviewer_agent.Crew", _FakeCrew)

    agent.review(_RAW_ERROR, report)

    description = captured["task"].description
    assert _RAW_ERROR in description
    assert report.get_report_details() in description
