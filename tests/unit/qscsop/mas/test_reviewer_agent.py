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
from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO

_RAW_ERROR = "Il circuito refattorizzato non e' funzionalmente equivalente al baseline."
_FEEDBACK = (
    "Il tentativo precedente ha rimosso il qubit inerte ma ha anche cancellato gate legittimi "
    "sugli altri qubit: rimuovi solo il qubit idle e lascia intatto il resto."
)
_FAILED_CODE = "qc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\n"


def _make_agent() -> ReviewerAgent:
    """Costruisce un ReviewerAgent con un LLM finto (l'LLM reale non viene mai usato nei mock)."""
    return ReviewerAgent(llm=Mock(spec=BaseLLM))


def _smell_report() -> SmellReportDTO:
    return SmellReportDTO(
        has_smells=True, report_details="Qubit 2 non contribuisce al risultato: Idle Qubits."
    )


def _equivalence_failure(raw_error_details: str = _RAW_ERROR) -> ValidationResultDTO:
    """Fallimento di equivalenza (o compilazione): new_metrics=None, come costruisce ValidationService."""
    return ValidationResultDTO(is_valid=False, raw_error_data=raw_error_details, new_metrics=None)


def _metrics_not_improved_failure() -> ValidationResultDTO:
    """Fallimento del nodo "Migliori?": new_metrics popolato, e' il segnale strutturale."""
    return ValidationResultDTO(
        is_valid=False,
        raw_error_data="Il refactoring non ha ridotto nessuna delle due metriche di smell.",
        new_metrics={
            "longCircuit": {"maxOpsPerQubit": 7, "maxParallelOps": 5, "value": 35},
            "idleQubits": {"value": 3, "worstQubit": 0},
        },
    )


def _metrics_no_improvement_at_all_failure() -> ValidationResultDTO:
    """CASE 1: ogni valore del refactored e' identico alla baseline, nessuna dimensione migliora."""
    return ValidationResultDTO(
        is_valid=False,
        raw_error_data=(
            "Il refactoring non ha ridotto nessuna delle due metriche di smell rispetto alla "
            "baseline. Baseline: l*c=35, IdQ=3. Refactored: l*c=35, IdQ=3."
        ),
        new_metrics={
            "longCircuit": {"maxOpsPerQubit": 7, "maxParallelOps": 5, "value": 35},
            "idleQubits": {"value": 3, "worstQubit": 0},
        },
    )


def _metrics_near_miss_failure() -> ValidationResultDTO:
    """CASE 2: IdQ migliora ma l*c peggiora -- il baratto fra i due smell, vietato da Pareto."""
    return ValidationResultDTO(
        is_valid=False,
        raw_error_data=(
            "Il refactoring non ha ridotto nessuna delle due metriche di smell rispetto alla "
            "baseline. Baseline: l*c=35, IdQ=3. Refactored: l*c=42, IdQ=0."
        ),
        new_metrics={
            "longCircuit": {"maxOpsPerQubit": 7, "maxParallelOps": 5, "value": 35},
            "idleQubits": {"value": 3, "worstQubit": 0},
        },
    )


@pytest.mark.unit
def test_review_returns_contextualized_feedback(mocker) -> None:
    agent = _make_agent()
    mocker.patch.object(
        agent,
        "_run_review_crew",
        return_value=_ReviewSchema(contextualized_feedback=_FEEDBACK),
    )

    result = agent.review(_equivalence_failure(), _smell_report(), _FAILED_CODE)

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
        agent.review(_equivalence_failure(), _smell_report(), _FAILED_CODE)


@pytest.mark.unit
def test_review_invokes_crew_once_with_all_arguments(mocker) -> None:
    agent = _make_agent()
    report = _smell_report()
    validation_result = _equivalence_failure()
    run_mock = mocker.patch.object(
        agent,
        "_run_review_crew",
        return_value=_ReviewSchema(contextualized_feedback=_FEEDBACK),
    )

    agent.review(validation_result, report, _FAILED_CODE)

    run_mock.assert_called_once_with(validation_result, report, _FAILED_CODE)


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

    result = agent.review(_equivalence_failure(), _smell_report(), _FAILED_CODE)

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

    result = agent.review(_equivalence_failure(), _smell_report(), _FAILED_CODE)

    assert result == ""


def _capture_task(mocker, agent: ReviewerAgent) -> dict:
    """Intercetta la creazione del Crew per catturare il Task senza chiamare un LLM reale."""
    captured: dict = {}

    class _FakeCrew:
        def __init__(self, agents, tasks, process) -> None:
            captured["task"] = tasks[0]

        def kickoff(self):
            return SimpleNamespace(
                pydantic=_ReviewSchema(contextualized_feedback=_FEEDBACK), raw=""
            )

    mocker.patch("qscsop_pipeline.qscsop.mas.agents.reviewer_agent.Crew", _FakeCrew)
    return captured


@pytest.mark.unit
def test_task_prompt_includes_equivalence_hint_only_on_equivalence_errors(mocker) -> None:
    # Il suggerimento sui gate persi per errore deve comparire solo quando l'errore riguarda
    # davvero l'equivalenza funzionale: su un errore di compilazione sarebbe fuorviante.
    agent = _make_agent()
    captured = _capture_task(mocker, agent)

    agent.review(_equivalence_failure(), _smell_report(), _FAILED_CODE)
    assert "ADDITIONAL HINT FOR THIS SPECIFIC FAILURE" in captured["task"].description
    assert "FUNCTIONAL EQUIVALENCE" in captured["task"].description

    agent.review(_equivalence_failure("SyntaxError: invalid syntax"), _smell_report(), _FAILED_CODE)
    assert "ADDITIONAL HINT FOR THIS SPECIFIC FAILURE" not in captured["task"].description


@pytest.mark.unit
def test_task_prompt_carries_raw_error_and_original_smell(mocker) -> None:
    # Il Task deve contenere entrambi gli input: l'errore grezzo verbatim e il contesto dello
    # smell che il refactoring stava cercando di correggere.
    agent = _make_agent()
    report = _smell_report()
    captured = _capture_task(mocker, agent)

    agent.review(_equivalence_failure(), report, _FAILED_CODE)

    description = captured["task"].description
    assert _RAW_ERROR in description
    assert report.get_report_details() in description


@pytest.mark.unit
def test_task_prompt_includes_failed_code(mocker) -> None:
    # Il codice del tentativo appena fallito deve arrivare nel prompt, cosi' il feedback puo'
    # riferirsi concretamente a cosa il tentativo ha fatto invece di restare astratto.
    agent = _make_agent()
    captured = _capture_task(mocker, agent)

    agent.review(_equivalence_failure(), _smell_report(), _FAILED_CODE)

    description = captured["task"].description
    assert "THE CODE THAT WAS ATTEMPTED" in description
    assert _FAILED_CODE in description


@pytest.mark.unit
def test_task_prompt_includes_metrics_not_improved_hint_when_new_metrics_is_populated(
    mocker,
) -> None:
    # Quando new_metrics NON e' None (fallimento del nodo "Migliori?"), va iniettato
    # _METRICS_NOT_IMPROVED_HINT_SECTION, mai _EQUIVALENCE_HINT_SECTION.
    agent = _make_agent()
    captured = _capture_task(mocker, agent)

    agent.review(_metrics_not_improved_failure(), _smell_report(), _FAILED_CODE)

    description = captured["task"].description
    assert "did not satisfy the improvement criterion" in description
    assert "FUNCTIONAL EQUIVALENCE" not in description


@pytest.mark.unit
def test_task_prompt_metrics_hint_covers_case_1_no_improvement_at_all(mocker) -> None:
    # raw_error_data riflette nessuna metrica migliorata: il ramo CASE 1 dell'istruzione deve
    # essere presente nel prompt (il modello lo sceglie da solo leggendo i numeri, non e' materia
    # di questo test unitario -- vedi test e2e per quello).
    agent = _make_agent()
    captured = _capture_task(mocker, agent)

    agent.review(_metrics_no_improvement_at_all_failure(), _smell_report(), _FAILED_CODE)

    description = captured["task"].description
    assert "CASE 1" in description


@pytest.mark.unit
def test_task_prompt_metrics_hint_covers_case_2_near_miss(mocker) -> None:
    # raw_error_data riflette un "near miss" (una metrica peggiorata, altre migliorate): il blocco
    # hint completo (che include entrambi i rami, CASE 1 e CASE 2) deve comunque arrivare nel
    # prompt -- non si testa quale ramo il modello sceglie di applicare.
    agent = _make_agent()
    captured = _capture_task(mocker, agent)

    agent.review(_metrics_near_miss_failure(), _smell_report(), _FAILED_CODE)

    description = captured["task"].description
    assert "ADDITIONAL HINT FOR THIS SPECIFIC FAILURE" in description
    assert "CASE 2" in description


@pytest.mark.unit
def test_task_prompt_includes_equivalence_hint_when_new_metrics_is_none_and_error_mentions_it(
    mocker,
) -> None:
    # new_metrics=None e raw_error_data che menziona "equivalen...": comportamento esistente,
    # ora guidato anche dal controllo strutturale su new_metrics=None (non solo dalla sottostringa).
    agent = _make_agent()
    captured = _capture_task(mocker, agent)

    agent.review(_equivalence_failure(), _smell_report(), _FAILED_CODE)

    description = captured["task"].description
    assert "FUNCTIONAL EQUIVALENCE" in description
    assert "did not satisfy the improvement criterion" not in description


@pytest.mark.unit
def test_task_prompt_injects_no_special_hint_on_compilation_failure(mocker) -> None:
    # new_metrics=None e raw_error_data che sembra un errore di compilazione: nessuno dei due
    # hint speciali deve comparire.
    agent = _make_agent()
    captured = _capture_task(mocker, agent)

    compilation_failure = _equivalence_failure(
        'Traceback (most recent call last):\n  File "<string>", line 3\nSyntaxError: invalid syntax'
    )
    agent.review(compilation_failure, _smell_report(), _FAILED_CODE)

    description = captured["task"].description
    assert "FUNCTIONAL EQUIVALENCE" not in description
    assert "did not satisfy the improvement criterion" not in description
