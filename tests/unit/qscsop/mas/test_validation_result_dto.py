import pytest

from qscsop_pipeline.qscsop.mas.dto.failure_reason import FailureReason
from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO


@pytest.mark.unit
def test_constructor_and_getters_valid_result() -> None:
    metrics = {
        "longCircuit": {"maxOpsPerQubit": 2, "maxParallelOps": 2, "value": 4},
        "idleQubits": {"value": 0, "worstQubit": None},
    }

    dto = ValidationResultDTO(is_valid=True, raw_error_data=None, new_metrics=metrics)

    assert dto.get_is_valid() is True
    assert dto.get_raw_error_data() is None
    assert dto.get_new_metrics() == metrics
    assert dto.get_failure_reason() is None


@pytest.mark.unit
def test_constructor_and_getters_invalid_result() -> None:
    dto = ValidationResultDTO(is_valid=False, raw_error_data="SyntaxError: ...", new_metrics=None)

    assert dto.get_is_valid() is False
    assert dto.get_raw_error_data() == "SyntaxError: ..."
    assert dto.get_new_metrics() is None
    assert dto.get_failure_reason() is None


@pytest.mark.unit
def test_constructor_accepts_explicit_failure_reason() -> None:
    dto = ValidationResultDTO(
        is_valid=False,
        raw_error_data="SyntaxError: ...",
        new_metrics=None,
        failure_reason=FailureReason.COMPILATION_FAILED,
    )

    assert dto.get_failure_reason() == FailureReason.COMPILATION_FAILED


@pytest.mark.unit
def test_set_failure_reason_updates_field() -> None:
    dto = ValidationResultDTO(is_valid=True, raw_error_data=None, new_metrics={})

    dto.set_failure_reason(FailureReason.NOT_EQUIVALENT)

    assert dto.get_failure_reason() == FailureReason.NOT_EQUIVALENT


@pytest.mark.unit
def test_set_is_valid_updates_field() -> None:
    dto = ValidationResultDTO(is_valid=True, raw_error_data=None, new_metrics={})

    dto.set_is_valid(False)

    assert dto.get_is_valid() is False


@pytest.mark.unit
def test_set_raw_error_data_updates_field() -> None:
    dto = ValidationResultDTO(is_valid=True, raw_error_data=None, new_metrics={})

    dto.set_raw_error_data("nuovo errore")

    assert dto.get_raw_error_data() == "nuovo errore"


@pytest.mark.unit
def test_set_new_metrics_updates_field() -> None:
    dto = ValidationResultDTO(is_valid=False, raw_error_data="errore", new_metrics=None)
    metrics = {
        "longCircuit": {"maxOpsPerQubit": 1, "maxParallelOps": 1, "value": 1},
        "idleQubits": {"value": 0, "worstQubit": None},
    }

    dto.set_new_metrics(metrics)

    assert dto.get_new_metrics() == metrics


@pytest.mark.unit
def test_set_new_metrics_is_not_affected_by_later_mutation_of_the_original_dict() -> None:
    dto = ValidationResultDTO(is_valid=False, raw_error_data="errore", new_metrics=None)
    metrics = {
        "longCircuit": {"maxOpsPerQubit": 1, "maxParallelOps": 1, "value": 1},
        "idleQubits": {"value": 0, "worstQubit": None},
    }

    dto.set_new_metrics(metrics)
    metrics["longCircuit"]["value"] = 999
    metrics["newKey"] = "unexpected"

    assert dto.get_new_metrics() == {
        "longCircuit": {"maxOpsPerQubit": 1, "maxParallelOps": 1, "value": 1},
        "idleQubits": {"value": 0, "worstQubit": None},
    }


@pytest.mark.unit
def test_get_new_metrics_returns_a_copy_not_a_shared_reference() -> None:
    metrics = {
        "longCircuit": {"maxOpsPerQubit": 1, "maxParallelOps": 1, "value": 1},
        "idleQubits": {"value": 0, "worstQubit": None},
    }
    dto = ValidationResultDTO(is_valid=True, raw_error_data=None, new_metrics=metrics)

    retrieved = dto.get_new_metrics()
    retrieved["longCircuit"]["value"] = 999

    assert dto.get_new_metrics() == {
        "longCircuit": {"maxOpsPerQubit": 1, "maxParallelOps": 1, "value": 1},
        "idleQubits": {"value": 0, "worstQubit": None},
    }
