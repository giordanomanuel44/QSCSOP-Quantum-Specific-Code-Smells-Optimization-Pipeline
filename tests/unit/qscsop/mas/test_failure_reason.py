import pytest

from qscsop_pipeline.qscsop.mas.dto.failure_reason import FailureReason


@pytest.mark.unit
def test_failure_reason_values_are_the_expected_strings() -> None:
    assert FailureReason.COMPILATION_FAILED.value == "compilation_failed"
    assert FailureReason.NOT_EQUIVALENT.value == "not_equivalent"
    assert FailureReason.METRICS_NOT_IMPROVED.value == "metrics_not_improved"
    assert FailureReason.UNEXPECTED_ERROR.value == "unexpected_error"


@pytest.mark.unit
def test_failure_reason_compares_directly_as_string() -> None:
    # Ereditando da str, i membri sono confrontabili direttamente con la stringa corrispondente.
    assert FailureReason.COMPILATION_FAILED == "compilation_failed"
    assert FailureReason.NOT_EQUIVALENT == "not_equivalent"
    assert FailureReason.METRICS_NOT_IMPROVED == "metrics_not_improved"
    assert FailureReason.UNEXPECTED_ERROR == "unexpected_error"
