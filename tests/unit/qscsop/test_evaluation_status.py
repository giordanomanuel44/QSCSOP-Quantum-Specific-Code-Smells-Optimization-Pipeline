import pytest

from qscsop_pipeline.qscsop.entities.evaluation_status import EvaluationStatus


@pytest.mark.unit
def test_status_values_are_the_expected_strings() -> None:
    assert EvaluationStatus.PROCESSING.value == "PROCESSING"
    assert EvaluationStatus.SMELL_FREE.value == "SMELL_FREE"
    assert EvaluationStatus.OPTIMIZED.value == "OPTIMIZED"
    assert EvaluationStatus.OPT_FAILED.value == "OPT_FAILED"


@pytest.mark.unit
def test_status_compares_directly_as_string() -> None:
    # Ereditando da str, i membri sono confrontabili direttamente con la stringa corrispondente.
    assert EvaluationStatus.PROCESSING == "PROCESSING"
    assert EvaluationStatus.SMELL_FREE == "SMELL_FREE"
    assert EvaluationStatus.OPTIMIZED == "OPTIMIZED"
    assert EvaluationStatus.OPT_FAILED == "OPT_FAILED"
