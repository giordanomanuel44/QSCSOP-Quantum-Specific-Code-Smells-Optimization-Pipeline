import pytest

from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO


@pytest.mark.unit
def test_constructor_and_getters_valid_result() -> None:
    metrics = {
        "abstractMetrics": {"gateCount": 2, "depth": 2},
        "physicalMetrics": {"gateCount": 4, "depth": 3},
    }

    dto = ValidationResultDTO(is_valid=True, raw_error_data=None, new_metrics=metrics)

    assert dto.get_is_valid() is True
    assert dto.get_raw_error_data() is None
    assert dto.get_new_metrics() == metrics


@pytest.mark.unit
def test_constructor_and_getters_invalid_result() -> None:
    dto = ValidationResultDTO(is_valid=False, raw_error_data="SyntaxError: ...", new_metrics=None)

    assert dto.get_is_valid() is False
    assert dto.get_raw_error_data() == "SyntaxError: ..."
    assert dto.get_new_metrics() is None


@pytest.mark.unit
def test_has_no_setters() -> None:
    dto = ValidationResultDTO(is_valid=True, raw_error_data=None, new_metrics={})

    assert not hasattr(dto, "set_is_valid")
    assert not hasattr(dto, "set_raw_error_data")
    assert not hasattr(dto, "set_new_metrics")
