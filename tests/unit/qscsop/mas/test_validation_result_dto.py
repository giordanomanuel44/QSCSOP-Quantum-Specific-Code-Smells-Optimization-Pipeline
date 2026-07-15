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
        "abstractMetrics": {"gateCount": 1, "depth": 1},
        "physicalMetrics": {"gateCount": 2, "depth": 2},
    }

    dto.set_new_metrics(metrics)

    assert dto.get_new_metrics() == metrics


@pytest.mark.unit
def test_set_new_metrics_is_not_affected_by_later_mutation_of_the_original_dict() -> None:
    dto = ValidationResultDTO(is_valid=False, raw_error_data="errore", new_metrics=None)
    metrics = {
        "abstractMetrics": {"gateCount": 1, "depth": 1},
        "physicalMetrics": {"gateCount": 2, "depth": 2},
    }

    dto.set_new_metrics(metrics)
    metrics["abstractMetrics"]["gateCount"] = 999
    metrics["newKey"] = "unexpected"

    assert dto.get_new_metrics() == {
        "abstractMetrics": {"gateCount": 1, "depth": 1},
        "physicalMetrics": {"gateCount": 2, "depth": 2},
    }


@pytest.mark.unit
def test_get_new_metrics_returns_a_copy_not_a_shared_reference() -> None:
    metrics = {
        "abstractMetrics": {"gateCount": 1, "depth": 1},
        "physicalMetrics": {"gateCount": 2, "depth": 2},
    }
    dto = ValidationResultDTO(is_valid=True, raw_error_data=None, new_metrics=metrics)

    retrieved = dto.get_new_metrics()
    retrieved["abstractMetrics"]["gateCount"] = 999

    assert dto.get_new_metrics() == {
        "abstractMetrics": {"gateCount": 1, "depth": 1},
        "physicalMetrics": {"gateCount": 2, "depth": 2},
    }
