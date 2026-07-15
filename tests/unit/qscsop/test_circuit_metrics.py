import pytest

from qscsop_pipeline.qscsop.entities.circuit_metrics import CircuitMetrics


@pytest.mark.unit
def test_constructor_and_getters() -> None:
    metrics = CircuitMetrics(gate_count=10, depth=5)

    assert metrics.get_gate_count() == 10
    assert metrics.get_depth() == 5


@pytest.mark.unit
def test_setters_update_fields() -> None:
    metrics = CircuitMetrics(gate_count=10, depth=5)

    metrics.set_gate_count(20)
    metrics.set_depth(8)

    assert metrics.get_gate_count() == 20
    assert metrics.get_depth() == 8


@pytest.mark.unit
def test_to_dict_produces_expected_keys() -> None:
    metrics = CircuitMetrics(gate_count=10, depth=5)

    assert metrics.to_dict() == {"gateCount": 10, "depth": 5}


@pytest.mark.unit
def test_to_dict_reflects_updates_after_set() -> None:
    metrics = CircuitMetrics(gate_count=10, depth=5)

    metrics.set_gate_count(1)
    metrics.set_depth(2)

    assert metrics.to_dict() == {"gateCount": 1, "depth": 2}
