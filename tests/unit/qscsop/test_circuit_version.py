import pytest

from qscsop_pipeline.qscsop.entities.circuit_metrics import CircuitMetrics
from qscsop_pipeline.qscsop.entities.circuit_version import CircuitVersion


def make_version() -> CircuitVersion:
    return CircuitVersion(
        source_code="qc = QuantumCircuit(2)",
        logical_qubits=2,
        abstract_metrics=CircuitMetrics(gate_count=2, depth=2),
        physical_metrics=CircuitMetrics(gate_count=4, depth=3),
    )


@pytest.mark.unit
def test_constructor_and_getters() -> None:
    version = make_version()

    assert version.get_source_code() == "qc = QuantumCircuit(2)"
    assert version.get_logical_qubits() == 2
    assert version.get_abstract_metrics().to_dict() == {"gateCount": 2, "depth": 2}
    assert version.get_physical_metrics().to_dict() == {"gateCount": 4, "depth": 3}


@pytest.mark.unit
def test_setters_update_fields() -> None:
    version = make_version()
    new_abstract = CircuitMetrics(gate_count=9, depth=9)
    new_physical = CircuitMetrics(gate_count=8, depth=8)

    version.set_source_code("qc = QuantumCircuit(3)")
    version.set_logical_qubits(3)
    version.set_abstract_metrics(new_abstract)
    version.set_physical_metrics(new_physical)

    assert version.get_source_code() == "qc = QuantumCircuit(3)"
    assert version.get_logical_qubits() == 3
    assert version.get_abstract_metrics() is new_abstract
    assert version.get_physical_metrics() is new_physical


@pytest.mark.unit
def test_to_dict_produces_nested_structure_as_plain_dicts() -> None:
    version = make_version()

    result = version.to_dict()

    assert result == {
        "sourceCode": "qc = QuantumCircuit(2)",
        "logicalQubits": 2,
        "abstractMetrics": {"gateCount": 2, "depth": 2},
        "physicalMetrics": {"gateCount": 4, "depth": 3},
    }
    assert isinstance(result["abstractMetrics"], dict)
    assert isinstance(result["physicalMetrics"], dict)


@pytest.mark.unit
def test_to_dict_reflects_updates_after_set() -> None:
    version = make_version()

    version.set_logical_qubits(7)
    version.set_abstract_metrics(CircuitMetrics(gate_count=1, depth=1))

    result = version.to_dict()

    assert result["logicalQubits"] == 7
    assert result["abstractMetrics"] == {"gateCount": 1, "depth": 1}
