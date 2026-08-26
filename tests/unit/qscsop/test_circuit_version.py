"""Unit test della versione di circuito (baseline o refactored)."""

import pytest

from qscsop_pipeline.qscsop.entities.circuit_version import CircuitVersion
from qscsop_pipeline.qscsop.entities.smell_metrics import SmellMetrics

_SOURCE = "qc = QuantumCircuit(2)"


def _version() -> CircuitVersion:
    return CircuitVersion(
        source_code=_SOURCE,
        smell_metrics=SmellMetrics(max_ops_per_qubit=7, max_parallel_ops=5, idle_qubits=3),
    )


@pytest.mark.unit
def test_constructor_and_getters() -> None:
    version = _version()

    assert version.get_source_code() == _SOURCE
    assert version.get_smell_metrics().long_circuit == 35


@pytest.mark.unit
def test_setters_update_fields() -> None:
    version = _version()
    replacement = SmellMetrics(max_ops_per_qubit=1, max_parallel_ops=1, idle_qubits=0)

    version.set_source_code("qc = QuantumCircuit(3)")
    version.set_smell_metrics(replacement)

    assert version.get_source_code() == "qc = QuantumCircuit(3)"
    assert version.get_smell_metrics() is replacement


@pytest.mark.unit
def test_to_dict_nests_the_measure_as_a_plain_dict() -> None:
    result = _version().to_dict()

    assert result == {
        "sourceCode": _SOURCE,
        "smellMetrics": {
            "maxOpsPerQubit": 7,
            "maxParallelOps": 5,
            "longCircuit": 35,
            "idleQubits": 3,
        },
    }
    assert isinstance(result["smellMetrics"], dict)


@pytest.mark.unit
def test_to_dict_carries_no_cost_metrics_any_more() -> None:
    """Regressione: le chiavi rimosse non devono rientrare dalla finestra.

    logicalQubits, abstractMetrics e physicalMetrics sono uscite dal contratto dati con questo
    refactoring; se un giorno ricomparissero, il record tornerebbe a portare misure che nessuno
    consuma e che sono cieche ai refactoring della pipeline.
    """
    result = _version().to_dict()

    assert set(result) == {"sourceCode", "smellMetrics"}


@pytest.mark.unit
def test_to_dict_reflects_updates_after_set() -> None:
    version = _version()

    version.set_smell_metrics(SmellMetrics(max_ops_per_qubit=4, max_parallel_ops=2, idle_qubits=1))

    assert version.to_dict()["smellMetrics"] == {
        "maxOpsPerQubit": 4,
        "maxParallelOps": 2,
        "longCircuit": 8,
        "idleQubits": 1,
    }
