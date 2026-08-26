"""Unit test dell'entità che porta la misura QSMELL di una versione di circuito.

Sostituisce test_circuit_metrics.py, sparito con CircuitMetrics. La differenza che conta e' il
test sull'invariante di long_circuit: e' la proprieta' che giustifica la scelta di memorizzare
l e c invece del loro prodotto.
"""

import pytest

from qscsop_pipeline.qscsop.entities.smell_metrics import SmellMetrics

# Forma misurata su data/raw/bugs4q/Terra-0-4000_10_fix.py, l'unico circuito reale affidabile
# che porta entrambi gli smell: l = 7, c = 5, IdQ = 3.
_L, _C, _IDQ = 7, 5, 3


def _metrics() -> SmellMetrics:
    return SmellMetrics(max_ops_per_qubit=_L, max_parallel_ops=_C, idle_qubits=_IDQ)


@pytest.mark.unit
def test_constructor_and_getters() -> None:
    metrics = _metrics()

    assert metrics.get_max_ops_per_qubit() == _L
    assert metrics.get_max_parallel_ops() == _C
    assert metrics.get_idle_qubits() == _IDQ


@pytest.mark.unit
def test_setters_update_fields() -> None:
    metrics = _metrics()

    metrics.set_max_ops_per_qubit(4)
    metrics.set_max_parallel_ops(6)
    metrics.set_idle_qubits(0)

    assert metrics.get_max_ops_per_qubit() == 4
    assert metrics.get_max_parallel_ops() == 6
    assert metrics.get_idle_qubits() == 0


@pytest.mark.unit
def test_long_circuit_is_always_the_product_and_never_stored() -> None:
    """L'INVARIANTE che giustifica la property invece di un quarto campo.

    Se long_circuit fosse memorizzato, esisterebbe uno stato in cui vale qualcosa di diverso da
    l * c -- un'incoerenza che nessun controllo puo' impedire. Derivandolo ad ogni accesso quello
    stato non e' rappresentabile: qui si cambiano i due fattori e il prodotto segue da solo.
    """
    metrics = _metrics()
    assert metrics.long_circuit == _L * _C

    metrics.set_max_ops_per_qubit(3)
    assert metrics.long_circuit == 3 * _C

    metrics.set_max_parallel_ops(2)
    assert metrics.long_circuit == 3 * 2

    assert not hasattr(SmellMetrics, "set_long_circuit")


@pytest.mark.unit
def test_long_circuit_is_read_only() -> None:
    """Assegnarlo deve fallire: senza setter, la property e' l'unica via di accesso."""
    with pytest.raises(AttributeError):
        _metrics().long_circuit = 99


@pytest.mark.unit
def test_to_dict_carries_the_three_measures_plus_the_derived_one() -> None:
    assert _metrics().to_dict() == {
        "maxOpsPerQubit": 7,
        "maxParallelOps": 5,
        "longCircuit": 35,
        "idleQubits": 3,
    }


@pytest.mark.unit
def test_to_dict_reflects_updates_after_set() -> None:
    metrics = _metrics()

    metrics.set_max_ops_per_qubit(2)
    metrics.set_idle_qubits(0)

    result = metrics.to_dict()
    assert result["maxOpsPerQubit"] == 2
    assert result["longCircuit"] == 2 * _C
    assert result["idleQubits"] == 0
