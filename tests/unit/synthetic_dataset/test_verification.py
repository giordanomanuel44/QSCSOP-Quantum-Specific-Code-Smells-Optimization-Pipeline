"""Unit test delle verifiche del tooling di generazione sintetica.

La facade e' sempre MOCKATA (spec=IQiskitFacade): qui si verifica la logica di decisione delle
funzioni di verification.py -- come traducono in etichette i numeri che la facade ritorna --
non il comportamento di Qiskit, gia' coperto dai test su QiskitFacade.

I test sulle vecchie verify_declared_simplification / verify_declared_idle_qubits sono spariti
insieme alle funzioni: verificavano che una DICHIARAZIONE del generatore reggesse, e il
generatore non dichiara piu' nulla (vedi prompts.GeneratedCircuit). Al loro posto ci sono i test
su measure_shape, che e' il punto in cui l'etichetta viene ora prodotta.

Sono spariti anche i test su matches_batch_theme, insieme alla funzione: i lotti non chiedono
piu' una forma metrica da confrontare (vedi il docstring di prompts.py).
"""

from unittest.mock import Mock

import pytest

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qscsop.mas.detection_thresholds import LC_PRODUCT_CUTOFF
from scripts.synthetic_dataset.prompts import GeneratedCircuit
from scripts.synthetic_dataset.verification import (
    is_near_duplicate,
    measure_shape,
    structural_idle_check,
)

SOURCE_CODE = "qc = QuantumCircuit(1)\nqc.h(0)\nqc.z(0)\nqc.h(0)\n"


def _circuit(source_code: str = SOURCE_CODE) -> GeneratedCircuit:
    """Costruisce un GeneratedCircuit: source_code e' ormai il suo unico campo."""
    return GeneratedCircuit(source_code=source_code)


def _facade_measuring(
    max_ops_per_qubit: int, max_parallel_ops: int, idq: int, worst_qubit: int | None = 0
) -> Mock:
    """Mock di facade il cui calculate_smell_metrics ritorna la forma indicata."""
    facade = Mock(spec=IQiskitFacade)
    facade.calculate_smell_metrics.return_value = {
        "longCircuit": {
            "maxOpsPerQubit": max_ops_per_qubit,
            "maxParallelOps": max_parallel_ops,
            "value": max_ops_per_qubit * max_parallel_ops,
            "gateError": 0.00485,
            "errorFreeProbability": 1.0,
        },
        "idleQubits": {"value": idq, "worstQubit": worst_qubit if idq else None},
    }
    return facade


# ------------------------------------------------------------------------------------------
# measure_shape: la traduzione da numeri a etichetta, cioe' dove nasce il ground truth.
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_measure_shape_labels_long_circuit_exactly_at_the_cutoff() -> None:
    """La soglia e' inclusiva: l*c == LC_PRODUCT_CUTOFF e' gia' Long Circuit."""
    facade = _facade_measuring(max_ops_per_qubit=LC_PRODUCT_CUTOFF, max_parallel_ops=1, idq=0)

    shape = measure_shape(SOURCE_CODE, facade)

    assert shape["lc_product"] == LC_PRODUCT_CUTOFF
    assert shape["measured_smells"] == ["long_circuit"]


@pytest.mark.unit
def test_measure_shape_does_not_label_one_below_the_cutoff() -> None:
    """Un'unita' sotto la soglia il circuito e' pulito: e' il caso reale di idq-smelly (18)."""
    facade = _facade_measuring(max_ops_per_qubit=LC_PRODUCT_CUTOFF - 1, max_parallel_ops=1, idq=0)

    assert measure_shape(SOURCE_CODE, facade)["measured_smells"] == []


@pytest.mark.unit
def test_measure_shape_labels_idle_qubits_from_a_single_empty_column() -> None:
    """La soglia di IdQ e' > 0: una sola colonna di attesa basta (caso StackExchange_16_fix)."""
    facade = _facade_measuring(max_ops_per_qubit=3, max_parallel_ops=3, idq=1, worst_qubit=0)

    shape = measure_shape(SOURCE_CODE, facade)

    assert shape["measured_smells"] == ["idle_qubits"]
    assert shape["idq_worst_qubit"] == 0


@pytest.mark.unit
def test_measure_shape_labels_both_smells_together() -> None:
    """Forma misurata su Terra-0-4000_10_fix.py, l'unico circuito reale con entrambi."""
    facade = _facade_measuring(max_ops_per_qubit=7, max_parallel_ops=5, idq=3, worst_qubit=0)

    shape = measure_shape(SOURCE_CODE, facade)

    assert shape["l"] == 7
    assert shape["c"] == 5
    assert shape["lc_product"] == 35
    assert shape["measured_smells"] == ["long_circuit", "idle_qubits"]


# ------------------------------------------------------------------------------------------
# is_near_duplicate: il confronto non deve dipendere da come si chiama la variabile.
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_is_near_duplicate_detects_an_identical_source() -> None:
    assert is_near_duplicate(_circuit(), [_circuit()]) == "identical_source"


@pytest.mark.unit
def test_is_near_duplicate_detects_the_same_gate_sequence_under_a_different_variable() -> None:
    """I circuiti reali usano qc, qc_bad, circuit: ancorare il confronto a 'qc.' lo renderebbe cieco."""
    first = _circuit("qc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\nqc.measure_all()\n")
    second = _circuit(
        "circuit = QuantumCircuit(2)\ncircuit.h(1)\ncircuit.cx(1, 0)\ncircuit.measure_all()\n"
    )

    assert is_near_duplicate(first, [second]) == "same_gate_sequence"


@pytest.mark.unit
def test_is_near_duplicate_accepts_a_structurally_different_circuit() -> None:
    first = _circuit("qc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\nqc.measure_all()\n")
    second = _circuit("qc = QuantumCircuit(2)\nqc.x(0)\nqc.z(1)\nqc.barrier()\n")

    assert is_near_duplicate(first, [second]) is None


# ------------------------------------------------------------------------------------------
# structural_idle_check: metadato sul punto cieco di QSMELL, non piu' criterio di scarto.
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_structural_idle_check_reports_qubits_that_receive_nothing() -> None:
    """Caso Terra-11000-15000_12801: 4 qubit su 5 mai toccati, e IdQ vale comunque 0."""
    facade = Mock(spec=IQiskitFacade)
    circuit = facade.isolate_circuit.return_value
    circuit.num_qubits = 5
    instruction = Mock()
    instruction.operation.name = "ry"
    instruction.qubits = ["q4"]
    circuit.data = [instruction]
    circuit.find_bit.return_value.index = 4

    assert structural_idle_check(SOURCE_CODE, facade)["idle_qubit_indices"] == [0, 1, 2, 3]


@pytest.mark.unit
def test_structural_idle_check_does_not_count_a_barrier_as_a_real_operation() -> None:
    """Un barrier register-wide mascherebbe come attivo un qubit che non riceve nulla."""
    facade = Mock(spec=IQiskitFacade)
    circuit = facade.isolate_circuit.return_value
    circuit.num_qubits = 2
    barrier = Mock()
    barrier.operation.name = "barrier"
    barrier.qubits = ["q0", "q1"]
    circuit.data = [barrier]

    assert structural_idle_check(SOURCE_CODE, facade)["idle_qubit_indices"] == [0, 1]
