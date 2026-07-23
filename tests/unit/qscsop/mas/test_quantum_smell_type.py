import pytest

from qscsop_pipeline.qscsop.mas.dto.quantum_smell_type import QuantumSmellType


@pytest.mark.unit
def test_smell_type_values_are_the_expected_strings() -> None:
    assert QuantumSmellType.LONG_CIRCUIT.value == "long_circuit"
    assert QuantumSmellType.IDLE_QUBITS.value == "idle_qubits"


@pytest.mark.unit
def test_smell_type_compares_directly_as_string() -> None:
    # Ereditando da str, i membri sono confrontabili direttamente con la stringa corrispondente.
    assert QuantumSmellType.LONG_CIRCUIT == "long_circuit"
    assert QuantumSmellType.IDLE_QUBITS == "idle_qubits"
