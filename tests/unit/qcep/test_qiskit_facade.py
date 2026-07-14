import io
import sys

import pytest
from qiskit import QuantumCircuit

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade

BELL_SOURCE = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""

DIFFERENT_NAME_SOURCE = """
from qiskit import QuantumCircuit

my_circuit = QuantumCircuit(3)
my_circuit.h(0)
"""

MULTIPLE_CIRCUITS_SOURCE = """
from qiskit import QuantumCircuit

first_qc = QuantumCircuit(1)
second_qc = QuantumCircuit(4)
final_qc = QuantumCircuit(2)
final_qc.h(0)
"""

BROKEN_SOURCE = "def broken(:\n    pass"

NO_CIRCUIT_SOURCE = "x = 1 + 1"


@pytest.fixture
def facade() -> QiskitFacade:
    return QiskitFacade()


@pytest.mark.unit
def test_isolate_circuit_bell_circuit(facade: QiskitFacade) -> None:
    circuit = facade.isolate_circuit(BELL_SOURCE)

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


@pytest.mark.unit
def test_isolate_circuit_with_different_variable_name(facade: QiskitFacade) -> None:
    circuit = facade.isolate_circuit(DIFFERENT_NAME_SOURCE)

    assert circuit.num_qubits == 3


@pytest.mark.unit
def test_isolate_circuit_returns_last_assigned_circuit(facade: QiskitFacade) -> None:
    circuit = facade.isolate_circuit(MULTIPLE_CIRCUITS_SOURCE)

    assert circuit.num_qubits == 2


@pytest.mark.unit
def test_isolate_circuit_propagates_syntax_errors(facade: QiskitFacade) -> None:
    with pytest.raises(SyntaxError):
        facade.isolate_circuit(BROKEN_SOURCE)


@pytest.mark.unit
def test_isolate_circuit_raises_value_error_without_circuit(facade: QiskitFacade) -> None:
    with pytest.raises(ValueError):
        facade.isolate_circuit(NO_CIRCUIT_SOURCE)


@pytest.mark.unit
def test_isolate_circuit_survives_unencodable_print_output(
    facade: QiskitFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simula una console con encoding stretto (es. cp1252 su Windows) forzando stdout ad
    # ascii/strict: senza il reconfigure tollerante in isolate_circuit, il print() di un
    # carattere non-ASCII solleverebbe UnicodeEncodeError e farebbe fallire l'estrazione di
    # un circuito altrimenti valido.
    restrictive_stdout = io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")
    monkeypatch.setattr(sys, "stdout", restrictive_stdout)

    source_with_unencodable_print = BELL_SOURCE + "\nprint('\\u03b8')\n"

    circuit = facade.isolate_circuit(source_with_unencodable_print)

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


@pytest.mark.unit
def test_get_abstract_metrics_on_known_circuit(facade: QiskitFacade) -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    metrics = facade.get_abstract_metrics(qc)

    assert metrics == {"gateCount": 2, "depth": 2}


@pytest.mark.unit
def test_transpile_and_physical_metrics_are_consistent_with_abstract(facade: QiskitFacade) -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    abstract_metrics = facade.get_abstract_metrics(qc)
    transpiled = facade.transpile_circuit(qc)
    physical_metrics = facade.get_physical_metrics(transpiled)

    assert physical_metrics["gateCount"] >= abstract_metrics["gateCount"]
    assert physical_metrics["depth"] >= abstract_metrics["depth"]
