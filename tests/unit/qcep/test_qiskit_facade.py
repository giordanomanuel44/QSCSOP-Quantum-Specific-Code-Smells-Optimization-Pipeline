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


@pytest.mark.unit
def test_compile_circuit_on_valid_code_returns_true_and_no_message(facade: QiskitFacade) -> None:
    is_valid, error_message = facade.compile_circuit(BELL_SOURCE)

    assert is_valid is True
    assert error_message is None


@pytest.mark.unit
def test_compile_circuit_on_broken_code_returns_false_and_message(facade: QiskitFacade) -> None:
    is_valid, error_message = facade.compile_circuit(BROKEN_SOURCE)

    assert is_valid is False
    assert isinstance(error_message, str)
    assert error_message != ""


@pytest.mark.unit
def test_check_equivalence_on_identical_circuits_is_true(facade: QiskitFacade) -> None:
    assert facade.check_equivalence(BELL_SOURCE, BELL_SOURCE) is True


@pytest.mark.unit
def test_check_equivalence_on_functionally_different_circuits_is_false(
    facade: QiskitFacade,
) -> None:
    bell_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""
    hadamard_only_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
"""

    assert facade.check_equivalence(bell_source, hadamard_only_source) is False


@pytest.mark.unit
def test_check_equivalence_on_syntactically_different_but_equivalent_circuits_is_true(
    facade: QiskitFacade,
) -> None:
    # X e H-Z-H sono la stessa trasformazione (a meno di fase globale): gate diversi, stesso
    # stato risultante.
    baseline_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(1)
qc.x(0)
"""
    equivalent_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(1)
qc.h(0)
qc.z(0)
qc.h(0)
"""

    assert facade.check_equivalence(baseline_source, equivalent_source) is True


@pytest.mark.unit
def test_check_equivalence_on_identical_circuits_with_measure_all_is_true(
    facade: QiskitFacade,
) -> None:
    # Circuiti con measure_all() finale (stile idq-smelly): le misure vanno rimosse prima del
    # confronto Statevector, altrimenti Qiskit solleva "Cannot apply instruction with classical
    # bits". Due circuiti identici che misurano tutti i qubit devono risultare equivalenti.
    measured_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
"""

    assert facade.check_equivalence(measured_source, measured_source) is True


@pytest.mark.unit
def test_check_equivalence_on_identical_circuits_with_per_qubit_measures_is_true(
    facade: QiskitFacade,
) -> None:
    # Misure individuali per singolo qubit (stile idq-fixed), non measure_all: altro pattern di
    # misura terminale presente nel dataset reale, da gestire allo stesso modo.
    measured_source = """
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
qc = QuantumCircuit(qreg_q, creg_c)
qc.h(qreg_q[0])
qc.cx(qreg_q[0], qreg_q[1])
qc.measure(qreg_q[0], creg_c[0])
qc.measure(qreg_q[1], creg_c[1])
"""

    assert facade.check_equivalence(measured_source, measured_source) is True


@pytest.mark.unit
def test_check_equivalence_rejects_circuits_with_classical_feedback(
    facade: QiskitFacade,
) -> None:
    # Un circuito con un gate condizionato da un bit classico (if_test, il sostituto di c_if in
    # Qiskit 2.x) modella feedback classico: non e' rappresentabile come Statevector puro e va
    # rifiutato esplicitamente, non confrontato silenziosamente.
    conditional_source = """
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qreg_q = QuantumRegister(1, 'q')
creg_c = ClassicalRegister(1, 'c')
qc = QuantumCircuit(qreg_q, creg_c)
qc.h(qreg_q[0])
qc.measure(qreg_q[0], creg_c[0])
with qc.if_test((creg_c, 1)):
    qc.x(qreg_q[0])
"""

    with pytest.raises(NotImplementedError, match="feedback"):
        facade.check_equivalence(conditional_source, conditional_source)


@pytest.mark.unit
def test_check_equivalence_on_same_qubit_count_still_uses_direct_comparison(
    facade: QiskitFacade,
) -> None:
    # Non-regressione: a parita' di numero di qubit il percorso resta quello Statevector diretto,
    # invariato rispetto a prima dell'estensione con partial_trace. Equivalenti restano True,
    # non equivalenti restano False.
    hadamard_only_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
"""

    assert facade.check_equivalence(BELL_SOURCE, BELL_SOURCE) is True
    assert facade.check_equivalence(BELL_SOURCE, hadamard_only_source) is False


@pytest.mark.unit
def test_check_equivalence_ignores_a_genuinely_idle_qubit(facade: QiskitFacade) -> None:
    # Baseline a 3 qubit in cui q2 non riceve ALCUN gate (idle in senso stretto), confrontato con
    # il circuito a 2 qubit che replica esattamente la logica dei primi due: rimuovere un qubit
    # mai toccato non cambia nulla di osservabile, quindi devono risultare equivalenti.
    baseline_with_idle_qubit = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
"""
    reduced_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""

    assert facade.check_equivalence(baseline_with_idle_qubit, reduced_source) is True


@pytest.mark.unit
def test_check_equivalence_rejects_removal_of_an_entangled_qubit(facade: QiskitFacade) -> None:
    # Baseline GHZ: tutti e tre i qubit sono entangled, nessuno e' idle. Tracciarne via uno
    # qualsiasi produce uno stato MISTO, mai uguale allo stato puro di Bell a 2 qubit: il
    # meccanismo non deve diventare permissivo solo perche' le dimensioni sono diverse.
    ghz_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
"""
    bell_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""

    assert facade.check_equivalence(ghz_source, bell_source) is False


@pytest.mark.unit
def test_check_equivalence_rejects_circuits_above_the_partial_trace_limit(
    facade: QiskitFacade,
) -> None:
    # Il limite deve scattare PRIMA di costruire la DensityMatrix (2^13 x 2^13 sarebbe ~1 GB):
    # il test resta istantaneo proprio perche' nessuna matrice viene mai allocata.
    oversized_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(13)
qc.h(0)
"""
    smaller_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
"""

    with pytest.raises(ValueError, match="13 qubit, limite 12"):
        facade.check_equivalence(oversized_source, smaller_source)


@pytest.mark.unit
def test_calculate_metrics_returns_exactly_abstract_and_physical_keys(
    facade: QiskitFacade,
) -> None:
    metrics = facade.calculate_metrics(BELL_SOURCE)

    assert set(metrics.keys()) == {"abstractMetrics", "physicalMetrics"}
    assert metrics["abstractMetrics"] == {"gateCount": 2, "depth": 2}
    assert metrics["physicalMetrics"]["gateCount"] >= metrics["abstractMetrics"]["gateCount"]
    assert metrics["physicalMetrics"]["depth"] >= metrics["abstractMetrics"]["depth"]
