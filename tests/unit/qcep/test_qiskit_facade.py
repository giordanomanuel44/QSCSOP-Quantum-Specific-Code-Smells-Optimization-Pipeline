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
def test_check_equivalence_on_same_dimension_rejects_a_control_only_false_positive(
    facade: QiskitFacade,
) -> None:
    # Falso positivo reale che la vecchia logica (Statevector da |00>, quindi solo il
    # comportamento a partire dall'input di default) avrebbe accettato erroneamente: da |00> un
    # CX con controllo 0 non scatta mai, quindi entrambi i circuiti restano fermi su |00> e
    # sarebbero risultati "equivalenti". Operator confronta l'intera trasformazione (valida per
    # QUALUNQUE input, incluso |10>, dove i due circuiti si comportano in modo diverso): il
    # confronto deve ora correttamente ritornare False.
    cx_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.cx(0, 1)
"""
    empty_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
"""

    assert facade.check_equivalence(cx_source, empty_source) is False


@pytest.mark.unit
def test_check_equivalence_across_dimensions_rejects_control_only_gates_inert_from_zero(
    facade: QiskitFacade,
) -> None:
    # Riproduzione ridotta del bug reale osservato in produzione: baseline a 5 qubit con tre ccx
    # concatenati (tutti inerti da |00000>) validato come equivalente a un refactored a 3 qubit
    # con un solo ccx (anch'esso inerte da |000>) — entrambi "non fanno nulla" dall'origine, quindi
    # il vecchio confronto (solo |0...0>) li dichiarava a torto equivalenti. Qui in forma ridotta
    # (4 vs 3 qubit invece di 5 vs 3): un ccx richiede sempre 3 qubit distinti, quindi la coppia
    # letterale "3 vs 2 qubit" non e' costruibile con gate ccx — 4 vs 3 e' la riduzione minima che
    # preserva la stessa struttura del caso reale (due ccx concatenati vs un solo ccx, entrambi
    # inerti da zero).
    baseline_double_ccx_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(4)
qc.ccx(0, 1, 2)
qc.ccx(0, 1, 3)
"""
    refactored_single_ccx_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.ccx(1, 2, 0)
"""

    # Da |0000>/|000> entrambi restano inerti (i controlli sono 0): il vecchio confronto (solo
    # |0...0>) li avrebbe dichiarati equivalenti. Da |1111>/|111> i ccx scattano davvero,
    # producendo pattern di bit finali strutturalmente incompatibili per qualunque scelta di
    # qubit tracciato via: il confronto ora deve ritornare False.
    assert (
        facade.check_equivalence(baseline_double_ccx_source, refactored_single_ccx_source) is False
    )


@pytest.mark.unit
def test_check_equivalence_rejects_circuits_above_the_operator_limit_at_same_dimension(
    facade: QiskitFacade,
) -> None:
    # Stesso tipo di guardia gia' verificata per il ramo partial_trace (dimensioni diverse), ora
    # applicata al ramo Operator (stessa dimensione): il limite deve scattare PRIMA di costruire
    # qualunque Operator (2^13 x 2^13 sarebbe enorme), quindi il test resta istantaneo.
    oversized_same_dimension_source = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(13)
qc.h(0)
"""

    with pytest.raises(ValueError, match="13 qubit, limite 12"):
        facade.check_equivalence(oversized_same_dimension_source, oversized_same_dimension_source)


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


# --- Metriche QSMELL (Chen et al., ICSE 2023) -------------------------------------------------
#
# I sorgenti che seguono sono i Listing del paper, trascritti inline invece di essere letti da
# data/raw/thesmellyeight/: un test unitario non deve dipendere dalla presenza di un dataset su
# disco. I valori attesi sono gli oracoli PUBBLICATI o derivati dall'implementazione di
# riferimento github.com/jose/qsmell, non da questa implementazione.

# Listing 3, ricostruito dalla execution matrix stampata nel paper (Sez. V-A1): il paper dichiara
# per questo circuito l=6, c=5 e quindi (1-0.03512)^30 = 0.34.
PAPER_LISTING_3 = """
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(1, 'c')
qc = QuantumCircuit(qreg_q, creg_c)
qc.u(0, 0, 0, qreg_q[0])
qc.cx(qreg_q[0], qreg_q[1])
qc.cx(qreg_q[1], qreg_q[2])
qc.cx(qreg_q[2], qreg_q[3])
qc.cx(qreg_q[3], qreg_q[4])
qc.barrier()
qc.rz(0.1, qreg_q[0])
qc.rz(0.1, qreg_q[1])
qc.rz(0.1, qreg_q[2])
qc.rz(0.1, qreg_q[3])
qc.rz(0.1, qreg_q[4])
qc.barrier()
qc.cx(qreg_q[3], qreg_q[4])
qc.cx(qreg_q[2], qreg_q[3])
qc.cx(qreg_q[1], qreg_q[2])
qc.cx(qreg_q[0], qreg_q[1])
qc.u(0, 0, 0, qreg_q[0])
qc.measure(qreg_q[0], creg_c[0])
"""

# Listing 4: l'esempio con cui il paper ILLUSTRA Long Circuit (identita' HZH = X).
PAPER_LISTING_4_LONG_CIRCUIT = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(1)
qc.h(0)
qc.z(0)
qc.h(0)
"""

# Listing 6: l'esempio di Idle Qubits, nelle sue due versioni smelly e fixed.
PAPER_LISTING_6_SMELLY = """
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
qc = QuantumCircuit(qreg_q, creg_c)
qc.h(qreg_q)
qc.p(pi / 2, qreg_q[0])
qc.z(qreg_q[0])
qc.s(qreg_q[0])
qc.barrier()
qc.p(pi / 4, qreg_q[1])
qc.z(qreg_q[1])
qc.s(qreg_q[1])
qc.barrier()
qc.h(qreg_q[2])
qc.p(pi / 8, qreg_q[2])
qc.z(qreg_q[2])
qc.s(qreg_q[2])
qc.measure_all(add_bits=False)
"""

PAPER_LISTING_6_FIXED = """
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
qc = QuantumCircuit(qreg_q, creg_c)
qc.h(qreg_q[0])
qc.p(pi / 2, qreg_q[0])
qc.z(qreg_q[0])
qc.s(qreg_q[0])
qc.measure(qreg_q[0], creg_c[0])
qc.barrier()
qc.h(qreg_q[1])
qc.p(pi / 4, qreg_q[1])
qc.z(qreg_q[1])
qc.s(qreg_q[1])
qc.measure(qreg_q[1], creg_c[1])
qc.barrier()
qc.h(qreg_q[2])
qc.p(pi / 8, qreg_q[2])
qc.z(qreg_q[2])
qc.s(qreg_q[2])
qc.measure(qreg_q[2], creg_c[2])
"""


@pytest.mark.unit
def test_long_circuit_metric_reproduces_the_published_worked_example(
    facade: QiskitFacade,
) -> None:
    # Oracolo esterno: il paper pubblica l=6, c=5 per questo circuito. E' il test piu' importante
    # della suite sulle metriche, perche' e' l'unico verificabile da terzi senza fidarsi di noi.
    long_circuit = facade.calculate_smell_metrics(PAPER_LISTING_3)["longCircuit"]

    assert long_circuit["maxOpsPerQubit"] == 6
    assert long_circuit["maxParallelOps"] == 5
    assert long_circuit["value"] == 30
    # Con l'errore di gate del paper (0.03512), lo stesso prodotto da' lo 0.34 stampato in tabella.
    assert (1 - 0.03512) ** long_circuit["value"] == pytest.approx(0.34, abs=0.005)


@pytest.mark.unit
def test_error_free_probability_is_derived_from_the_declared_gate_error(
    facade: QiskitFacade,
) -> None:
    # La forma esponenziale e' presentazione, non rilevamento: deve essere ricostruibile dai due
    # valori che il payload espone, senza conoscere costanti interne alla facade.
    long_circuit = facade.calculate_smell_metrics(PAPER_LISTING_3)["longCircuit"]

    expected = (1 - long_circuit["gateError"]) ** long_circuit["value"]
    assert long_circuit["errorFreeProbability"] == pytest.approx(expected)


@pytest.mark.unit
def test_long_circuit_metric_ignores_barriers_but_counts_measures(
    facade: QiskitFacade,
) -> None:
    # Due convenzioni dell'implementazione di riferimento: i barrier non sono operazioni (ma
    # occupano un livello), le measure si'.
    with_barrier = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.barrier()
qc.x(0)
qc.x(1)
"""
    with_measure = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.measure_all()
"""

    assert facade.calculate_smell_metrics(with_barrier)["longCircuit"]["maxOpsPerQubit"] == 2
    assert facade.calculate_smell_metrics(with_measure)["longCircuit"]["maxOpsPerQubit"] == 2


@pytest.mark.unit
def test_idle_qubits_metric_counts_empty_timestamps_between_two_uses(
    facade: QiskitFacade,
) -> None:
    # q1 riceve una H e poi resta fermo tre colonne, finche' la cx non lo coinvolge di nuovo:
    # due delle tre sono celle vuote fra due sue operazioni, la terza e' la cx stessa.
    waiting_qubit = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(1)
qc.x(0)
qc.x(0)
qc.x(0)
qc.cx(0, 1)
"""

    idle_qubits = facade.calculate_smell_metrics(waiting_qubit)["idleQubits"]

    assert idle_qubits["value"] == 2
    assert idle_qubits["worstQubit"] == 1


@pytest.mark.unit
def test_idle_qubits_metric_skips_barrier_columns(facade: QiskitFacade) -> None:
    # La colonna del barrier non conta ne' come operazione ne' come attesa: i due qubit passano
    # da h a x senza accumulare alcun gap.
    barrier_between = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.barrier()
qc.x(0)
qc.x(1)
"""

    assert facade.calculate_smell_metrics(barrier_between)["idleQubits"]["value"] == 0


@pytest.mark.unit
def test_idle_qubits_metric_ignores_idleness_before_and_after_a_qubit_is_used(
    facade: QiskitFacade,
) -> None:
    # Nella versione fixed del Listing 6 ogni qubit viene misurato appena finite le sue
    # operazioni: le colonne vuote che lo precedono e quelle che lo seguono non sono attesa fra
    # DUE usi, quindi la metrica deve annullarsi. E' il caso che valida entrambe le convenzioni
    # di bordo dell'implementazione di riferimento.
    assert facade.calculate_smell_metrics(PAPER_LISTING_6_FIXED)["idleQubits"]["value"] == 0


@pytest.mark.unit
def test_smell_metrics_on_the_paper_idle_qubits_listing(facade: QiskitFacade) -> None:
    # Il Listing 6 smelly deve risultare peggiore della sua versione fixed su ENTRAMBE le
    # metriche: e' la coppia di riferimento del paper per Idle Qubits.
    smelly = facade.calculate_smell_metrics(PAPER_LISTING_6_SMELLY)
    fixed = facade.calculate_smell_metrics(PAPER_LISTING_6_FIXED)

    assert smelly["longCircuit"]["value"] == 18
    assert smelly["idleQubits"] == {"value": 7, "worstQubit": 0}
    assert fixed["longCircuit"]["value"] < smelly["longCircuit"]["value"]
    assert fixed["idleQubits"]["value"] < smelly["idleQubits"]["value"]


@pytest.mark.unit
def test_smell_metrics_on_the_paper_long_circuit_listing(facade: QiskitFacade) -> None:
    # Incoerenza interna del paper, fissata qui come comportamento atteso e non come difetto:
    # il Listing 4 e' l'esempio con cui il paper ILLUSTRA Long Circuit, ma il suo prodotto l*c
    # vale 3, ben sotto il taglio di 20 che la soglia pubblicata (0.50 con error 0.03512)
    # impone. Con la metrica del paper, quel circuito non e' Long Circuit.
    long_circuit = facade.calculate_smell_metrics(PAPER_LISTING_4_LONG_CIRCUIT)["longCircuit"]

    assert long_circuit["value"] == 3
    assert (1 - 0.03512) ** long_circuit["value"] > 0.50


@pytest.mark.unit
def test_calculate_smell_metrics_on_a_circuit_without_operations(facade: QiskitFacade) -> None:
    empty_circuit = """
from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
"""

    metrics = facade.calculate_smell_metrics(empty_circuit)

    assert metrics["longCircuit"]["value"] == 0
    assert metrics["longCircuit"]["errorFreeProbability"] == 1.0
    assert metrics["idleQubits"] == {"value": 0, "worstQubit": None}


@pytest.mark.unit
def test_calculate_smell_metrics_does_not_require_binding_free_parameters(
    facade: QiskitFacade,
) -> None:
    # A differenza di is_qubit_idle, qui non si simula alcuno stato: si contano celle. Un
    # Parameter non legato non deve quindi essere un ostacolo.
    parametric_source = """
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

theta = Parameter('theta')
qc = QuantumCircuit(2)
qc.rx(theta, 0)
qc.cx(0, 1)
"""

    assert facade.calculate_smell_metrics(parametric_source)["longCircuit"]["value"] == 4


# ------------------------------------------------------------------------------------------
# maxOpsQubits: il puntatore che dice DA DOVE rimuovere per abbassare l.
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_max_ops_qubits_names_the_single_qubit_holding_the_maximum(facade: QiskitFacade) -> None:
    code = (
        "from qiskit import QuantumCircuit\n"
        "qc = QuantumCircuit(2)\n"
        "qc.x(0)\nqc.x(0)\nqc.h(0)\nqc.h(1)\nqc.cx(0, 1)\n"
    )

    long_circuit = facade.calculate_smell_metrics(code)["longCircuit"]

    assert long_circuit["maxOpsPerQubit"] == 4
    assert long_circuit["maxOpsQubits"] == [0]


@pytest.mark.unit
def test_max_ops_qubits_lists_every_qubit_when_the_maximum_is_shared(
    facade: QiskitFacade,
) -> None:
    """E' UNA LISTA per una ragione operativa, non per generalita'.

    Togliere operazioni da uno solo dei qubit che realizzano il massimo non abbassa l di
    un'unita': il massimo resta, tenuto dagli altri. Sui 72 circuiti del dataset sintetico questo
    caso ricorre in 34, quindi un singolo indice manderebbe il RefactorerAgent a fare lavoro
    inutile quasi una volta su due.
    """
    code = (
        "from qiskit import QuantumCircuit\n"
        "qc = QuantumCircuit(2)\n"
        "qc.x(0)\nqc.x(0)\nqc.h(0)\nqc.x(1)\nqc.x(1)\nqc.h(1)\nqc.cx(0, 1)\n"
    )

    long_circuit = facade.calculate_smell_metrics(code)["longCircuit"]

    assert long_circuit["maxOpsPerQubit"] == 4
    assert long_circuit["maxOpsQubits"] == [0, 1]


@pytest.mark.unit
def test_max_ops_qubits_is_empty_on_a_circuit_without_operations(facade: QiskitFacade) -> None:
    code = "from qiskit import QuantumCircuit\nqc = QuantumCircuit(3)\n"

    long_circuit = facade.calculate_smell_metrics(code)["longCircuit"]

    assert long_circuit["maxOpsPerQubit"] == 0
    assert long_circuit["maxOpsQubits"] == []


@pytest.mark.unit
def test_no_symmetric_pointer_exists_for_the_columns(facade: QiskitFacade) -> None:
    """L'ASIMMETRIA E' UNA DECISIONE, non una svista: questo test la protegge.

    Non esiste un "maxParallelColumns" perche' c non e' una leva. Le due misure che lo dicono
    sono riprodotte qui sotto: rimuovere operazioni abbassa l e lascia c dov'era, mentre l'unico
    modo di abbassare c deliberatamente e' serializzare con barrier -- stessi gate, zero
    rimozioni, equivalenza preservata e metrica migliorata, cioe' esattamente il "fix" peggiore
    dell'originale su hardware reale che il modello non deve imparare.
    """
    header = "from qiskit import QuantumCircuit\nqc = QuantumCircuit(4)\n"
    dense = header + "qc.h(0)\nqc.h(1)\nqc.h(2)\nqc.h(3)\nqc.cx(0, 1)\nqc.cx(2, 3)\n"
    serialised = header + (
        "qc.h(0)\nqc.barrier()\nqc.h(1)\nqc.barrier()\nqc.h(2)\nqc.barrier()\nqc.h(3)\n"
        "qc.cx(0, 1)\nqc.cx(2, 3)\n"
    )

    dense_metrics = facade.calculate_smell_metrics(dense)["longCircuit"]
    serialised_metrics = facade.calculate_smell_metrics(serialised)["longCircuit"]

    assert "maxParallelColumns" not in dense_metrics
    # La serializzazione abbassa c -- e quindi l*c -- senza togliere un solo gate.
    assert serialised_metrics["maxParallelOps"] < dense_metrics["maxParallelOps"]
    assert serialised_metrics["value"] < dense_metrics["value"]
    assert facade.check_equivalence(dense, serialised) is True


# ------------------------------------------------------------------------------------------
# operationsPerQubit: il ponte fra il sorgente (che ha loop) e il circuito eseguito.
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_timeline_per_qubit_bridges_source_lines_and_executed_operations(
    facade: QiskitFacade,
) -> None:
    """IL PUNTO DELL'INTERA CHIAVE, in un test.

    Il sorgente ha 7 righe e costruisce 21 operazioni su q0 con un for. Dando al DetectorAgent
    il solo conteggio l = 21, quello immaginava uno srotolamento inesistente e prescriveva
    rimozioni a righe che nel file non c'erano. La sequenza dice cosa succede davvero, e la sua
    lunghezza deve coincidere con l per il qubit al massimo.
    """
    code = (
        "from qiskit import QuantumCircuit\n"
        "qc = QuantumCircuit(2, 2)\n"
        "for i in range(10):\n    qc.h(0)\n    qc.cx(0, 1)\n"
        "qc.measure(0, 0)\nqc.measure(1, 1)\n"
    )

    long_circuit = facade.calculate_smell_metrics(code)["longCircuit"]
    sequences = long_circuit["timelinePerQubit"]

    assert len(code.splitlines()) == 7
    assert long_circuit["maxOpsPerQubit"] == 21
    assert len(sequences[0].split(", ")) == 21
    # Gli h su q0 sono separati dai cx: nessuna coppia adiacente da cancellare. E' esattamente
    # la ridondanza che il modello aveva inventato.
    assert sequences[0].startswith("h, cx, h, cx")
    assert "h, h" not in sequences[0]


@pytest.mark.unit
def test_timeline_per_qubit_is_ordered_by_index_and_excludes_barriers(
    facade: QiskitFacade,
) -> None:
    code = (
        "from qiskit import QuantumCircuit\n"
        "qc = QuantumCircuit(3)\n"
        "qc.h(0)\nqc.barrier()\nqc.x(0)\nqc.y(1)\n"
    )

    sequences = facade.calculate_smell_metrics(code)["longCircuit"]["timelinePerQubit"]

    assert sequences == ["h, x", "y", ""]


@pytest.mark.unit
def test_timeline_per_qubit_has_one_entry_per_qubit_even_when_empty(
    facade: QiskitFacade,
) -> None:
    code = "from qiskit import QuantumCircuit\nqc = QuantumCircuit(3)\n"

    sequences = facade.calculate_smell_metrics(code)["longCircuit"]["timelinePerQubit"]

    assert sequences == ["", "", ""]
