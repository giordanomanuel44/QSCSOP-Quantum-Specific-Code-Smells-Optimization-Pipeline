"""Unit test delle verifiche semantiche del tooling di generazione sintetica.

La facade e' sempre MOCKATA (spec=IQiskitFacade): qui si verifica la logica di decisione delle
funzioni di verification.py -- quando delegano alla facade, cosa concludono dal suo verdetto --
non il comportamento di Qiskit, gia' coperto dai test su QiskitFacade.
"""

from unittest.mock import Mock

import pytest

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qscsop.mas.dto.quantum_smell_type import QuantumSmellType
from scripts.synthetic_dataset.prompts import GeneratedCircuit
from scripts.synthetic_dataset.verification import (
    verify_declared_idle_qubits,
    verify_declared_simplification,
)

SOURCE_CODE = "qc = QuantumCircuit(1)\nqc.h(0)\nqc.z(0)\nqc.h(0)\n"
SIMPLIFIED_CODE = "qc = QuantumCircuit(1)\nqc.x(0)\n"


def _circuit(
    intended_smells: list[QuantumSmellType],
    simplified_source_code: str | None = None,
) -> GeneratedCircuit:
    """Costruisce un GeneratedCircuit minimo: solo i campi che le verifiche leggono contano."""
    return GeneratedCircuit(
        source_code=SOURCE_CODE,
        line_by_line_expansion="h -> 0, z -> 0, h -> 0",
        qubit_operation_analysis="q0: h, z, h",
        reasoning="H-Z-H equivale a una singola X.",
        intended_smells=intended_smells,
        simplified_source_code=simplified_source_code,
    )


def _facade_with_num_qubits(num_qubits: int) -> Mock:
    """Mock di facade il cui isolate_circuit espone solo num_qubits, l'unico attributo letto."""
    facade = Mock(spec=IQiskitFacade)
    facade.isolate_circuit.return_value.num_qubits = num_qubits
    return facade


@pytest.mark.unit
def test_verify_declared_simplification_returns_none_when_long_circuit_not_declared() -> None:
    facade = Mock(spec=IQiskitFacade)

    result = verify_declared_simplification(_circuit([QuantumSmellType.IDLE_QUBITS]), facade)

    assert result is None
    facade.check_equivalence.assert_not_called()


@pytest.mark.unit
def test_verify_declared_simplification_is_false_when_simplified_code_is_missing() -> None:
    facade = Mock(spec=IQiskitFacade)

    result = verify_declared_simplification(_circuit([QuantumSmellType.LONG_CIRCUIT]), facade)

    assert result is False
    facade.check_equivalence.assert_not_called()


@pytest.mark.unit
def test_verify_declared_simplification_is_false_when_simplified_code_is_blank() -> None:
    # Un campo riempito con soli spazi/a-capo e' una dichiarazione non sostanziata quanto uno
    # assente: non va inoltrato alla facade come se fosse codice.
    facade = Mock(spec=IQiskitFacade)

    result = verify_declared_simplification(
        _circuit([QuantumSmellType.LONG_CIRCUIT], simplified_source_code="   \n  "), facade
    )

    assert result is False
    facade.check_equivalence.assert_not_called()


@pytest.mark.unit
def test_verify_declared_simplification_is_true_when_facade_confirms_equivalence() -> None:
    facade = Mock(spec=IQiskitFacade)
    facade.check_equivalence.return_value = True

    result = verify_declared_simplification(
        _circuit([QuantumSmellType.LONG_CIRCUIT], simplified_source_code=SIMPLIFIED_CODE), facade
    )

    assert result is True
    facade.check_equivalence.assert_called_once_with(SOURCE_CODE, SIMPLIFIED_CODE)


@pytest.mark.unit
def test_verify_declared_simplification_is_false_when_circuits_are_not_equivalent() -> None:
    # Esattamente il caso da smascherare: il generatore dichiara una riduzione che non conserva
    # il comportamento del circuito originale.
    facade = Mock(spec=IQiskitFacade)
    facade.check_equivalence.return_value = False

    result = verify_declared_simplification(
        _circuit([QuantumSmellType.LONG_CIRCUIT], simplified_source_code=SIMPLIFIED_CODE), facade
    )

    assert result is False


@pytest.mark.unit
def test_verify_declared_simplification_is_false_when_the_facade_raises() -> None:
    # Diagnostica pura: un circuito semplificato che non compila, o oltre i limiti dimensionali
    # della facade, vale come verifica NON superata, non come errore da propagare al chiamante.
    facade = Mock(spec=IQiskitFacade)
    facade.check_equivalence.side_effect = ValueError("Circuito troppo grande")

    result = verify_declared_simplification(
        _circuit([QuantumSmellType.LONG_CIRCUIT], simplified_source_code=SIMPLIFIED_CODE), facade
    )

    assert result is False


@pytest.mark.unit
def test_verify_declared_idle_qubits_returns_none_when_idle_qubits_not_declared() -> None:
    facade = _facade_with_num_qubits(3)

    result = verify_declared_idle_qubits(_circuit([QuantumSmellType.LONG_CIRCUIT]), facade)

    assert result is None
    facade.is_qubit_idle.assert_not_called()


@pytest.mark.unit
def test_verify_declared_idle_qubits_is_true_when_one_qubit_is_confirmed_idle() -> None:
    facade = _facade_with_num_qubits(3)
    facade.is_qubit_idle.side_effect = [False, False, True]

    result = verify_declared_idle_qubits(_circuit([QuantumSmellType.IDLE_QUBITS]), facade)

    assert result is True


@pytest.mark.unit
def test_verify_declared_idle_qubits_checks_every_qubit_before_concluding_false() -> None:
    # Nessun qubit idle: la dichiarazione del generatore e' smentita, ma solo dopo aver
    # interrogato la facade su OGNI qubit del circuito.
    facade = _facade_with_num_qubits(3)
    facade.is_qubit_idle.return_value = False

    result = verify_declared_idle_qubits(_circuit([QuantumSmellType.IDLE_QUBITS]), facade)

    assert result is False
    assert facade.is_qubit_idle.call_count == 3
    assert [call.args for call in facade.is_qubit_idle.call_args_list] == [
        (SOURCE_CODE, 0),
        (SOURCE_CODE, 1),
        (SOURCE_CODE, 2),
    ]


@pytest.mark.unit
def test_verify_declared_idle_qubits_is_false_when_the_facade_raises() -> None:
    facade = _facade_with_num_qubits(2)
    facade.is_qubit_idle.side_effect = NotImplementedError("feedback classico")

    result = verify_declared_idle_qubits(_circuit([QuantumSmellType.IDLE_QUBITS]), facade)

    assert result is False
