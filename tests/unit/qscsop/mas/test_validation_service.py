"""Unit test della pipeline di verifica deterministica.

La facade e' sempre MOCKATA (spec=IQiskitFacade): qui si verifica la logica di decisione del
servizio -- ordine dei controlli, corto-circuiti, criterio "Migliori?", forma del DTO -- non il
comportamento di Qiskit, gia' coperto dai test su QiskitFacade.

I test sul criterio sono stati riscritti insieme al criterio. Prima verificavano il Pareto sulla
terna (physicalMetrics.gateCount, physicalMetrics.depth, logicalQubits); ora quello sulla coppia
(l*c, IdQ). La forma e' la stessa, i valori confrontati no, e il test che fissa la ragione del
cambio e' test_accepts_an_idle_qubits_fix_that_leaves_every_cost_metric_untouched.
"""

from unittest.mock import Mock

import pytest

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qscsop.mas.dto.failure_reason import FailureReason
from qscsop_pipeline.qscsop.mas.validation.validation_service import ValidationService

BASELINE_CODE = "qc = QuantumCircuit(1)\nqc.x(0)\n"
NEW_CODE = "qc = QuantumCircuit(1)\nqc.h(0)\nqc.z(0)\nqc.h(0)\n"


def _payload(*, lc: int, idq: int, max_ops: int = 7, max_parallel: int = 5) -> dict:
    """Payload di calculate_smell_metrics. lc e idq sono espliciti: sono cio' che decide."""
    return {
        "longCircuit": {
            "maxOpsPerQubit": max_ops,
            "maxParallelOps": max_parallel,
            "value": lc,
            "gateError": 0.00485,
            "errorFreeProbability": 0.84,
        },
        "idleQubits": {"value": idq, "worstQubit": 0 if idq else None},
    }


# Forma misurata su Terra-0-4000_10_fix.py, l'unico circuito reale affidabile con entrambi gli
# smell: l*c = 35, IdQ = 3.
BASELINE_METRICS = _payload(lc=35, idq=3)
IMPROVED_METRICS = _payload(lc=35, idq=0)


def _ready_facade(baseline_metrics: dict, new_metrics: dict) -> Mock:
    """Facade che compila, e' equivalente e misura le forme indicate."""
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = True

    def _measure(code: str) -> dict:
        return baseline_metrics if code == BASELINE_CODE else new_metrics

    facade.calculate_smell_metrics.side_effect = _measure
    return facade


# ------------------------------------------------------------------------------------------
# Ordine dei controlli e corto-circuiti.
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_stops_at_compile_failure() -> None:
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (False, "SyntaxError: invalid syntax")

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    assert result.get_raw_error_data() == "SyntaxError: invalid syntax"
    assert result.get_new_metrics() is None
    assert result.get_failure_reason() == FailureReason.COMPILATION_FAILED
    facade.check_equivalence.assert_not_called()
    facade.calculate_smell_metrics.assert_not_called()


@pytest.mark.unit
def test_validate_stops_at_equivalence_failure() -> None:
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = False

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    assert isinstance(result.get_raw_error_data(), str)
    assert result.get_new_metrics() is None
    assert result.get_failure_reason() == FailureReason.NOT_EQUIVALENT
    facade.calculate_smell_metrics.assert_not_called()


@pytest.mark.unit
def test_validate_calls_facade_methods_in_order() -> None:
    facade = _ready_facade(BASELINE_METRICS, IMPROVED_METRICS)

    ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    # La misura e' invocata due volte: prima sul new_code, poi sulla baseline.
    assert [call[0] for call in facade.mock_calls] == [
        "compile_circuit",
        "check_equivalence",
        "calculate_smell_metrics",
        "calculate_smell_metrics",
    ]


# ------------------------------------------------------------------------------------------
# Il criterio "Migliori?": Pareto sulla coppia (l*c, IdQ).
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_accepts_an_idle_qubits_fix_that_leaves_every_cost_metric_untouched() -> None:
    """LA RAGIONE DEL CAMBIO DI CRITERIO, in un test solo.

    Il fix canonico di Idle Qubits non riduce ne' gateCount, ne' depth, ne' logicalQubits: su una
    coppia costruita equivalente le tre restavano identiche (9, 6, 2) mentre IdQ scendeva da 2 a
    0, e il vecchio criterio la respingeva. Il ciclo del MASEngine non poteva chiudersi con
    successo su quello smell, mai. Qui l*c resta identico e solo IdQ scende: se questo test
    tornasse rosso, il verdetto sarebbe di nuovo appeso a grandezze che con gli smell non
    c'entrano.
    """
    facade = _ready_facade(_payload(lc=35, idq=2), _payload(lc=35, idq=0))

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is True
    assert result.get_failure_reason() is None


@pytest.mark.unit
def test_accepts_a_long_circuit_fix_that_leaves_idle_qubits_untouched() -> None:
    """L'altra direzione: l*c scende rimuovendo operazioni, IdQ resta dov'era."""
    facade = _ready_facade(_payload(lc=35, idq=3), _payload(lc=18, idq=3))

    assert ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE).get_is_valid()


@pytest.mark.unit
def test_rejects_identity_refactoring() -> None:
    """Compila ed e' equivalente, ma non muove nessuna delle due metriche."""
    facade = _ready_facade(BASELINE_METRICS, BASELINE_METRICS)

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    # A differenza degli altri due fallimenti, qui new_metrics NON e' None: il new_code e' stato
    # compilato, verificato equivalente e misurato con successo, solo il criterio "Migliori?" non
    # e' soddisfatto. E' anche il segnale STRUTTURALE con cui il ReviewerAgent riconosce questo
    # fallimento fra i tre.
    assert result.get_new_metrics() == BASELINE_METRICS
    assert result.get_failure_reason() == FailureReason.METRICS_NOT_IMPROVED


@pytest.mark.unit
def test_rejects_a_fix_that_trades_one_smell_for_the_other() -> None:
    """IdQ scende ma l*c sale: Pareto vieta il baratto.

    E' il caso del riempimento dell'attesa che allunga la catena del qubit piu' carico -- il
    motivo per cui il messaggio al Reviewer avverte esplicitamente di non farlo.
    """
    facade = _ready_facade(_payload(lc=35, idq=3), _payload(lc=42, idq=0))

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    assert result.get_failure_reason() == FailureReason.METRICS_NOT_IMPROVED


@pytest.mark.unit
def test_the_error_message_reports_the_metrics_the_verdict_was_given_on() -> None:
    """Quel messaggio e' il feedback che arriva al ReviewerAgent.

    Se continuasse a riportare gateCount e depth, manderebbe il tentativo successivo a
    ottimizzare grandezze che non decidono piu' nulla.
    """
    facade = _ready_facade(_payload(lc=35, idq=3), _payload(lc=35, idq=3))

    error = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE).get_raw_error_data()

    assert "l*c=35" in error
    assert "IdQ=3" in error
    assert "gateCount" not in error
    assert "depth" not in error


# ------------------------------------------------------------------------------------------
# Robustezza: nessuna eccezione deve propagarsi fuori da validate().
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_returns_terminal_dto_when_check_equivalence_raises() -> None:
    # check_equivalence puo' sollevare NotImplementedError (feedback classico): non deve
    # propagarsi, ma diventare un DTO terminale con l'errore descritto.
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.side_effect = NotImplementedError("feedback classico non supportato")

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    assert result.get_new_metrics() is None
    assert "NotImplementedError" in result.get_raw_error_data()
    assert result.get_failure_reason() == FailureReason.UNEXPECTED_ERROR


@pytest.mark.unit
def test_validate_returns_terminal_dto_when_the_measure_raises() -> None:
    # calculate_smell_metrics isola ed ESEGUE il sorgente: puo' fallire su codice non fidato.
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = True
    facade.calculate_smell_metrics.side_effect = ValueError("nessun QuantumCircuit assegnato")

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    assert result.get_new_metrics() is None
    assert "ValueError" in result.get_raw_error_data()
    assert result.get_failure_reason() == FailureReason.UNEXPECTED_ERROR


# ------------------------------------------------------------------------------------------
# Cache della baseline.
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_caches_baseline_metrics_across_calls() -> None:
    # Due validate() con lo STESSO baseline_code ma new_code diversi: la baseline va misurata una
    # sola volta (cache), i due new_code ogni volta.
    facade = _ready_facade(BASELINE_METRICS, IMPROVED_METRICS)
    new_code_a = "qc = QuantumCircuit(1)\nqc.h(0)\n"
    new_code_b = "qc = QuantumCircuit(1)\nqc.y(0)\n"

    service = ValidationService(facade=facade)
    service.validate(BASELINE_CODE, new_code_a)
    service.validate(BASELINE_CODE, new_code_b)

    called_codes = [call.args[0] for call in facade.calculate_smell_metrics.call_args_list]
    assert called_codes.count(BASELINE_CODE) == 1
    assert called_codes.count(new_code_a) == 1
    assert called_codes.count(new_code_b) == 1
