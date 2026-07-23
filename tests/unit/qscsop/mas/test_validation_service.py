from unittest.mock import Mock

import pytest

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qscsop.mas.validation.validation_service import ValidationService

BASELINE_CODE = "qc = QuantumCircuit(1)\nqc.x(0)\n"
NEW_CODE = "qc = QuantumCircuit(1)\nqc.h(0)\nqc.z(0)\nqc.h(0)\n"

# Metriche baseline di riferimento condivise dai test sul criterio di miglioramento.
BASELINE_METRICS = {
    "logicalQubits": 3,
    "abstractMetrics": {"gateCount": 5, "depth": 4},
    "physicalMetrics": {"gateCount": 10, "depth": 8},
}

# new con un solo gateCount fisico ridotto, resto invariato: soddisfa il criterio Pareto.
IMPROVED_METRICS = {
    "logicalQubits": 3,
    "abstractMetrics": {"gateCount": 4, "depth": 4},
    "physicalMetrics": {"gateCount": 8, "depth": 8},
}


def _metrics_side_effect(baseline_metrics: dict, new_metrics: dict):
    """Restituisce un side_effect per calculate_metrics che distingue baseline da new_code."""

    def _side_effect(code: str) -> dict:
        return baseline_metrics if code == BASELINE_CODE else new_metrics

    return _side_effect


@pytest.mark.unit
def test_validate_stops_at_compile_failure() -> None:
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (False, "SyntaxError: invalid syntax")

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    assert result.get_raw_error_data() == "SyntaxError: invalid syntax"
    assert result.get_new_metrics() is None
    facade.check_equivalence.assert_not_called()
    facade.calculate_metrics.assert_not_called()


@pytest.mark.unit
def test_validate_stops_at_equivalence_failure() -> None:
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = False

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    assert isinstance(result.get_raw_error_data(), str)
    assert result.get_raw_error_data() != ""
    assert result.get_new_metrics() is None
    facade.calculate_metrics.assert_not_called()


@pytest.mark.unit
def test_validate_succeeds_and_returns_calculated_metrics() -> None:
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = True
    facade.calculate_metrics.side_effect = _metrics_side_effect(BASELINE_METRICS, IMPROVED_METRICS)

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is True
    assert result.get_raw_error_data() is None
    assert result.get_new_metrics() == IMPROVED_METRICS


@pytest.mark.unit
def test_validate_calls_facade_methods_in_order() -> None:
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = True
    facade.calculate_metrics.side_effect = _metrics_side_effect(BASELINE_METRICS, IMPROVED_METRICS)

    ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    called_method_names = [call[0] for call in facade.mock_calls]
    # calculate_metrics e' invocato due volte: prima sul new_code, poi sulla baseline.
    assert called_method_names == [
        "compile_circuit",
        "check_equivalence",
        "calculate_metrics",
        "calculate_metrics",
    ]


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
    assert result.get_raw_error_data() is not None
    assert "NotImplementedError" in result.get_raw_error_data()


@pytest.mark.unit
def test_validate_returns_terminal_dto_when_calculate_metrics_raises() -> None:
    # calculate_metrics puo' sollevare ValueError (oltre il limite partial_trace / transpilazione):
    # stesso comportamento, nessuna propagazione.
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = True
    facade.calculate_metrics.side_effect = ValueError("circuito troppo grande")

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    assert result.get_new_metrics() is None
    assert result.get_raw_error_data() is not None
    assert "ValueError" in result.get_raw_error_data()


@pytest.mark.unit
def test_validate_rejects_identity_refactoring() -> None:
    # Refactoring-identita': baseline e new con metriche e logicalQubits identici. Compila ed e'
    # equivalente, ma non migliora nulla: deve risultare non valido, col messaggio che riporta i
    # valori confrontati.
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = True
    facade.calculate_metrics.side_effect = _metrics_side_effect(BASELINE_METRICS, BASELINE_METRICS)

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    assert result.get_new_metrics() is None
    error = result.get_raw_error_data()
    assert error is not None
    assert "gateCount=10" in error
    assert "depth=8" in error
    assert "qubit=3" in error


@pytest.mark.unit
def test_validate_accepts_valid_improvement() -> None:
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = True
    facade.calculate_metrics.side_effect = _metrics_side_effect(BASELINE_METRICS, IMPROVED_METRICS)

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is True
    assert result.get_new_metrics() == IMPROVED_METRICS


@pytest.mark.unit
def test_validate_rejects_pareto_violation() -> None:
    # new ha logicalQubits minore (3 -> 2) MA physicalMetrics.gateCount maggiore (10 -> 12):
    # migliora una metrica ma ne peggiora un'altra, quindi viola Pareto ed e' non valido.
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = True
    pareto_violation = {
        "logicalQubits": 2,
        "abstractMetrics": {"gateCount": 5, "depth": 4},
        "physicalMetrics": {"gateCount": 12, "depth": 8},
    }
    facade.calculate_metrics.side_effect = _metrics_side_effect(BASELINE_METRICS, pareto_violation)

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is False
    assert result.get_new_metrics() is None


@pytest.mark.unit
def test_validate_never_calls_calculate_metrics_when_compile_fails() -> None:
    # Corto-circuito preservato: se compile_circuit fallisce, calculate_metrics non deve MAI
    # essere raggiunto.
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (False, "errore di compilazione")

    ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    facade.calculate_metrics.assert_not_called()


@pytest.mark.unit
def test_validate_caches_baseline_metrics_across_calls() -> None:
    # Due validate() con lo STESSO baseline_code ma new_code diversi: la baseline va transpilata
    # una sola volta (cache), i due new_code ogni volta.
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = True

    new_code_a = "qc = QuantumCircuit(1)\nqc.h(0)\n"
    new_code_b = "qc = QuantumCircuit(1)\nqc.y(0)\n"

    def _side_effect(code: str) -> dict:
        return BASELINE_METRICS if code == BASELINE_CODE else IMPROVED_METRICS

    facade.calculate_metrics.side_effect = _side_effect

    service = ValidationService(facade=facade)
    service.validate(BASELINE_CODE, new_code_a)
    service.validate(BASELINE_CODE, new_code_b)

    called_codes = [call.args[0] for call in facade.calculate_metrics.call_args_list]
    assert called_codes.count(BASELINE_CODE) == 1
    assert called_codes.count(new_code_a) == 1
    assert called_codes.count(new_code_b) == 1
