from unittest.mock import Mock

import pytest

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qscsop.mas.validation.validation_service import ValidationService

BASELINE_CODE = "qc = QuantumCircuit(1)\nqc.x(0)\n"
NEW_CODE = "qc = QuantumCircuit(1)\nqc.h(0)\nqc.z(0)\nqc.h(0)\n"


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
    expected_metrics = {
        "abstractMetrics": {"gateCount": 3, "depth": 3},
        "physicalMetrics": {"gateCount": 6, "depth": 5},
    }
    facade.calculate_metrics.return_value = expected_metrics

    result = ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    assert result.get_is_valid() is True
    assert result.get_raw_error_data() is None
    assert result.get_new_metrics() == expected_metrics


@pytest.mark.unit
def test_validate_calls_facade_methods_in_order() -> None:
    facade = Mock(spec=IQiskitFacade)
    facade.compile_circuit.return_value = (True, None)
    facade.check_equivalence.return_value = True
    facade.calculate_metrics.return_value = {"abstractMetrics": {}, "physicalMetrics": {}}

    ValidationService(facade=facade).validate(BASELINE_CODE, NEW_CODE)

    called_method_names = [call[0] for call in facade.mock_calls]
    assert called_method_names == ["compile_circuit", "check_equivalence", "calculate_metrics"]
