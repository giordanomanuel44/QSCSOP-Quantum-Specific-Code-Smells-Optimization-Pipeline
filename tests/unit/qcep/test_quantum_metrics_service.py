from unittest.mock import Mock

import pytest

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qcep.services.quantum_metrics_service import QuantumMetricsService

VALID_SOURCE = "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)\n"
INVALID_SYNTAX_SOURCE = "def broken(:\n    pass"

RAW_RECORD = {
    "circuitId": "bug_1",
    "datasetSource": "Bugs4Q",
    "sourceCode": VALID_SOURCE,
}


@pytest.fixture
def mock_facade() -> Mock:
    return Mock(spec=IQiskitFacade)


@pytest.fixture
def service(mock_facade: Mock) -> QuantumMetricsService:
    return QuantumMetricsService(facade=mock_facade)


@pytest.mark.unit
def test_calculate_metrics_returns_listing_1_2_structure(
    service: QuantumMetricsService, mock_facade: Mock
) -> None:
    mock_circuit = Mock(num_qubits=3)
    mock_transpiled_circuit = Mock(num_qubits=3)
    mock_facade.isolate_circuit.return_value = mock_circuit
    mock_facade.get_abstract_metrics.return_value = {"gateCount": 15, "depth": 8}
    mock_facade.transpile_circuit.return_value = mock_transpiled_circuit
    mock_facade.get_physical_metrics.return_value = {"gateCount": 42, "depth": 25}

    result = service.calculate_metrics(RAW_RECORD)

    assert result == {
        "circuitId": "bug_1",
        "datasetSource": "Bugs4Q",
        "baseline": {
            "sourceCode": VALID_SOURCE,
            "logicalQubits": 3,
            "abstractMetrics": {"gateCount": 15, "depth": 8},
            "physicalMetrics": {"gateCount": 42, "depth": 25},
        },
    }


@pytest.mark.unit
def test_calculate_metrics_returns_none_on_invalid_syntax_without_calling_facade(
    service: QuantumMetricsService, mock_facade: Mock
) -> None:
    raw_record = {**RAW_RECORD, "sourceCode": INVALID_SYNTAX_SOURCE}

    result = service.calculate_metrics(raw_record)

    assert result is None
    mock_facade.isolate_circuit.assert_not_called()
    mock_facade.get_abstract_metrics.assert_not_called()
    mock_facade.transpile_circuit.assert_not_called()
    mock_facade.get_physical_metrics.assert_not_called()


@pytest.mark.unit
def test_calculate_metrics_returns_none_when_isolate_circuit_raises(
    service: QuantumMetricsService, mock_facade: Mock
) -> None:
    mock_facade.isolate_circuit.side_effect = ValueError("Nessun QuantumCircuit trovato.")

    result = service.calculate_metrics(RAW_RECORD)

    assert result is None


@pytest.mark.unit
def test_calculate_metrics_returns_none_when_transpile_circuit_raises(
    service: QuantumMetricsService, mock_facade: Mock
) -> None:
    mock_facade.isolate_circuit.return_value = Mock(num_qubits=2)
    mock_facade.get_abstract_metrics.return_value = {"gateCount": 5, "depth": 3}
    mock_facade.transpile_circuit.side_effect = RuntimeError("Transpilazione fallita.")

    result = service.calculate_metrics(RAW_RECORD)

    assert result is None


@pytest.mark.unit
def test_logical_qubits_comes_from_circuit_num_qubits_not_abstract_metrics(
    service: QuantumMetricsService, mock_facade: Mock
) -> None:
    mock_facade.isolate_circuit.return_value = Mock(num_qubits=4)
    mock_facade.get_abstract_metrics.return_value = {
        "gateCount": 10,
        "depth": 6,
        "logicalQubits": 999,
    }
    mock_facade.transpile_circuit.return_value = Mock(num_qubits=4)
    mock_facade.get_physical_metrics.return_value = {"gateCount": 20, "depth": 12}

    result = service.calculate_metrics(RAW_RECORD)

    assert result["baseline"]["logicalQubits"] == 4
