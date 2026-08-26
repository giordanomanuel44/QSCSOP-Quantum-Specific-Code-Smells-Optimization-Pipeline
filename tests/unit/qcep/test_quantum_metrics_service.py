"""Unit test del servizio che misura il baseline in QCEP.

La facade e' mockata: qui si verifica il CONTRATTO del record prodotto (il Listing 1.2 della
tesi) e la rete di sicurezza sul codice non fidato, non il comportamento di Qiskit.

I test sono stati riscritti insieme al contratto: il baseline non porta piu' logicalQubits,
abstractMetrics e physicalMetrics ma la sola misura QSMELL, e con loro e' sparita la
transpilazione da questo servizio.
"""

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

# Payload della facade, nella sua forma completa: porta piu' di quanto il record persista.
FACADE_PAYLOAD = {
    "longCircuit": {
        "maxOpsPerQubit": 7,
        "maxParallelOps": 5,
        "value": 35,
        "gateError": 0.00485,
        "errorFreeProbability": 0.8434,
    },
    "idleQubits": {"value": 3, "worstQubit": 0},
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
    mock_facade.calculate_smell_metrics.return_value = FACADE_PAYLOAD

    result = service.calculate_metrics(RAW_RECORD)

    assert result == {
        "circuitId": "bug_1",
        "datasetSource": "Bugs4Q",
        "baseline": {
            "sourceCode": VALID_SOURCE,
            "smellMetrics": {
                "maxOpsPerQubit": 7,
                "maxParallelOps": 5,
                "longCircuit": 35,
                "idleQubits": 3,
            },
        },
    }


@pytest.mark.unit
def test_the_record_carries_only_what_it_needs_to_persist(
    service: QuantumMetricsService, mock_facade: Mock
) -> None:
    """gateError, errorFreeProbability e worstQubit restano fuori dal record.

    I primi due sono la forma con cui il paper PRESENTA la metrica, non una misura da
    persistere; worstQubit e' un puntatore dentro al circuito che serve al momento della
    riparazione, e chi ne ha bisogno lo chiede alla facade.
    """
    mock_facade.calculate_smell_metrics.return_value = FACADE_PAYLOAD

    baseline = service.calculate_metrics(RAW_RECORD)["baseline"]

    assert set(baseline) == {"sourceCode", "smellMetrics"}
    assert set(baseline["smellMetrics"]) == {
        "maxOpsPerQubit",
        "maxParallelOps",
        "longCircuit",
        "idleQubits",
    }


@pytest.mark.unit
def test_the_service_touches_the_facade_exactly_once(
    service: QuantumMetricsService, mock_facade: Mock
) -> None:
    """Una sola misura per record, e nessun'altra chiamata alla facade.

    Prima ne servivano quattro (isolate + abstract + transpile + physical) e la transpilazione
    era l'operazione piu' costosa dell'intero servizio. Ora e' sparita anche dalla facade, quindi
    non e' piu' nemmeno invocabile: quello che questo test protegge e' che non ricompaiano
    chiamate superflue lungo il percorso.
    """
    mock_facade.calculate_smell_metrics.return_value = FACADE_PAYLOAD

    service.calculate_metrics(RAW_RECORD)

    assert [call[0] for call in mock_facade.mock_calls] == ["calculate_smell_metrics"]


@pytest.mark.unit
def test_calculate_metrics_returns_none_on_invalid_syntax_without_calling_facade(
    service: QuantumMetricsService, mock_facade: Mock
) -> None:
    raw_record = {**RAW_RECORD, "sourceCode": INVALID_SYNTAX_SOURCE}

    result = service.calculate_metrics(raw_record)

    assert result is None
    mock_facade.calculate_smell_metrics.assert_not_called()


@pytest.mark.unit
def test_calculate_metrics_returns_none_when_the_measure_raises(
    service: QuantumMetricsService, mock_facade: Mock
) -> None:
    """Il corpus e' codice non fidato: un'eccezione qui romperebbe il generator di QCEPMain.

    calculate_smell_metrics isola ed ESEGUE il sorgente, quindi puo' fallire in modi
    imprevedibili anche su codice sintatticamente valido.
    """
    mock_facade.calculate_smell_metrics.side_effect = ValueError("Nessun QuantumCircuit trovato.")

    assert service.calculate_metrics(RAW_RECORD) is None
