from unittest.mock import Mock, call

import pytest

from qscsop_pipeline.qcep.factories.dataset_parser_factory import DatasetParserFactory
from qscsop_pipeline.qcep.interfaces.i_dataset_writer import IDatasetWriter
from qscsop_pipeline.qcep.interfaces.i_quantum_metrics_service import IQuantumMetricsService
from qscsop_pipeline.qcep.main import QCEPMain


def make_parser(records: list[dict]) -> Mock:
    parser = Mock()
    parser.extract_circuits.return_value = iter(records)
    return parser


@pytest.fixture
def mock_metrics_service() -> Mock:
    return Mock(spec=IQuantumMetricsService)


@pytest.fixture
def mock_dataset_writer() -> Mock:
    return Mock(spec=IDatasetWriter)


@pytest.fixture
def mock_factory() -> Mock:
    return Mock(spec=DatasetParserFactory)


@pytest.mark.unit
def test_run_calls_get_parser_once_per_dataset_name_in_order(
    mock_factory: Mock, mock_metrics_service: Mock, mock_dataset_writer: Mock
) -> None:
    mock_factory.get_parser.side_effect = lambda name: make_parser([])
    qcep_main = QCEPMain(
        metrics_service=mock_metrics_service,
        dataset_writer=mock_dataset_writer,
        factory=mock_factory,
        dataset_names=["fake_a", "fake_b"],
    )

    qcep_main.run()

    assert mock_factory.get_parser.call_args_list == [call("fake_a"), call("fake_b")]


@pytest.mark.unit
def test_run_loads_every_processed_record_of_a_single_dataset(
    mock_factory: Mock, mock_metrics_service: Mock, mock_dataset_writer: Mock
) -> None:
    raw_records = [{"circuitId": "c1"}, {"circuitId": "c2"}]
    processed_records = [{"circuitId": "c1", "baseline": {}}, {"circuitId": "c2", "baseline": {}}]
    mock_factory.get_parser.return_value = make_parser(raw_records)
    mock_metrics_service.calculate_metrics.side_effect = processed_records
    qcep_main = QCEPMain(
        metrics_service=mock_metrics_service,
        dataset_writer=mock_dataset_writer,
        factory=mock_factory,
        dataset_names=["fake_a"],
    )

    qcep_main.run()

    assert mock_dataset_writer.load.call_count == 2
    assert mock_dataset_writer.load.call_args_list == [
        call(processed_records[0]),
        call(processed_records[1]),
    ]


@pytest.mark.unit
def test_run_skips_none_records_without_calling_writer_or_raising(
    mock_factory: Mock, mock_metrics_service: Mock, mock_dataset_writer: Mock
) -> None:
    raw_records = [{"circuitId": "c1"}, {"circuitId": "c2"}, {"circuitId": "c3"}]
    valid_a = {"circuitId": "c1", "baseline": {}}
    valid_c = {"circuitId": "c3", "baseline": {}}
    mock_factory.get_parser.return_value = make_parser(raw_records)
    mock_metrics_service.calculate_metrics.side_effect = [valid_a, None, valid_c]
    qcep_main = QCEPMain(
        metrics_service=mock_metrics_service,
        dataset_writer=mock_dataset_writer,
        factory=mock_factory,
        dataset_names=["fake_a"],
    )

    qcep_main.run()

    assert mock_dataset_writer.load.call_count == 2
    assert mock_dataset_writer.load.call_args_list == [call(valid_a), call(valid_c)]


@pytest.mark.unit
def test_run_processes_all_records_from_both_datasets_in_order(
    mock_factory: Mock, mock_metrics_service: Mock, mock_dataset_writer: Mock
) -> None:
    records_a = [{"circuitId": "a1"}, {"circuitId": "a2"}]
    records_b = [{"circuitId": "b1"}]
    parsers_by_name = {
        "a": make_parser(records_a),
        "b": make_parser(records_b),
    }
    mock_factory.get_parser.side_effect = lambda name: parsers_by_name[name]
    mock_metrics_service.calculate_metrics.side_effect = lambda raw: {**raw, "baseline": {}}
    qcep_main = QCEPMain(
        metrics_service=mock_metrics_service,
        dataset_writer=mock_dataset_writer,
        factory=mock_factory,
        dataset_names=["a", "b"],
    )

    qcep_main.run()

    assert mock_dataset_writer.load.call_count == 3
    processed_ids_in_order = [
        written_call.args[0]["circuitId"]
        for written_call in mock_dataset_writer.load.call_args_list
    ]
    assert processed_ids_in_order == ["a1", "a2", "b1"]


@pytest.mark.unit
def test_run_with_empty_dataset_names_calls_nothing(
    mock_factory: Mock, mock_metrics_service: Mock, mock_dataset_writer: Mock
) -> None:
    qcep_main = QCEPMain(
        metrics_service=mock_metrics_service,
        dataset_writer=mock_dataset_writer,
        factory=mock_factory,
        dataset_names=[],
    )

    qcep_main.run()

    mock_factory.get_parser.assert_not_called()
    mock_metrics_service.calculate_metrics.assert_not_called()
    mock_dataset_writer.load.assert_not_called()
