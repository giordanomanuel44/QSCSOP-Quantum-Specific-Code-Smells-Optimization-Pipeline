
"""Orchestratore radice del modulo QCEP: coordina estrazione, metriche e persistenza."""

from qscsop_pipeline.qcep.factories.dataset_parser_factory import DatasetParserFactory
from qscsop_pipeline.qcep.interfaces.i_dataset_writer import IDatasetWriter
from qscsop_pipeline.qcep.interfaces.i_quantum_metrics_service import IQuantumMetricsService


class QCEPMain:
    """Coordina il ciclo ETL di QCEP su piu' dataset, mantenendo una complessita' O(1)."""

    def __init__(
        self,
        metrics_service: IQuantumMetricsService,
        dataset_writer: IDatasetWriter,
        factory: DatasetParserFactory,
        dataset_names: list[str] | None = None,
    ) -> None:
        self._metrics_service = metrics_service
        self._dataset_writer = dataset_writer
        self._factory = factory
        self._dataset_names = (
            dataset_names
            if dataset_names is not None
            else [
                "bugs4q",
                "thesmellyeight",
            ]
        )

    def run(self) -> None:
        """Estrae, valida e persiste i circuiti di ogni dataset configurato, uno alla volta."""
        for dataset_name in self._dataset_names:
            parser = self._factory.get_parser(dataset_name)
            for raw_record in parser.extract_circuits():
                processed = self._metrics_service.calculate_metrics(raw_record)
                if processed is not None:
                    self._dataset_writer.load(processed)
