"""Orchestratore radice del modulo Analytics: coordina caricamento, calcolo e visualizzazione."""

from qscsop_pipeline.analytics.interfaces.i_data_loader import IDataLoader
from qscsop_pipeline.analytics.interfaces.i_metrics_calculator import IMetricsCalculator
from qscsop_pipeline.analytics.interfaces.i_report_visualizer import IReportVisualizer


class AnalyticsMain:
    """Coordina DataLoader, MetricsCalculator e ReportVisualizer (sezione 1.5).

    Componente radice del modulo Analytics, equivalente di QCEPMain/PipelineCoordinator:
    nessuna interfaccia dedicata, non viene iniettato altrove.
    """

    def __init__(
        self,
        data_loader: IDataLoader,
        metrics_calculator: IMetricsCalculator,
        report_visualizer: IReportVisualizer,
    ) -> None:
        self._data_loader = data_loader
        self._metrics_calculator = metrics_calculator
        self._report_visualizer = report_visualizer

    def run(self) -> None:
        """Carica il dataset, calcola le metriche aggregate e genera il report grafico."""
        df = self._data_loader.load()
        metrics = self._metrics_calculator.calculate(df)
        self._report_visualizer.visualize(df, metrics)
