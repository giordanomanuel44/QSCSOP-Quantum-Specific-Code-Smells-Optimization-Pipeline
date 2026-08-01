from unittest.mock import Mock

import pandas as pd
import pytest

from qscsop_pipeline.analytics.analytics_main import AnalyticsMain
from qscsop_pipeline.analytics.interfaces.i_data_loader import IDataLoader
from qscsop_pipeline.analytics.interfaces.i_metrics_calculator import IMetricsCalculator
from qscsop_pipeline.analytics.interfaces.i_report_visualizer import IReportVisualizer


def _make_collaborators() -> dict:
    return {
        "data_loader": Mock(spec=IDataLoader),
        "metrics_calculator": Mock(spec=IMetricsCalculator),
        "report_visualizer": Mock(spec=IReportVisualizer),
    }


@pytest.mark.unit
def test_run_calls_load_then_calculate_then_visualize_with_correct_arguments() -> None:
    collaborators = _make_collaborators()
    df = pd.DataFrame({"circuitId": ["c1"]})
    metrics = {"tasso_risoluzione": 1.0}
    collaborators["data_loader"].load.return_value = df
    collaborators["metrics_calculator"].calculate.return_value = metrics
    analytics_main = AnalyticsMain(**collaborators)

    analytics_main.run()

    collaborators["data_loader"].load.assert_called_once_with()
    collaborators["metrics_calculator"].calculate.assert_called_once_with(df)
    collaborators["report_visualizer"].visualize.assert_called_once_with(df, metrics)


@pytest.mark.unit
def test_run_calls_collaborators_in_order() -> None:
    collaborators = _make_collaborators()
    call_order: list[str] = []
    collaborators["data_loader"].load.side_effect = lambda: call_order.append("load") or pd.DataFrame()
    collaborators["metrics_calculator"].calculate.side_effect = (
        lambda df: call_order.append("calculate") or {}
    )
    collaborators["report_visualizer"].visualize.side_effect = (
        lambda df, metrics: call_order.append("visualize")
    )
    analytics_main = AnalyticsMain(**collaborators)

    analytics_main.run()

    assert call_order == ["load", "calculate", "visualize"]
