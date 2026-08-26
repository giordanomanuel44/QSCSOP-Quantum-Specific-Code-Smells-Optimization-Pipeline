import pandas as pd
import pytest

from qscsop_pipeline.analytics.visualizers.report_visualizer import ReportVisualizer

_FULL_METRICS = {
    "tasso_risoluzione": 0.5,
    "equivalenza_funzionale": 0.75,
    "tasso_successo_globale": 1.0,
    "numero_medio_iterazioni": 2.0,
    "long_circuit_reduction_pct": {"mean": 20.0, "values": [10.0, 30.0]},
    "idle_qubits_reduction": {"mean": 2.0, "values": [1, 2, 3]},
    "distribuzione_failure_reason": {"not_equivalent": 2, "compilation_failed": 1},
    "distribuzione_stati": {"OPTIMIZED": 2, "OPT_FAILED": 3, "SMELL_FREE": 1},
    "metriche_per_dataset_source": {
        "Bugs4Q": {"tasso_risoluzione": 0.5, "equivalenza_funzionale": 0.5},
        "TheSmellyEight": {"tasso_risoluzione": 1.0, "equivalenza_funzionale": 1.0},
    },
}

_EMPTY_METRICS = {
    "tasso_risoluzione": None,
    "equivalenza_funzionale": None,
    "tasso_successo_globale": None,
    "numero_medio_iterazioni": None,
    "long_circuit_reduction_pct": {"mean": None, "values": []},
    "idle_qubits_reduction": {"mean": None, "values": []},
    "distribuzione_failure_reason": {},
    "distribuzione_stati": {},
    "metriche_per_dataset_source": {},
}


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "evaluation.status": ["OPTIMIZED", "OPT_FAILED", "SMELL_FREE"],
            "evaluation.iterationCount": [1, 3, 0],
        }
    )


@pytest.mark.unit
def test_visualize_creates_pdf_file_on_disk_with_full_metrics(tmp_path) -> None:
    output_path = tmp_path / "Report_Metriche.pdf"

    ReportVisualizer(output_path=str(output_path)).visualize(_sample_df(), _FULL_METRICS)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.unit
def test_visualize_creates_pdf_file_with_empty_metrics_via_placeholder_pages(tmp_path) -> None:
    output_path = tmp_path / "Report_Metriche.pdf"

    ReportVisualizer(output_path=str(output_path)).visualize(pd.DataFrame(), _EMPTY_METRICS)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.unit
def test_visualize_creates_missing_output_directory(tmp_path) -> None:
    output_path = tmp_path / "nested" / "does" / "not" / "exist" / "Report_Metriche.pdf"

    ReportVisualizer(output_path=str(output_path)).visualize(_sample_df(), _FULL_METRICS)

    assert output_path.exists()
