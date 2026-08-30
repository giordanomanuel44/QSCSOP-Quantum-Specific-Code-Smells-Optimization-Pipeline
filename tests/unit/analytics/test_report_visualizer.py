import pandas as pd
import pytest

from qscsop_pipeline.analytics.visualizers.report_visualizer import ReportVisualizer

_EMPTY_SHARE = {"count": 0, "total": 0, "rate": None}

_FULL_METRICS = {
    "tasso_risoluzione": 0.5,
    "tasso_successo_globale": 1.0,
    "iterazioni_alla_convergenza": {"mean": 1.25, "distribution": {1: 3, 2: 1}, "n": 4},
    "long_circuit_reduction_pct": {
        "mean": 20.0,
        "values": [10.0, 30.0],
        "labels": ["c1", "c2"],
    },
    "idle_qubits_reduction": {"mean": 2.0, "values": [1, 2, 3], "labels": ["c1", "c2", "c3"]},
    "rientro_sotto_soglia": {
        "long_circuit": {"count": 3, "total": 5, "rate": 0.6},
        "idle_qubits": {"count": 6, "total": 6, "rate": 1.0},
        "smell_free": {"count": 4, "total": 6, "rate": 4 / 6},
    },
    "distribuzione_failure_reason": {"not_equivalent": 2, "compilation_failed": 1},
    "distribuzione_stati": {"OPTIMIZED": 2, "OPT_FAILED": 3, "SMELL_FREE": 1},
}

_EMPTY_METRICS = {
    "tasso_risoluzione": None,
    "tasso_successo_globale": None,
    "iterazioni_alla_convergenza": {"mean": None, "distribution": {}, "n": 0},
    "long_circuit_reduction_pct": {"mean": None, "values": [], "labels": []},
    "idle_qubits_reduction": {"mean": None, "values": [], "labels": []},
    "rientro_sotto_soglia": {
        "long_circuit": dict(_EMPTY_SHARE),
        "idle_qubits": dict(_EMPTY_SHARE),
        "smell_free": dict(_EMPTY_SHARE),
    },
    "distribuzione_failure_reason": {},
    "distribuzione_stati": {},
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
def test_visualize_handles_single_value_reduction_subsets(tmp_path) -> None:
    """n = 1: una sola barra per pannello, con la linea della media sovrapposta."""
    metrics = dict(_FULL_METRICS)
    metrics["long_circuit_reduction_pct"] = {"mean": 42.0, "values": [42.0], "labels": ["c1"]}
    metrics["idle_qubits_reduction"] = {"mean": 1.0, "values": [1], "labels": ["c1"]}
    output_path = tmp_path / "Report_Metriche.pdf"

    ReportVisualizer(output_path=str(output_path)).visualize(_sample_df(), metrics)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.unit
def test_visualize_creates_missing_output_directory(tmp_path) -> None:
    output_path = tmp_path / "nested" / "does" / "not" / "exist" / "Report_Metriche.pdf"

    ReportVisualizer(output_path=str(output_path)).visualize(_sample_df(), _FULL_METRICS)

    assert output_path.exists()


@pytest.mark.unit
def test_visualize_renders_reduction_page_with_one_empty_panel(tmp_path) -> None:
    """Nessun OPTIMIZED con Idle Qubits: il pannello destro riporta il placeholder, ma la
    pagina resta e il pannello sinistro viene disegnato normalmente."""
    metrics = dict(_FULL_METRICS)
    metrics["idle_qubits_reduction"] = {"mean": None, "values": [], "labels": []}
    output_path = tmp_path / "Report_Metriche.pdf"

    ReportVisualizer(output_path=str(output_path)).visualize(_sample_df(), metrics)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
