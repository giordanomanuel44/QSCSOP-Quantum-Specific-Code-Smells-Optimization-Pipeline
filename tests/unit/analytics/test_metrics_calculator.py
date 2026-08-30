import pandas as pd
import pytest

from qscsop_pipeline.analytics.calculators.metrics_calculator import MetricsCalculator


def _smell_metrics(long_circuit, idle_qubits):
    """Misura coerente: i due fattori si moltiplicano davvero nel prodotto dichiarato."""
    return {
        "maxOpsPerQubit": long_circuit,
        "maxParallelOps": 1,
        "longCircuit": long_circuit,
        "idleQubits": idle_qubits,
    }


def _baseline(long_circuit=10, idle_qubits=3):
    return {
        "sourceCode": "qc = QuantumCircuit(3)",
        "smellMetrics": _smell_metrics(long_circuit, idle_qubits),
    }


def _refactored(long_circuit, idle_qubits):
    return {
        "sourceCode": "qc = QuantumCircuit(3) + h",
        "smellMetrics": _smell_metrics(long_circuit, idle_qubits),
    }


def _smell_free(circuit_id="c", dataset_source="Bugs4Q"):
    return {
        "circuitId": circuit_id,
        "datasetSource": dataset_source,
        "baseline": _baseline(),
        "evaluation": {
            "isFunctionallyEquivalent": None,
            "iterationCount": 0,
            "status": "SMELL_FREE",
            "detected_smells": [],
        },
    }


def _opt_failed(
    circuit_id="c", dataset_source="Bugs4Q", failure_reason="not_equivalent", iteration_count=3
):
    return {
        "circuitId": circuit_id,
        "datasetSource": dataset_source,
        "baseline": _baseline(),
        "evaluation": {
            "isFunctionallyEquivalent": False,
            "iterationCount": iteration_count,
            "status": "OPT_FAILED",
            "detected_smells": ["long_circuit"],
            "failureReason": failure_reason,
        },
    }


def _optimized(
    circuit_id="c",
    dataset_source="Bugs4Q",
    detected_smells=("long_circuit",),
    baseline_long_circuit=10,
    baseline_idle_qubits=3,
    new_long_circuit=8,
    new_idle_qubits=1,
    iteration_count=1,
):
    return {
        "circuitId": circuit_id,
        "datasetSource": dataset_source,
        "baseline": _baseline(baseline_long_circuit, baseline_idle_qubits),
        "refactored": _refactored(new_long_circuit, new_idle_qubits),
        "evaluation": {
            "isFunctionallyEquivalent": True,
            "iterationCount": iteration_count,
            "status": "OPTIMIZED",
            "detected_smells": list(detected_smells),
        },
    }


def _df(records: list[dict]) -> pd.DataFrame:
    return pd.json_normalize(records)


@pytest.mark.unit
def test_calculate_computes_category1_and_3_rates_on_processed_subset_only() -> None:
    df = _df(
        [
            _smell_free("c0"),
            _optimized("c1", iteration_count=1),
            _opt_failed("c2", failure_reason="not_equivalent", iteration_count=3),
            _opt_failed("c3", failure_reason="metrics_not_improved", iteration_count=2),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    # Sottoinsieme processato: c1, c2, c3 (esclude c0 SMELL_FREE) -> denominatore 3.
    assert metrics["tasso_risoluzione"] == pytest.approx(1 / 3)
    # Successo globale: c1 + c2 (not_equivalent) + c3 (metrics_not_improved) -> 3/3.
    assert metrics["tasso_successo_globale"] == pytest.approx(1.0)


@pytest.mark.unit
def test_calculate_excludes_unexpected_error_from_global_success() -> None:
    df = _df(
        [
            _optimized("c1"),
            _opt_failed("c2", failure_reason="unexpected_error"),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    # unexpected_error non conta come successo globale: un crash imprevisto non garantisce che
    # sia mai stato prodotto codice valido.
    assert metrics["tasso_successo_globale"] == pytest.approx(0.5)


@pytest.mark.unit
def test_calculate_returns_none_when_processed_subset_is_empty() -> None:
    df = _df([_smell_free("c0"), _smell_free("c1")])

    metrics = MetricsCalculator().calculate(df)

    assert metrics["tasso_risoluzione"] is None
    assert metrics["tasso_successo_globale"] is None
    assert metrics["iterazioni_alla_convergenza"] == {"mean": None, "distribution": {}, "n": 0}


@pytest.mark.unit
def test_iterazioni_alla_convergenza_ignores_failed_circuits_stuck_at_max_iterations() -> None:
    """Il difetto che la metrica sostituisce: gli OPT_FAILED sono tutti al soffitto per
    costruzione, e mediarli coi convergenti censurava il risultato."""
    df = _df(
        [
            _optimized("c1", iteration_count=1),
            _optimized("c2", iteration_count=1),
            _optimized("c3", iteration_count=2),
            _opt_failed("c4", iteration_count=3),
            _opt_failed("c5", iteration_count=3),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    # Media sui soli OPTIMIZED: (1 + 1 + 2) / 3, non (1+1+2+3+3)/5 = 2.0.
    assert metrics["iterazioni_alla_convergenza"]["mean"] == pytest.approx(4 / 3)
    assert metrics["iterazioni_alla_convergenza"]["distribution"] == {1: 2, 2: 1}
    assert metrics["iterazioni_alla_convergenza"]["n"] == 3


@pytest.mark.unit
def test_calculate_category2_gate_and_depth_reduction_on_long_circuit_optimized() -> None:
    df = _df(
        [
            _optimized(
                "c1",
                detected_smells=("long_circuit",),
                baseline_long_circuit=10,
                new_long_circuit=8,
            ),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    # (10-8)/10*100 = 20%.
    assert metrics["long_circuit_reduction_pct"]["values"] == pytest.approx([20.0])
    assert metrics["long_circuit_reduction_pct"]["mean"] == pytest.approx(20.0)


@pytest.mark.unit
def test_calculate_category2_handles_double_smell_circuit_in_both_subsets() -> None:
    df = _df(
        [
            _optimized(
                "c1",
                detected_smells=("long_circuit", "idle_qubits"),
                baseline_long_circuit=10,
                baseline_idle_qubits=5,
                new_long_circuit=5,
                new_idle_qubits=2,
            ),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    # Lo stesso circuito contribuisce indipendentemente a entrambe le sotto-metriche.
    assert metrics["long_circuit_reduction_pct"]["values"] == pytest.approx([50.0])
    assert metrics["idle_qubits_reduction"]["values"] == pytest.approx([3.0])


@pytest.mark.unit
def test_calculate_category2_empty_subset_returns_none_mean_and_empty_values() -> None:
    # Nessun OPTIMIZED con idle_qubits nel dataset.
    df = _df([_optimized("c1", detected_smells=("long_circuit",))])

    metrics = MetricsCalculator().calculate(df)

    assert metrics["idle_qubits_reduction"] == {"mean": None, "values": [], "labels": []}


@pytest.mark.unit
def test_calculate_handles_dataset_with_no_optimized_circuits_without_keyerror() -> None:
    # Nessun record ha "refactored": la colonna refactored.* e' del tutto assente dal DataFrame,
    # non semplicemente NaN. calculate() non deve sollevare KeyError.
    df = _df([_smell_free("c0"), _opt_failed("c1")])
    assert "refactored.smellMetrics.longCircuit" not in df.columns

    metrics = MetricsCalculator().calculate(df)

    assert metrics["long_circuit_reduction_pct"] == {"mean": None, "values": [], "labels": []}
    assert metrics["idle_qubits_reduction"] == {"mean": None, "values": [], "labels": []}


@pytest.mark.unit
def test_calculate_distribuzione_stati_and_failure_reason() -> None:
    df = _df(
        [
            _smell_free("c0"),
            _optimized("c1"),
            _optimized("c2"),
            _opt_failed("c3", failure_reason="not_equivalent"),
            _opt_failed("c4", failure_reason="not_equivalent"),
            _opt_failed("c5", failure_reason="compilation_failed"),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    assert metrics["distribuzione_stati"] == {"OPTIMIZED": 2, "OPT_FAILED": 3, "SMELL_FREE": 1}
    assert metrics["distribuzione_failure_reason"] == {
        "not_equivalent": 2,
        "compilation_failed": 1,
    }


@pytest.mark.unit
def test_rientro_sotto_soglia_distinguishes_improved_from_actually_resolved() -> None:
    """Il punto della metrica: una riduzione ampia non implica che lo smell sia stato rimosso."""
    df = _df(
        [
            # -48% su l*c ma ancora sopra la soglia inclusiva di 20: migliorato, non risolto.
            _optimized(
                "c1",
                detected_smells=("long_circuit",),
                baseline_long_circuit=50,
                new_long_circuit=26,
                baseline_idle_qubits=0,
                new_idle_qubits=0,
            ),
            _optimized(
                "c2",
                detected_smells=("long_circuit",),
                baseline_long_circuit=50,
                new_long_circuit=6,
                baseline_idle_qubits=0,
                new_idle_qubits=0,
            ),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    assert metrics["rientro_sotto_soglia"]["long_circuit"] == {
        "count": 1,
        "total": 2,
        "rate": pytest.approx(0.5),
    }
    # c1 resta smelly su l*c, quindi non e' privo di smell residui nonostante IdQ = 0.
    assert metrics["rientro_sotto_soglia"]["smell_free"]["count"] == 1


@pytest.mark.unit
def test_rientro_sotto_soglia_uses_strict_comparison_for_idle_qubits() -> None:
    """IdQ e' smelly a > 0: una sola colonna di attesa residua non e' un rientro."""
    df = _df(
        [
            _optimized(
                "c1",
                detected_smells=("idle_qubits",),
                baseline_idle_qubits=3,
                new_idle_qubits=1,
            ),
            _optimized(
                "c2",
                detected_smells=("idle_qubits",),
                baseline_idle_qubits=3,
                new_idle_qubits=0,
            ),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    assert metrics["rientro_sotto_soglia"]["idle_qubits"] == {
        "count": 1,
        "total": 2,
        "rate": pytest.approx(0.5),
    }


@pytest.mark.unit
def test_rientro_sotto_soglia_is_empty_without_optimized_circuits() -> None:
    df = _df([_smell_free("c0"), _opt_failed("c1")])

    recovery = MetricsCalculator().calculate(df)["rientro_sotto_soglia"]

    for key in ("long_circuit", "idle_qubits", "smell_free"):
        assert recovery[key] == {"count": 0, "total": 0, "rate": None}


@pytest.mark.unit
def test_calculate_on_completely_empty_dataframe_does_not_raise() -> None:
    metrics = MetricsCalculator().calculate(pd.DataFrame())

    assert metrics["tasso_risoluzione"] is None
    assert metrics["tasso_successo_globale"] is None
    assert metrics["iterazioni_alla_convergenza"] == {"mean": None, "distribution": {}, "n": 0}
    assert metrics["long_circuit_reduction_pct"] == {"mean": None, "values": [], "labels": []}
    assert metrics["rientro_sotto_soglia"]["smell_free"] == {
        "count": 0,
        "total": 0,
        "rate": None,
    }
    assert metrics["distribuzione_failure_reason"] == {}
    assert metrics["distribuzione_stati"] == {}
    assert "equivalenza_funzionale" not in metrics
    assert "metriche_per_dataset_source" not in metrics


@pytest.mark.unit
def test_pct_reduction_avoids_division_by_zero_on_degenerate_baseline() -> None:
    df = _df(
        [
            _optimized(
                "c1",
                detected_smells=("long_circuit",),
                baseline_long_circuit=0,
                new_long_circuit=0,
            ),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    # baseline=0 -> divisione per zero evitata, il valore e' scartato (NaN -> dropna), non 0/0.
    assert metrics["long_circuit_reduction_pct"] == {"mean": None, "values": [], "labels": []}
