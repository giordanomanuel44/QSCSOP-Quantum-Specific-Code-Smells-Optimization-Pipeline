import pandas as pd
import pytest

from qscsop_pipeline.analytics.calculators.metrics_calculator import MetricsCalculator


def _baseline(gate_count=10, depth=8, logical_qubits=3):
    return {
        "sourceCode": "qc = QuantumCircuit(3)\n",
        "logicalQubits": logical_qubits,
        "abstractMetrics": {"gateCount": gate_count, "depth": depth},
        "physicalMetrics": {"gateCount": gate_count, "depth": depth},
    }


def _refactored(gate_count, depth, logical_qubits):
    return {
        "sourceCode": "qc = QuantumCircuit(3)\nqc.h(0)\n",
        "logicalQubits": logical_qubits,
        "abstractMetrics": {"gateCount": gate_count, "depth": depth},
        "physicalMetrics": {"gateCount": gate_count, "depth": depth},
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
    baseline_gate_count=10,
    baseline_depth=8,
    baseline_qubits=3,
    new_gate_count=8,
    new_depth=6,
    new_qubits=3,
    iteration_count=1,
):
    return {
        "circuitId": circuit_id,
        "datasetSource": dataset_source,
        "baseline": _baseline(baseline_gate_count, baseline_depth, baseline_qubits),
        "refactored": _refactored(new_gate_count, new_depth, new_qubits),
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
    # Equivalenza preservata: c1 (OPTIMIZED) + c3 (metrics_not_improved) -> 2/3.
    assert metrics["equivalenza_funzionale"] == pytest.approx(2 / 3)
    # Successo globale: c1 + c2 (not_equivalent) + c3 (metrics_not_improved) -> 3/3.
    assert metrics["tasso_successo_globale"] == pytest.approx(1.0)
    # Media iterazioni sullo stesso sottoinsieme: (1 + 3 + 2) / 3.
    assert metrics["numero_medio_iterazioni"] == pytest.approx(2.0)


@pytest.mark.unit
def test_calculate_excludes_unexpected_error_from_equivalence_and_success() -> None:
    df = _df(
        [
            _optimized("c1"),
            _opt_failed("c2", failure_reason="unexpected_error"),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    # unexpected_error non conta ne' come equivalenza preservata ne' come successo globale.
    assert metrics["equivalenza_funzionale"] == pytest.approx(0.5)
    assert metrics["tasso_successo_globale"] == pytest.approx(0.5)


@pytest.mark.unit
def test_calculate_returns_none_when_processed_subset_is_empty() -> None:
    df = _df([_smell_free("c0"), _smell_free("c1")])

    metrics = MetricsCalculator().calculate(df)

    assert metrics["tasso_risoluzione"] is None
    assert metrics["equivalenza_funzionale"] is None
    assert metrics["tasso_successo_globale"] is None
    assert metrics["numero_medio_iterazioni"] is None


@pytest.mark.unit
def test_calculate_category2_gate_and_depth_reduction_on_long_circuit_optimized() -> None:
    df = _df(
        [
            _optimized(
                "c1",
                detected_smells=("long_circuit",),
                baseline_gate_count=10,
                baseline_depth=8,
                new_gate_count=8,
                new_depth=4,
            ),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    # (10-8)/10*100 = 20%, (8-4)/8*100 = 50%.
    assert metrics["long_circuit_gate_reduction_pct"]["values"] == pytest.approx([20.0])
    assert metrics["long_circuit_gate_reduction_pct"]["mean"] == pytest.approx(20.0)
    assert metrics["long_circuit_depth_reduction_pct"]["values"] == pytest.approx([50.0])
    assert metrics["long_circuit_depth_reduction_pct"]["mean"] == pytest.approx(50.0)


@pytest.mark.unit
def test_calculate_category2_handles_double_smell_circuit_in_both_subsets() -> None:
    df = _df(
        [
            _optimized(
                "c1",
                detected_smells=("long_circuit", "idle_qubits"),
                baseline_gate_count=10,
                baseline_qubits=5,
                new_gate_count=5,
                new_qubits=2,
            ),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    # Lo stesso circuito contribuisce indipendentemente a entrambe le sotto-metriche.
    assert metrics["long_circuit_gate_reduction_pct"]["values"] == pytest.approx([50.0])
    assert metrics["idle_qubits_reduction"]["values"] == pytest.approx([3.0])


@pytest.mark.unit
def test_calculate_category2_empty_subset_returns_none_mean_and_empty_values() -> None:
    # Nessun OPTIMIZED con idle_qubits nel dataset.
    df = _df([_optimized("c1", detected_smells=("long_circuit",))])

    metrics = MetricsCalculator().calculate(df)

    assert metrics["idle_qubits_reduction"] == {"mean": None, "values": []}


@pytest.mark.unit
def test_calculate_handles_dataset_with_no_optimized_circuits_without_keyerror() -> None:
    # Nessun record ha "refactored": la colonna refactored.* e' del tutto assente dal DataFrame,
    # non semplicemente NaN. calculate() non deve sollevare KeyError.
    df = _df([_smell_free("c0"), _opt_failed("c1")])
    assert "refactored.physicalMetrics.gateCount" not in df.columns

    metrics = MetricsCalculator().calculate(df)

    assert metrics["long_circuit_gate_reduction_pct"] == {"mean": None, "values": []}
    assert metrics["long_circuit_depth_reduction_pct"] == {"mean": None, "values": []}
    assert metrics["idle_qubits_reduction"] == {"mean": None, "values": []}


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
def test_calculate_metriche_per_dataset_source_are_computed_independently() -> None:
    df = _df(
        [
            _optimized("c1", dataset_source="Bugs4Q"),
            _opt_failed("c2", dataset_source="Bugs4Q", failure_reason="not_equivalent"),
            _optimized("c3", dataset_source="TheSmellyEight"),
            _optimized("c4", dataset_source="TheSmellyEight"),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    assert metrics["metriche_per_dataset_source"]["Bugs4Q"]["tasso_risoluzione"] == pytest.approx(
        0.5
    )
    assert metrics["metriche_per_dataset_source"]["TheSmellyEight"][
        "tasso_risoluzione"
    ] == pytest.approx(1.0)


@pytest.mark.unit
def test_calculate_on_completely_empty_dataframe_does_not_raise() -> None:
    metrics = MetricsCalculator().calculate(pd.DataFrame())

    assert metrics["tasso_risoluzione"] is None
    assert metrics["equivalenza_funzionale"] is None
    assert metrics["tasso_successo_globale"] is None
    assert metrics["numero_medio_iterazioni"] is None
    assert metrics["long_circuit_gate_reduction_pct"] == {"mean": None, "values": []}
    assert metrics["distribuzione_failure_reason"] == {}
    assert metrics["distribuzione_stati"] == {}
    assert metrics["metriche_per_dataset_source"] == {}


@pytest.mark.unit
def test_pct_reduction_avoids_division_by_zero_on_degenerate_baseline() -> None:
    df = _df(
        [
            _optimized(
                "c1",
                detected_smells=("long_circuit",),
                baseline_gate_count=0,
                new_gate_count=0,
            ),
        ]
    )

    metrics = MetricsCalculator().calculate(df)

    # baseline=0 -> divisione per zero evitata, il valore e' scartato (NaN -> dropna), non 0/0.
    assert metrics["long_circuit_gate_reduction_pct"] == {"mean": None, "values": []}
