import json

import pandas as pd
import pytest

from qscsop_pipeline.analytics.loaders.data_loader import DataLoader

_SMELL_FREE_RECORD = {
    "circuitId": "c1",
    "datasetSource": "Bugs4Q",
    "baseline": {
        "sourceCode": "qc = QuantumCircuit(1)\n",
        "smellMetrics": {
            "maxOpsPerQubit": 1,
            "maxParallelOps": 1,
            "longCircuit": 1,
            "idleQubits": 0,
        },
    },
    "evaluation": {
        "isFunctionallyEquivalent": None,
        "iterationCount": 0,
        "status": "SMELL_FREE",
        "detected_smells": [],
    },
}

_OPT_FAILED_RECORD = {
    "circuitId": "c2",
    "datasetSource": "TheSmellyEight",
    "baseline": {
        "sourceCode": "qc = QuantumCircuit(2)\n",
        "smellMetrics": {
            "maxOpsPerQubit": 5,
            "maxParallelOps": 4,
            "longCircuit": 20,
            "idleQubits": 2,
        },
    },
    "evaluation": {
        "isFunctionallyEquivalent": False,
        "iterationCount": 3,
        "status": "OPT_FAILED",
        "detected_smells": ["long_circuit"],
        "failureReason": "not_equivalent",
    },
}

_OPTIMIZED_RECORD = {
    "circuitId": "c3",
    "datasetSource": "Bugs4Q",
    "baseline": {
        "sourceCode": "qc = QuantumCircuit(3)\n",
        "smellMetrics": {
            "maxOpsPerQubit": 7,
            "maxParallelOps": 5,
            "longCircuit": 35,
            "idleQubits": 3,
        },
    },
    "refactored": {
        "sourceCode": "qc = QuantumCircuit(3)\nqc.h(0)\n",
        "smellMetrics": {
            "maxOpsPerQubit": 4,
            "maxParallelOps": 5,
            "longCircuit": 20,
            "idleQubits": 0,
        },
    },
    "evaluation": {
        "isFunctionallyEquivalent": True,
        "iterationCount": 1,
        "status": "OPTIMIZED",
        "detected_smells": ["long_circuit"],
    },
}


def _write_jsonl(path, records: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return str(path)


@pytest.mark.unit
def test_load_returns_one_row_per_record(tmp_path) -> None:
    filepath = _write_jsonl(
        tmp_path / "risultati.jsonl",
        [_SMELL_FREE_RECORD, _OPT_FAILED_RECORD, _OPTIMIZED_RECORD],
    )

    df = DataLoader(filepath=filepath).load()

    assert len(df) == 3
    assert list(df["circuitId"]) == ["c1", "c2", "c3"]


@pytest.mark.unit
def test_load_flattens_nested_keys_with_dot_notation(tmp_path) -> None:
    filepath = _write_jsonl(tmp_path / "risultati.jsonl", [_OPTIMIZED_RECORD])

    df = DataLoader(filepath=filepath).load()

    assert df.loc[0, "baseline.smellMetrics.longCircuit"] == 35
    assert df.loc[0, "baseline.smellMetrics.idleQubits"] == 3
    assert df.loc[0, "refactored.smellMetrics.longCircuit"] == 20
    assert df.loc[0, "evaluation.status"] == "OPTIMIZED"


@pytest.mark.unit
def test_load_fills_missing_refactored_block_with_nan(tmp_path) -> None:
    # _SMELL_FREE_RECORD e _OPT_FAILED_RECORD non hanno "refactored".
    filepath = _write_jsonl(tmp_path / "risultati.jsonl", [_SMELL_FREE_RECORD, _OPT_FAILED_RECORD])

    df = DataLoader(filepath=filepath).load()

    assert "refactored.sourceCode" not in df.columns or df["refactored.sourceCode"].isna().all()


@pytest.mark.unit
def test_load_fills_missing_failure_reason_with_nan(tmp_path) -> None:
    # _SMELL_FREE_RECORD e _OPTIMIZED_RECORD non hanno "failureReason".
    filepath = _write_jsonl(
        tmp_path / "risultati.jsonl", [_SMELL_FREE_RECORD, _OPT_FAILED_RECORD, _OPTIMIZED_RECORD]
    )

    df = DataLoader(filepath=filepath).load()

    assert pd.isna(df.loc[0, "evaluation.failureReason"])
    assert df.loc[1, "evaluation.failureReason"] == "not_equivalent"
    assert pd.isna(df.loc[2, "evaluation.failureReason"])
