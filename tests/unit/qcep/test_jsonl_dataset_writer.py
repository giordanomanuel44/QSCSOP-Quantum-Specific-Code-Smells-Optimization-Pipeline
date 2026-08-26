import json
from pathlib import Path

import pytest

from qscsop_pipeline.qcep.writers.jsonl_dataset_writer import JsonlDatasetWriter

RECORD = {"circuitId": "bug_1", "datasetSource": "Bugs4Q"}

NESTED_RECORD = {
    "circuitId": "bug_1",
    "datasetSource": "Bugs4Q",
    "baseline": {
        "sourceCode": "from qiskit import *...",
        "smellMetrics": {
            "maxOpsPerQubit": 7,
            "maxParallelOps": 5,
            "longCircuit": 35,
            "idleQubits": 3,
        },
    },
}


@pytest.mark.unit
def test_load_writes_single_line_matching_input_dict(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset_pulito.jsonl"
    writer = JsonlDatasetWriter(output_path=str(output_path))

    writer.load(RECORD)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == RECORD


@pytest.mark.unit
def test_load_appends_across_multiple_calls(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset_pulito.jsonl"
    writer = JsonlDatasetWriter(output_path=str(output_path))
    second_record = {"circuitId": "bug_2", "datasetSource": "TheSmellyEight"}

    writer.load(RECORD)
    writer.load(second_record)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == RECORD
    assert json.loads(lines[1]) == second_record


@pytest.mark.unit
def test_load_writes_minified_json_line(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset_pulito.jsonl"
    writer = JsonlDatasetWriter(output_path=str(output_path))

    writer.load(RECORD)

    raw_line = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert ": " not in raw_line
    assert ", " not in raw_line


@pytest.mark.unit
def test_load_creates_missing_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "does_not_exist_yet" / "dataset_pulito.jsonl"
    writer = JsonlDatasetWriter(output_path=str(output_path))

    writer.load(RECORD)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8").splitlines()[0]) == RECORD


@pytest.mark.unit
def test_load_preserves_nested_structure(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset_pulito.jsonl"
    writer = JsonlDatasetWriter(output_path=str(output_path))

    writer.load(NESTED_RECORD)

    raw_line = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(raw_line) == NESTED_RECORD
