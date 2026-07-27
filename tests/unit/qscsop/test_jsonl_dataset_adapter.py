import json
from pathlib import Path
from types import GeneratorType

import pytest

from qscsop_pipeline.qscsop.adapters.jsonl_dataset_adapter import JsonlDatasetAdapter
from qscsop_pipeline.qscsop.entities.evaluation_status import EvaluationStatus

RECORD_1 = {
    "circuitId": "bell_state",
    "datasetSource": "Bugs4Q",
    "baseline": {
        "sourceCode": "qc = QuantumCircuit(2)",
        "logicalQubits": 2,
        "abstractMetrics": {"gateCount": 2, "depth": 2},
        "physicalMetrics": {"gateCount": 4, "depth": 3},
    },
}

RECORD_2 = {
    "circuitId": "lc-smelly",
    "datasetSource": "TheSmellyEight",
    "baseline": {
        "sourceCode": "qc = QuantumCircuit(3)",
        "logicalQubits": 3,
        "abstractMetrics": {"gateCount": 5, "depth": 4},
        "physicalMetrics": {"gateCount": 9, "depth": 7},
    },
}


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_stream_programs_produces_one_entity_per_record(tmp_path: Path) -> None:
    filepath = write_jsonl(tmp_path / "dataset.jsonl", [RECORD_1, RECORD_2])
    adapter = JsonlDatasetAdapter(filepath=str(filepath))

    entities = list(adapter.stream_programs())

    assert len(entities) == 2
    assert entities[0].get_circuit_id() == "bell_state"
    assert entities[0].get_dataset_source() == "Bugs4Q"
    assert entities[1].get_circuit_id() == "lc-smelly"
    assert entities[1].get_dataset_source() == "TheSmellyEight"


@pytest.mark.unit
def test_baseline_is_reconstructed_correctly(tmp_path: Path) -> None:
    filepath = write_jsonl(tmp_path / "dataset.jsonl", [RECORD_2])
    adapter = JsonlDatasetAdapter(filepath=str(filepath))

    entity = next(adapter.stream_programs())
    baseline = entity.get_baseline()

    assert baseline.get_source_code() == "qc = QuantumCircuit(3)"
    assert baseline.get_logical_qubits() == 3
    assert baseline.get_abstract_metrics().get_gate_count() == 5
    assert baseline.get_abstract_metrics().get_depth() == 4
    assert baseline.get_physical_metrics().get_gate_count() == 9
    assert baseline.get_physical_metrics().get_depth() == 7


@pytest.mark.unit
def test_reconstructed_entities_have_no_refactored_and_are_processing(tmp_path: Path) -> None:
    filepath = write_jsonl(tmp_path / "dataset.jsonl", [RECORD_1, RECORD_2])
    adapter = JsonlDatasetAdapter(filepath=str(filepath))

    for entity in adapter.stream_programs():
        assert entity.get_refactored() is None
        assert entity.get_evaluation().get_status() == EvaluationStatus.PROCESSING


@pytest.mark.unit
def test_stream_programs_is_a_generator(tmp_path: Path) -> None:
    filepath = write_jsonl(tmp_path / "dataset.jsonl", [RECORD_1])
    adapter = JsonlDatasetAdapter(filepath=str(filepath))

    result = adapter.stream_programs()

    assert isinstance(result, GeneratorType)


@pytest.mark.unit
def test_invalid_json_line_raises_instead_of_being_swallowed(tmp_path: Path) -> None:
    filepath = tmp_path / "dataset.jsonl"
    filepath.write_text('{"circuitId": "broken", "datasetSourc', encoding="utf-8")
    adapter = JsonlDatasetAdapter(filepath=str(filepath))

    with pytest.raises(json.JSONDecodeError):
        list(adapter.stream_programs())
