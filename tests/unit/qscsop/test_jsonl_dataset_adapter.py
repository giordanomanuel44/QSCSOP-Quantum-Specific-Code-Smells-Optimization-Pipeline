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
        "smellMetrics": {
            "maxOpsPerQubit": 2,
            "maxParallelOps": 2,
            "longCircuit": 4,
            "idleQubits": 0,
        },
    },
}

RECORD_2 = {
    "circuitId": "lc-smelly",
    "datasetSource": "TheSmellyEight",
    "baseline": {
        "sourceCode": "qc = QuantumCircuit(3)",
        "smellMetrics": {
            "maxOpsPerQubit": 7,
            "maxParallelOps": 5,
            "longCircuit": 35,
            "idleQubits": 3,
        },
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
    assert baseline.get_smell_metrics().get_max_ops_per_qubit() == 7
    assert baseline.get_smell_metrics().get_max_parallel_ops() == 5
    assert baseline.get_smell_metrics().get_idle_qubits() == 3


@pytest.mark.unit
def test_long_circuit_is_recomputed_and_not_trusted_from_the_file(tmp_path: Path) -> None:
    """Il prodotto nel file viene IGNORATO: l'entita' lo riderivera' dai due fattori.

    Qui il record porta un longCircuit deliberatamente falso (999 invece di 35). Se l'adapter lo
    leggesse, sarebbe possibile caricare un'entita' incoerente coi propri fattori -- l'esatta
    incoerenza che la property di SmellMetrics esiste per rendere irrappresentabile.
    """
    tampered = {
        **RECORD_2,
        "baseline": {
            **RECORD_2["baseline"],
            "smellMetrics": {**RECORD_2["baseline"]["smellMetrics"], "longCircuit": 999},
        },
    }
    filepath = write_jsonl(tmp_path / "dataset.jsonl", [tampered])

    baseline = next(JsonlDatasetAdapter(filepath=str(filepath)).stream_programs()).get_baseline()

    assert baseline.get_smell_metrics().long_circuit == 35


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
