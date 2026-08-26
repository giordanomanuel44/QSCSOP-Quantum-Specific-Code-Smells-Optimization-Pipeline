import json
from pathlib import Path

import pytest

from qscsop_pipeline.qscsop.entities.circuit_version import CircuitVersion
from qscsop_pipeline.qscsop.entities.smell_metrics import SmellMetrics
from qscsop_pipeline.qscsop.entities.quantum_program_entity import QuantumProgramEntity
from qscsop_pipeline.qscsop.sinks.jsonl_data_sink import JsonlDataSink


def make_version(source_code: str = "qc = QuantumCircuit(2)") -> CircuitVersion:
    return CircuitVersion(
        source_code=source_code,
        smell_metrics=SmellMetrics(max_ops_per_qubit=2, max_parallel_ops=2, idle_qubits=0),
    )


def make_entity() -> QuantumProgramEntity:
    return QuantumProgramEntity(
        circuit_id="bell_state",
        dataset_source="Bugs4Q",
        baseline=make_version(),
    )


@pytest.mark.unit
def test_save_program_without_refactored_omits_key(tmp_path: Path) -> None:
    output_path = tmp_path / "output" / "risultati.jsonl"
    sink = JsonlDataSink(filepath=str(output_path))

    sink.save_program(make_entity())

    line = output_path.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(line)
    assert "refactored" not in record


@pytest.mark.unit
def test_save_program_with_refactored_includes_key(tmp_path: Path) -> None:
    output_path = tmp_path / "output" / "risultati.jsonl"
    sink = JsonlDataSink(filepath=str(output_path))
    entity = make_entity()
    refactored = make_version("qc = QuantumCircuit(2)  # optimized")
    entity.set_refactored_version(refactored)

    sink.save_program(entity)

    line = output_path.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(line)
    assert "refactored" in record
    assert record["refactored"] == refactored.to_dict()


@pytest.mark.unit
def test_save_program_appends_across_multiple_calls(tmp_path: Path) -> None:
    output_path = tmp_path / "risultati.jsonl"
    sink = JsonlDataSink(filepath=str(output_path))

    sink.save_program(make_entity())
    sink.save_program(make_entity())

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


@pytest.mark.unit
def test_save_program_creates_missing_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "deep" / "risultati.jsonl"
    sink = JsonlDataSink(filepath=str(output_path))

    sink.save_program(make_entity())

    assert output_path.exists()
