import json
from pathlib import Path

import pytest

from qscsop_pipeline.qcep.parsers.synthetic_dataset_parser import SyntheticDatasetParser


def _record(circuit_id: str, source_code: str, intended_smells: list[str]) -> dict:
    """Riga di ground truth filtrato, con gli stessi campi scritti da generate.py."""
    return {
        "circuit_id": circuit_id,
        "source_code": source_code,
        "intended_smells": intended_smells,
        "structural_check_passed": True,
        "duplicate_of": None,
        "theme_consistent": True,
        "generation_batch": "qubit_1_3_long_circuit",
    }


@pytest.fixture
def ground_truth_file(tmp_path: Path) -> Path:
    records = [
        _record(
            "synthetic_qubit_1_3_long_circuit_1",
            "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\n",
            ["long_circuit"],
        ),
        _record(
            "synthetic_qubit_4_6_idle_qubits_1",
            "from qiskit import QuantumCircuit\nqc = QuantumCircuit(5)\n",
            ["idle_qubits"],
        ),
    ]
    path = tmp_path / "synthetic_ground_truth_f.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


@pytest.mark.unit
def test_one_record_is_extracted_per_line(ground_truth_file: Path) -> None:
    parser = SyntheticDatasetParser(dataset_path=str(ground_truth_file))

    records = list(parser.extract_circuits())

    assert len(records) == 2
    assert [record["circuitId"] for record in records] == [
        "synthetic_qubit_1_3_long_circuit_1",
        "synthetic_qubit_4_6_idle_qubits_1",
    ]


@pytest.mark.unit
def test_source_code_is_taken_verbatim_from_the_record(ground_truth_file: Path) -> None:
    parser = SyntheticDatasetParser(dataset_path=str(ground_truth_file))

    records = list(parser.extract_circuits())

    assert records[0]["sourceCode"] == (
        "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\n"
    )


@pytest.mark.unit
def test_dataset_source_is_synthetic(ground_truth_file: Path) -> None:
    parser = SyntheticDatasetParser(dataset_path=str(ground_truth_file))

    records = list(parser.extract_circuits())

    assert all(record["datasetSource"] == "Synthetic" for record in records)


@pytest.mark.unit
def test_ground_truth_fields_are_not_propagated(ground_truth_file: Path) -> None:
    # intended_smells e le verifiche di generazione restano fuori dal dataset di pipeline:
    # il confronto con i detected_smells e' un join su circuitId a valle, non un campo di QCEP.
    parser = SyntheticDatasetParser(dataset_path=str(ground_truth_file))

    records = list(parser.extract_circuits())

    assert all(set(record) == {"circuitId", "datasetSource", "sourceCode"} for record in records)


@pytest.mark.unit
def test_empty_dataset_yields_no_record(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    parser = SyntheticDatasetParser(dataset_path=str(empty_file))

    assert list(parser.extract_circuits()) == []
