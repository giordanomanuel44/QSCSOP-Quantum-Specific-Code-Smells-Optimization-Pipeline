import inspect
from pathlib import Path

import pytest

from qscsop_pipeline.qcep.parsers.bugs4q_dataset_parser import Bugs4QDatasetParser

FIXTURE_FILES = {
    "Aer_bug_1_fixed.py": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(1)\n",
    "StackExchange_1_fix.py": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\n",
    "Program_1999_fixed.py": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(3)\n",
}


@pytest.fixture
def bugs4q_dir(tmp_path: Path) -> Path:
    for filename, content in FIXTURE_FILES.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")
    return tmp_path


@pytest.mark.unit
def test_extract_circuits_produces_one_record_per_file(bugs4q_dir: Path) -> None:
    parser = Bugs4QDatasetParser(dataset_path=str(bugs4q_dir))

    records = list(parser.extract_circuits())

    assert len(records) == len(FIXTURE_FILES)
    expected_ids = {Path(name).stem for name in FIXTURE_FILES}
    assert {record["circuitId"] for record in records} == expected_ids
    assert all(record["datasetSource"] == "Bugs4Q" for record in records)


@pytest.mark.unit
def test_source_code_matches_fixture_content(bugs4q_dir: Path) -> None:
    parser = Bugs4QDatasetParser(dataset_path=str(bugs4q_dir))

    records = {record["circuitId"]: record for record in parser.extract_circuits()}

    for filename, content in FIXTURE_FILES.items():
        circuit_id = Path(filename).stem
        assert records[circuit_id]["sourceCode"] == content


@pytest.mark.unit
def test_extract_circuits_is_a_generator(bugs4q_dir: Path) -> None:
    parser = Bugs4QDatasetParser(dataset_path=str(bugs4q_dir))

    result = parser.extract_circuits()

    assert inspect.isgenerator(result)
    first_record = next(result)
    assert first_record["datasetSource"] == "Bugs4Q"
