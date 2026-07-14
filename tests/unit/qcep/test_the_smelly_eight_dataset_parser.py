from pathlib import Path

import pytest

from qscsop_pipeline.qcep.parsers.the_smelly_eight_dataset_parser import (
    TheSmellyEightDatasetParser,
)


@pytest.fixture
def single_smell_dir(tmp_path: Path) -> Path:
    smell_dir = tmp_path / "lc"
    smell_dir.mkdir()
    (smell_dir / "lc-smelly.py").write_text(
        "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\n", encoding="utf-8"
    )
    (smell_dir / "lc-fixed.py").write_text(
        "from qiskit import QuantumCircuit\nqc = QuantumCircuit(1)\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def multi_smell_dir(tmp_path: Path) -> Path:
    for smell in ("lc", "idq"):
        smell_dir = tmp_path / smell
        smell_dir.mkdir()
        (smell_dir / f"{smell}-smelly.py").write_text(
            f"from qiskit import QuantumCircuit\nqc = QuantumCircuit(1)  # {smell}\n",
            encoding="utf-8",
        )
        (smell_dir / f"{smell}-fixed.py").write_text(
            f"from qiskit import QuantumCircuit\nqc = QuantumCircuit(1)  # {smell} fixed\n",
            encoding="utf-8",
        )
    return tmp_path


@pytest.mark.unit
def test_only_smelly_file_is_extracted(single_smell_dir: Path) -> None:
    parser = TheSmellyEightDatasetParser(dataset_path=str(single_smell_dir))

    records = list(parser.extract_circuits())

    assert len(records) == 1
    assert records[0]["circuitId"] == "lc-smelly"


@pytest.mark.unit
def test_extraction_is_recursive_across_smell_subfolders(multi_smell_dir: Path) -> None:
    parser = TheSmellyEightDatasetParser(dataset_path=str(multi_smell_dir))

    records = list(parser.extract_circuits())

    assert len(records) == 2
    assert {record["circuitId"] for record in records} == {"lc-smelly", "idq-smelly"}


@pytest.mark.unit
def test_dataset_source_is_the_smelly_eight(multi_smell_dir: Path) -> None:
    parser = TheSmellyEightDatasetParser(dataset_path=str(multi_smell_dir))

    records = list(parser.extract_circuits())

    assert all(record["datasetSource"] == "TheSmellyEight" for record in records)
