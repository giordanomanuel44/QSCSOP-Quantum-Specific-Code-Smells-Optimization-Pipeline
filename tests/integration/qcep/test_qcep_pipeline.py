import json
from pathlib import Path

import pytest

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qcep.interfaces.i_dataset_parser import IDatasetParser
from qscsop_pipeline.qcep.main import QCEPMain
from qscsop_pipeline.qcep.parsers.bugs4q_dataset_parser import Bugs4QDatasetParser
from qscsop_pipeline.qcep.parsers.the_smelly_eight_dataset_parser import (
    TheSmellyEightDatasetParser,
)
from qscsop_pipeline.qcep.services.quantum_metrics_service import QuantumMetricsService
from qscsop_pipeline.qcep.writers.jsonl_dataset_writer import JsonlDatasetWriter

BELL_STATE_SOURCE = """from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
"""

BROKEN_SOURCE = "def broken(:\n    pass"

LC_SMELLY_SOURCE = """from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.h(1)
qc.h(2)
"""

# Contenuto deliberatamente diverso (5 qubit) da LC_SMELLY_SOURCE: se questo file venisse
# erroneamente estratto nonostante il filtro sul suffisso -fixed.py, il test lo rileverebbe
# subito (nessun record con 5 qubit e' atteso in output).
LC_FIXED_SOURCE = """from qiskit import QuantumCircuit

qc = QuantumCircuit(5)
qc.h(0)
"""


class _FixtureDatasetParserFactory:
    """Fake minimale che risolve i nomi dataset verso parser di fixture gia' istanziati.

    La DatasetParserFactory reale e' gia' coperta da propri unit test (mapping nomi->classi
    e risoluzione dei path di produzione); qui l'obiettivo e' verificare il cablaggio end-to-end
    tra parser, QiskitFacade, QuantumMetricsService e JsonlDatasetWriter, non ri-testare la
    factory ne' toccare i dataset reali in data/raw/.
    """

    def __init__(self, parsers_by_name: dict[str, IDatasetParser]) -> None:
        self._parsers_by_name = parsers_by_name

    def get_parser(self, dataset_name: str) -> IDatasetParser:
        return self._parsers_by_name[dataset_name]


@pytest.fixture
def raw_dataset_root(tmp_path: Path) -> Path:
    bugs4q_dir = tmp_path / "bugs4q_fixture"
    bugs4q_dir.mkdir()
    (bugs4q_dir / "bell_state.py").write_text(BELL_STATE_SOURCE, encoding="utf-8")
    (bugs4q_dir / "broken_script.py").write_text(BROKEN_SOURCE, encoding="utf-8")

    smelly_lc_dir = tmp_path / "thesmellyeight_fixture" / "lc"
    smelly_lc_dir.mkdir(parents=True)
    (smelly_lc_dir / "lc-smelly.py").write_text(LC_SMELLY_SOURCE, encoding="utf-8")
    (smelly_lc_dir / "lc-fixed.py").write_text(LC_FIXED_SOURCE, encoding="utf-8")

    return tmp_path


@pytest.mark.integration
def test_qcep_main_end_to_end_with_real_dependencies(
    raw_dataset_root: Path, tmp_path: Path
) -> None:
    facade = QiskitFacade()
    bugs4q_parser = Bugs4QDatasetParser(dataset_path=str(raw_dataset_root / "bugs4q_fixture"))
    smelly_parser = TheSmellyEightDatasetParser(
        dataset_path=str(raw_dataset_root / "thesmellyeight_fixture")
    )
    metrics_service = QuantumMetricsService(facade=facade)
    output_path = tmp_path / "output" / "dataset_pulito.jsonl"
    dataset_writer = JsonlDatasetWriter(output_path=str(output_path))
    factory = _FixtureDatasetParserFactory(
        {"bugs4q": bugs4q_parser, "thesmellyeight": smelly_parser}
    )

    qcep_main = QCEPMain(
        metrics_service=metrics_service,
        dataset_writer=dataset_writer,
        factory=factory,
        dataset_names=["bugs4q", "thesmellyeight"],
    )
    qcep_main.run()

    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]
    records_by_source = {record["datasetSource"]: record for record in records}

    assert set(records_by_source) == {"Bugs4Q", "TheSmellyEight"}
    circuit_ids = {record["circuitId"] for record in records}
    assert circuit_ids == {"bell_state", "lc-smelly"}

    for record in records:
        assert isinstance(record["circuitId"], str)
        assert isinstance(record["datasetSource"], str)
        baseline = record["baseline"]
        assert isinstance(baseline["sourceCode"], str)
        # Il baseline porta esattamente due chiavi: il sorgente e la misura degli smell. Le
        # metriche di costo (logicalQubits, abstractMetrics, physicalMetrics) sono uscite dal
        # contratto insieme alla transpilazione.
        assert set(baseline) == {"sourceCode", "smellMetrics"}
        for measure in baseline["smellMetrics"].values():
            assert isinstance(measure, int)

    # Valori misurati sulle due fixture con la facade reale, non stimati.
    bugs4q_record = records_by_source["Bugs4Q"]
    assert bugs4q_record["circuitId"] == "bell_state"
    assert bugs4q_record["baseline"]["smellMetrics"] == {
        "maxOpsPerQubit": 2,
        "maxParallelOps": 2,
        "longCircuit": 4,
        "idleQubits": 0,
    }

    # h su tre qubit disgiunti: scivolano tutti nella stessa colonna, quindi c = 3 con l = 1.
    smelly_record = records_by_source["TheSmellyEight"]
    assert smelly_record["circuitId"] == "lc-smelly"
    assert smelly_record["baseline"]["smellMetrics"] == {
        "maxOpsPerQubit": 1,
        "maxParallelOps": 3,
        "longCircuit": 3,
        "idleQubits": 0,
    }
