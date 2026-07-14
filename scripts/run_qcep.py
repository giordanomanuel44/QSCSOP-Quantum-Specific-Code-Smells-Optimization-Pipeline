"""
Script di esecuzione della pipeline QCEP sui dataset reali in data/raw/.

Istanzia le dipendenze concrete (QiskitFacade, QuantumMetricsService,
JsonlDatasetWriter, DatasetParserFactory) e le inietta in QCEPMain, poi lancia
il ciclo ETL end-to-end. Produce data/interim/dataset_pulito.jsonl.

Uso (dalla root del progetto, con il venv attivo):
    python scripts/run_qcep.py
"""

from pathlib import Path

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qcep.factories.dataset_parser_factory import DatasetParserFactory
from qscsop_pipeline.qcep.main import QCEPMain
from qscsop_pipeline.qcep.services.quantum_metrics_service import QuantumMetricsService
from qscsop_pipeline.qcep.writers.jsonl_dataset_writer import JsonlDatasetWriter

# scripts/run_qcep.py -> risale alla root del progetto, stesso pattern di fetch_datasets.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "dataset_pulito.jsonl"


if __name__ == "__main__":
    facade = QiskitFacade()
    metrics_service = QuantumMetricsService(facade=facade)
    dataset_writer = JsonlDatasetWriter(output_path=str(OUTPUT_PATH))
    factory = DatasetParserFactory()

    qcep_main = QCEPMain(
        metrics_service=metrics_service,
        dataset_writer=dataset_writer,
        factory=factory,
    )
    qcep_main.run()

    print(f"QCEP completato. Output scritto in {OUTPUT_PATH}")
