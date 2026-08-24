"""
Script di esecuzione della pipeline QCEP sui dataset di circuiti.

Istanzia le dipendenze concrete (QiskitFacade, QuantumMetricsService,
JsonlDatasetWriter, DatasetParserFactory) e le inietta in QCEPMain, poi lancia
il ciclo ETL end-to-end.

Uso (dalla root del progetto, con il venv attivo):
    python scripts/run_qcep.py                # dataset reali -> dataset_pulito.jsonl
    python scripts/run_qcep.py --synthetic    # dataset sintetico -> dataset_pulito_synthetic.jsonl

I due esperimenti scrivono su file distinti di proposito: JsonlDatasetWriter appende, quindi
un output condiviso mescolerebbe circuiti reali e sintetici a ogni riesecuzione.
"""

import argparse
from pathlib import Path

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qcep.factories.dataset_parser_factory import DatasetParserFactory
from qscsop_pipeline.qcep.main import QCEPMain
from qscsop_pipeline.qcep.services.quantum_metrics_service import QuantumMetricsService
from qscsop_pipeline.qcep.writers.jsonl_dataset_writer import JsonlDatasetWriter

# scripts/run_qcep.py -> risale alla root del progetto, stesso pattern di fetch_datasets.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INTERIM = PROJECT_ROOT / "data" / "interim"

OUTPUT_PATH = _INTERIM / "dataset_pulito.jsonl"
SYNTHETIC_OUTPUT_PATH = _INTERIM / "dataset_pulito_synthetic.jsonl"

DATASETS = ["bugs4q", "thesmellyeight"]
SYNTHETIC_DATASETS = ["synthetic"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Elabora il dataset sintetico verificato invece di Bugs4Q/TheSmellyEight.",
    )
    args = parser.parse_args()

    dataset_names = SYNTHETIC_DATASETS if args.synthetic else DATASETS
    output_path = SYNTHETIC_OUTPUT_PATH if args.synthetic else OUTPUT_PATH

    facade = QiskitFacade()
    metrics_service = QuantumMetricsService(facade=facade)
    dataset_writer = JsonlDatasetWriter(output_path=str(output_path))
    factory = DatasetParserFactory()

    qcep_main = QCEPMain(
        metrics_service=metrics_service,
        dataset_writer=dataset_writer,
        factory=factory,
        dataset_names=dataset_names,
    )
    qcep_main.run()

    print(f"QCEP completato ({', '.join(dataset_names)}). Output scritto in {output_path}")
