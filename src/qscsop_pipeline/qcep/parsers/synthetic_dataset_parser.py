"""Parser del dataset sintetico: legge i circuiti verificati dal ground truth filtrato."""

import json
from collections.abc import Generator
from pathlib import Path

from qscsop_pipeline.qcep.interfaces.i_dataset_parser import IDatasetParser


class SyntheticDatasetParser(IDatasetParser):
    """Estrae record grezzi dalle righe di synthetic_ground_truth_f.jsonl.

    A differenza degli altri due parser il sorgente non e' una cartella di file .py, ma un
    unico JSON Lines con il codice inline: i circuiti sintetici nascono gia' accompagnati dai
    metadati di generazione, e solo la versione filtrata (scritta da
    scripts/synthetic_dataset/filter_verified.py) contiene esattamente i circuiti ritenuti
    affidabili -- la cartella data/raw/synthetic/ ne e' un sovrainsieme non filtrato.
    """

    _DATASET_SOURCE = "Synthetic"

    def __init__(self, dataset_path: str = "data/interim/synthetic_ground_truth_f.jsonl") -> None:
        self._dataset_path = Path(dataset_path)

    def extract_circuits(self) -> Generator[dict, None, None]:
        """Genera un record grezzo per ciascuna riga del ground truth filtrato.

        Legge una riga alla volta, senza caricare il file in memoria. I campi di ground truth
        (intended_smells e le verifiche di generazione) sono deliberatamente scartati: QCEP
        produce solo il baseline con le sue metriche, e il confronto tra smell attesi e smell
        rilevati resta un join su circuitId a valle, fuori dal dataset di pipeline.
        """
        with self._dataset_path.open("r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                record = json.loads(line)
                yield {
                    "circuitId": record["circuit_id"],
                    "datasetSource": self._DATASET_SOURCE,
                    "sourceCode": record["source_code"],
                }
