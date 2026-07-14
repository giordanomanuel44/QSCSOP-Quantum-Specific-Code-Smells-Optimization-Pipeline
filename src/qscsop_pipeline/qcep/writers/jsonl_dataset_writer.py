"""Implementazione concreta della persistenza JSON Lines per QCEP."""

import json
from pathlib import Path

from qscsop_pipeline.qcep.interfaces.i_dataset_writer import IDatasetWriter


class JsonlDatasetWriter(IDatasetWriter):
    """Appende ogni record ricevuto, minificato, su una riga del file JSON Lines di output."""

    def __init__(self, output_path: str) -> None:
        self._output_path = Path(output_path)

    def load(self, dict_with_metrics: dict) -> None:
        """Serializza dict_with_metrics in una riga JSON minificata e la appende su disco."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(dict_with_metrics, separators=(",", ":"))
        with self._output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(line + "\n")
