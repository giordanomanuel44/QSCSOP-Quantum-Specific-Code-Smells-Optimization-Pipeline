"""Implementazione concreta della persistenza JSON Lines dei risultati di QSCSOP."""

import json
from pathlib import Path

from qscsop_pipeline.qscsop.entities.quantum_program_entity import QuantumProgramEntity
from qscsop_pipeline.qscsop.interfaces.i_data_sink import IDataSink


class JsonlDataSink(IDataSink):
    """Appende ogni entità ricevuta, minificata, su una riga del file JSON Lines di output."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def save_program(self, entity: QuantumProgramEntity) -> None:
        """Serializza entity in una riga JSON minificata e la appende su disco."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(entity.to_dict(), separators=(",", ":"))
        with self._filepath.open("a", encoding="utf-8") as output_file:
            output_file.write(line + "\n")
