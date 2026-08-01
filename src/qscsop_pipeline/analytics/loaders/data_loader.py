"""Implementazione concreta dell'ingestione JSON Lines del dataset prodotto da QSCSOP."""

import json
from pathlib import Path

import pandas as pd

from qscsop_pipeline.analytics.interfaces.i_data_loader import IDataLoader


class DataLoader(IDataLoader):
    """Legge risultati_dataset.jsonl e lo appiattisce in un DataFrame (sezione 1.5.1)."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> pd.DataFrame:
        """Legge il file riga per riga e appiattisce la gerarchia con json_normalize.

        Nessun vincolo O(1) qui (a differenza di QCEP/QSCSOP): la sezione 1.5.1 descrive
        esplicitamente il caricamento in un DataFrame Pandas in memoria. Le chiavi assenti
        (es. "refactored" sui circuiti SMELL_FREE, "failureReason" sui circuiti non falliti)
        vengono popolate automaticamente da Pandas con NaN, senza gestione speciale.
        """
        with self._filepath.open("r", encoding="utf-8") as input_file:
            records = [json.loads(line) for line in input_file]
        return pd.json_normalize(records)
