"""Interfaccia astratta per l'ingestione del dataset prodotto da QSCSOP."""

from abc import ABC, abstractmethod

import pandas as pd


class IDataLoader(ABC):
    """Astrae la lettura del file di risultati e la sua trasformazione in DataFrame."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Carica l'intero dataset di risultati in un DataFrame appiattito."""
        raise NotImplementedError
