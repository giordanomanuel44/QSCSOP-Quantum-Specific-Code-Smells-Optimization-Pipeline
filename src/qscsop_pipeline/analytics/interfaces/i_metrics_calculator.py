"""Interfaccia astratta per il calcolo delle metriche di valutazione empirica (sezione 1.6)."""

from abc import ABC, abstractmethod

import pandas as pd


class IMetricsCalculator(ABC):
    """Astrae il calcolo delle metriche aggregate a partire dal DataFrame dei risultati."""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> dict:
        """Calcola le metriche aggregate di valutazione empirica sul DataFrame ricevuto."""
        raise NotImplementedError
