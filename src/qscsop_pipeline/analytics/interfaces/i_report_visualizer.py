"""Interfaccia astratta per la generazione degli artefatti visivi del report finale."""

from abc import ABC, abstractmethod

import pandas as pd


class IReportVisualizer(ABC):
    """Astrae la generazione e l'esportazione su disco del report grafico."""

    @abstractmethod
    def visualize(self, df: pd.DataFrame, metrics: dict) -> None:
        """Genera ed esporta su disco gli artefatti visivi a partire da df e metrics."""
        raise NotImplementedError
