"""Interfaccia astratta per la persistenza incrementale dei record elaborati da QCEP."""

from abc import ABC, abstractmethod


class IDatasetWriter(ABC):
    """Astrae la scrittura su disco di un singolo record di metriche."""

    @abstractmethod
    def load(self, dict_with_metrics: dict) -> None:
        """Serializza e appende dict_with_metrics al dataset di output."""
        raise NotImplementedError
