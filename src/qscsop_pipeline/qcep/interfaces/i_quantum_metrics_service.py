"""Interfaccia astratta per la validazione e il calcolo delle metriche di un circuito."""

from abc import ABC, abstractmethod
from typing import Optional


class IQuantumMetricsService(ABC):
    """Astrae la trasformazione di un record grezzo in un profilo di metriche validato."""

    @abstractmethod
    def calculate_metrics(self, raw_record: dict) -> Optional[dict]:
        """Valida e arricchisce raw_record con le metriche baseline, o None se va scartato."""
        raise NotImplementedError
