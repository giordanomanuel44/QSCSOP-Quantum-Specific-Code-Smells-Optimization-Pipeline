"""Interfaccia astratta per l'estrazione di circuiti grezzi da un dataset sorgente."""

from abc import ABC, abstractmethod
from collections.abc import Generator


class IDatasetParser(ABC):
    """Astrae la navigazione di un dataset e la produzione di record grezzi."""

    @abstractmethod
    def extract_circuits(self) -> Generator[dict, None, None]:
        """Genera un record grezzo (circuitId, datasetSource, sourceCode) per ogni circuito."""
        raise NotImplementedError
