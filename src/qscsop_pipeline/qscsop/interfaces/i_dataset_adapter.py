"""Interfaccia astratta per l'ingestione del dataset arricchito prodotto da QCEP."""

from abc import ABC, abstractmethod
from collections.abc import Generator

from qscsop_pipeline.qscsop.entities.quantum_program_entity import QuantumProgramEntity


class IDatasetAdapter(ABC):
    """Astrae la lettura del dataset di input e la ricostruzione delle entità di dominio."""

    @abstractmethod
    def stream_programs(self) -> Generator[QuantumProgramEntity, None, None]:
        """Genera un QuantumProgramEntity per ogni record del dataset di input."""
        raise NotImplementedError
