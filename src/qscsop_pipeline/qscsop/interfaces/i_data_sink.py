"""Interfaccia astratta per la persistenza incrementale dei risultati di QSCSOP."""

from abc import ABC, abstractmethod

from qscsop_pipeline.qscsop.entities.quantum_program_entity import QuantumProgramEntity


class IDataSink(ABC):
    """Astrae la scrittura su disco di una singola entità elaborata."""

    @abstractmethod
    def save_program(self, entity: QuantumProgramEntity) -> None:
        """Serializza e persiste entity nel dataset di output."""
        raise NotImplementedError
