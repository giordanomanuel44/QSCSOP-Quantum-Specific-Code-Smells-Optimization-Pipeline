"""Interfaccia astratta per l'orchestratore del ciclo detect-refactor-validate-review."""

from abc import ABC, abstractmethod

from qscsop_pipeline.qscsop.entities.quantum_program_entity import QuantumProgramEntity


class IMASEngine(ABC):
    """Astrae l'orchestrazione del sistema multi-agente (sezione 1.4.3, Activity Diagram Fig. 1.6)."""

    @abstractmethod
    def process_entity(self, entity: QuantumProgramEntity) -> QuantumProgramEntity:
        """Porta entity a uno stato terminale (SMELL_FREE, OPTIMIZED o OPT_FAILED)."""
        raise NotImplementedError
