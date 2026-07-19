"""Interfaccia astratta per l'agente di rilevamento dei Quantum Code Smell."""

from abc import ABC, abstractmethod

from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO


class IDetectorAgent(ABC):
    """Astrae l'analisi di un circuito alla ricerca di Long Circuit e Idle Qubits."""

    @abstractmethod
    def detect_smell(self, code: str) -> SmellReportDTO:
        """Analizza code e ritorna un report sulla presenza di smell."""
        raise NotImplementedError
