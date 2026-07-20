"""Interfaccia astratta per l'agente di refactoring dei Quantum Code Smell."""

from abc import ABC, abstractmethod

from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO


class IRefactorerAgent(ABC):
    """Astrae la riscrittura di un circuito per eliminare Long Circuit e Idle Qubits."""

    @abstractmethod
    def refactor(self, code: str, smell_report: SmellReportDTO, review_feedback: str) -> str:
        """Riscrive code eliminando gli smell descritti da smell_report; ritorna il codice corretto.

        review_feedback e' vuoto alla prima iterazione; se valorizzato, riporta il motivo per cui
        il tentativo precedente non ha superato la validazione, da correggere nel nuovo tentativo.
        """
        raise NotImplementedError
