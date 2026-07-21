"""Interfaccia astratta per l'agente di revisione dei tentativi di refactoring falliti."""

from abc import ABC, abstractmethod

from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO


class IReviewerAgent(ABC):
    """Astrae la traduzione di un errore di validazione grezzo in feedback azionabile."""

    @abstractmethod
    def review(self, raw_error_details: str, original_smell: SmellReportDTO) -> str:
        """Contestualizza raw_error_details rispetto a original_smell; ritorna il feedback.

        L'output e' testo libero (non un DTO, coerente con il class diagram): va iniettato
        direttamente come review_feedback nel tentativo successivo del RefactorerAgent.
        """
        raise NotImplementedError
