"""Interfaccia astratta per l'agente di revisione dei tentativi di refactoring falliti."""

from abc import ABC, abstractmethod

from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO
from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO


class IReviewerAgent(ABC):
    """Astrae la traduzione di un esito di validazione fallito in feedback azionabile."""

    @abstractmethod
    def review(
        self,
        validation_result: ValidationResultDTO,
        original_smell: SmellReportDTO,
        failed_code: str,
    ) -> str:
        """Contestualizza validation_result rispetto a original_smell; ritorna il feedback.

        L'input e' l'intero ValidationResultDTO prodotto dal ValidationService (sezione "Gli
        Agenti e le loro Responsabilita'" della tesi), non solo il suo raw_error_data. failed_code
        e' il codice esatto prodotto dal tentativo di refactoring appena fallito (non il
        baseline): permette al feedback di riferirsi concretamente a cosa il tentativo ha fatto,
        invece di descrivere l'errore solo in astratto. L'output resta testo libero (non un DTO,
        coerente con il class diagram): va iniettato direttamente come review_feedback nel
        tentativo successivo del RefactorerAgent.
        """
        raise NotImplementedError
