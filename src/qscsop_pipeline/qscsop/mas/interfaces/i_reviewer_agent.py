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
        baseline_code: str,
        failed_code: str,
    ) -> str:
        """Confronta il tentativo fallito con il circuito di partenza; ritorna il feedback.

        L'input e' l'intero ValidationResultDTO prodotto dal ValidationService (sezione "Gli
        Agenti e le loro Responsabilita'" della tesi), non solo il suo raw_error_data.

        SERVONO ENTRAMBI I CIRCUITI, e baseline_code e' stato aggiunto dopo una diagnosi: senza
        di esso il Reviewer vedeva solo il risultato del refactoring e ne deduceva il punto di
        partenza, producendo affermazioni false su cui il tentativo successivo poi lavorava. Su
        un circuito di 23 operazioni ha scritto "the original circuit had only two operations" --
        aveva letto il codice fallito, che ne aveva due, e non aveva altro da leggere. Una
        revisione e' un confronto: darle un solo termine la costringe a inventare l'altro.

        L'output resta testo libero (non un DTO, coerente con il class diagram): va iniettato
        direttamente come review_feedback nel tentativo successivo del RefactorerAgent.
        """
        raise NotImplementedError
