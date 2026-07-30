"""Entità di dominio pura: esito e stato di avanzamento della valutazione di un circuito."""

from typing import Optional

from qscsop_pipeline.qscsop.entities.evaluation_status import EvaluationStatus
from qscsop_pipeline.qscsop.mas.dto.failure_reason import FailureReason


class EvaluationData:
    """Traccia lo stato del ciclo di rilevamento/refactoring/validazione (Activity Diagram)."""

    def __init__(self) -> None:
        self._is_functionally_equivalent: Optional[bool] = None
        self._iteration_count: int = 0
        self._status: EvaluationStatus = EvaluationStatus.PROCESSING
        self._detected_smells: list[str] = []
        self._failure_reason: Optional[FailureReason] = None

    def get_is_functionally_equivalent(self) -> Optional[bool]:
        return self._is_functionally_equivalent

    def set_is_functionally_equivalent(self, value: Optional[bool]) -> None:
        self._is_functionally_equivalent = value

    def get_iteration_count(self) -> int:
        return self._iteration_count

    def set_iteration_count(self, value: int) -> None:
        self._iteration_count = value

    def get_status(self) -> EvaluationStatus:
        return self._status

    def set_status(self, value: EvaluationStatus) -> None:
        self._status = value

    def get_detected_smells(self) -> list[str]:
        """Ritorna una copia della lista interna, per non esporre il riferimento mutabile."""
        return list(self._detected_smells)

    def set_detected_smells(self, value: list[str]) -> None:
        """Memorizza una copia di value, per non condividere il riferimento con il chiamante."""
        self._detected_smells = list(value)

    def get_failure_reason(self) -> Optional[FailureReason]:
        return self._failure_reason

    def set_failure_reason(self, value: Optional[FailureReason]) -> None:
        self._failure_reason = value

    def increment_iteration_count(self) -> None:
        """Incrementa di uno il contatore di iterazioni del ciclo detect-refactor-validate."""
        self.set_iteration_count(self.get_iteration_count() + 1)

    def update_result(self, is_functionally_equivalent: bool, status: EvaluationStatus) -> None:
        """Aggiorna atomicamente esito di equivalenza funzionale e stato del circuito."""
        self.set_is_functionally_equivalent(is_functionally_equivalent)
        self.set_status(status)

    def to_dict(self) -> dict:
        """Serializza la valutazione.

        Nota: la chiave "detected_smells" resta con underscore (non camelCase come le altre)
        per fedeltà al class diagram di tesi, che presenta la stessa incoerenza. "failureReason"
        non è previsto da alcun diagramma, quindi segue la convenzione camelCase delle altre
        chiavi ed è inclusa SOLO SE valorizzata (mai serializzata come null), stesso pattern di
        "refactored" in QuantumProgramEntity.to_dict().
        """
        result = {
            "isFunctionallyEquivalent": self._is_functionally_equivalent,
            "iterationCount": self._iteration_count,
            "status": self._status.value,
            "detected_smells": self.get_detected_smells(),
        }
        if self._failure_reason is not None:
            result["failureReason"] = self._failure_reason.value
        return result
