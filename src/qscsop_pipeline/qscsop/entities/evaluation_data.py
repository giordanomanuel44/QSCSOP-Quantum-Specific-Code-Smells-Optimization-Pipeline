"""Entità di dominio pura: esito e stato di avanzamento della valutazione di un circuito."""

from typing import Optional


class EvaluationData:
    """Traccia lo stato del ciclo di rilevamento/refactoring/validazione (Activity Diagram)."""

    def __init__(self) -> None:
        self._is_functionally_equivalent: Optional[bool] = None
        self._iteration_count: int = 0
        self._status: str = "PROCESSING"
        self._detected_smells: list[str] = []

    def get_is_functionally_equivalent(self) -> Optional[bool]:
        return self._is_functionally_equivalent

    def set_is_functionally_equivalent(self, value: Optional[bool]) -> None:
        self._is_functionally_equivalent = value

    def get_iteration_count(self) -> int:
        return self._iteration_count

    def set_iteration_count(self, value: int) -> None:
        self._iteration_count = value

    def get_status(self) -> str:
        return self._status

    def set_status(self, value: str) -> None:
        self._status = value

    def get_detected_smells(self) -> list[str]:
        """Ritorna una copia della lista interna, per non esporre il riferimento mutabile."""
        return list(self._detected_smells)

    def set_detected_smells(self, value: list[str]) -> None:
        """Memorizza una copia di value, per non condividere il riferimento con il chiamante."""
        self._detected_smells = list(value)

    def increment_iteration_count(self) -> None:
        """Incrementa di uno il contatore di iterazioni del ciclo detect-refactor-validate."""
        self.set_iteration_count(self.get_iteration_count() + 1)

    def update_result(self, is_functionally_equivalent: bool, status: str) -> None:
        """Aggiorna atomicamente esito di equivalenza funzionale e stato del circuito."""
        self.set_is_functionally_equivalent(is_functionally_equivalent)
        self.set_status(status)

    def to_dict(self) -> dict:
        """Serializza la valutazione.

        Nota: la chiave "detected_smells" resta con underscore (non camelCase come le altre)
        per fedeltà al class diagram di tesi, che presenta la stessa incoerenza.
        """
        return {
            "isFunctionallyEquivalent": self._is_functionally_equivalent,
            "iterationCount": self._iteration_count,
            "status": self._status,
            "detected_smells": self.get_detected_smells(),
        }
