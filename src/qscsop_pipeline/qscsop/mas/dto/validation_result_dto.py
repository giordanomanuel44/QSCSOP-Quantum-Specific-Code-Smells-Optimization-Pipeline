"""Esito della pipeline di verifica deterministica di QSCSOP."""

import copy
from typing import Optional

from qscsop_pipeline.qscsop.mas.dto.failure_reason import FailureReason


class ValidationResultDTO:
    """Esito di una validazione: validità, eventuale errore, eventuali nuove metriche."""

    def __init__(
        self,
        is_valid: bool,
        raw_error_data: Optional[str],
        new_metrics: Optional[dict],
        failure_reason: Optional[FailureReason] = None,
    ) -> None:
        self._is_valid = is_valid
        self._raw_error_data = raw_error_data
        self._new_metrics = new_metrics
        self._failure_reason = failure_reason

    def get_is_valid(self) -> bool:
        return self._is_valid

    def set_is_valid(self, value: bool) -> None:
        self._is_valid = value

    def get_raw_error_data(self) -> Optional[str]:
        return self._raw_error_data

    def set_raw_error_data(self, value: Optional[str]) -> None:
        self._raw_error_data = value

    def get_new_metrics(self) -> Optional[dict]:
        """Ritorna una deep copy del dict interno, per non esporre i sotto-dizionari annidati."""
        return copy.deepcopy(self._new_metrics)

    def set_new_metrics(self, value: Optional[dict]) -> None:
        """Memorizza una deep copy di value, essendo new_metrics una struttura annidata."""
        self._new_metrics = copy.deepcopy(value)

    def get_failure_reason(self) -> Optional[FailureReason]:
        return self._failure_reason

    def set_failure_reason(self, value: Optional[FailureReason]) -> None:
        self._failure_reason = value
