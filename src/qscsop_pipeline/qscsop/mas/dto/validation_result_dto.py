"""Value object immutabile: esito della pipeline di verifica deterministica di QSCSOP."""

from typing import Optional


class ValidationResultDTO:
    """Esito di una validazione: validità, eventuale errore, eventuali nuove metriche."""

    def __init__(
        self,
        is_valid: bool,
        raw_error_data: Optional[str],
        new_metrics: Optional[dict],
    ) -> None:
        self._is_valid = is_valid
        self._raw_error_data = raw_error_data
        self._new_metrics = new_metrics

    def get_is_valid(self) -> bool:
        return self._is_valid

    def get_raw_error_data(self) -> Optional[str]:
        return self._raw_error_data

    def get_new_metrics(self) -> Optional[dict]:
        return self._new_metrics
