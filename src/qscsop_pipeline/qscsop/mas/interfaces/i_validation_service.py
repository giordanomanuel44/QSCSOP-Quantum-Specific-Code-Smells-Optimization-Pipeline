"""Interfaccia astratta per la pipeline di verifica deterministica di un circuito refactored."""

from abc import ABC, abstractmethod

from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO


class IValidationService(ABC):
    """Astrae la verifica di compilabilità, equivalenza funzionale e calcolo metriche."""

    @abstractmethod
    def validate(self, baseline_code: str, new_code: str) -> ValidationResultDTO:
        """Verifica new_code rispetto a baseline_code e ritorna l'esito della pipeline."""
        raise NotImplementedError
