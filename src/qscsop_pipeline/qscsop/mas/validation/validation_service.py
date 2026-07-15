"""Implementazione concreta della pipeline di verifica deterministica (sezione 1.4.3)."""

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO
from qscsop_pipeline.qscsop.mas.interfaces.i_validation_service import IValidationService

_NOT_EQUIVALENT_ERROR = "Il circuito refactored non e' funzionalmente equivalente al baseline"


class ValidationService(IValidationService):
    """Verifica in sequenza compilabilità, equivalenza funzionale e metriche di new_code."""

    def __init__(self, facade: IQiskitFacade) -> None:
        self._facade = facade

    def validate(self, baseline_code: str, new_code: str) -> ValidationResultDTO:
        """Compila, verifica l'equivalenza col baseline e infine calcola le metriche.

        L'ordine (compilazione -> equivalenza -> metriche) segue l'Activity Diagram di tesi:
        un circuito che compila ma non e' equivalente non deve arrivare al calcolo metriche.
        """
        is_valid, error_message = self._facade.compile_circuit(new_code)
        if not is_valid:
            return ValidationResultDTO(
                is_valid=False, raw_error_data=error_message, new_metrics=None
            )

        equivalent = self._facade.check_equivalence(baseline_code, new_code)
        if not equivalent:
            return ValidationResultDTO(
                is_valid=False, raw_error_data=_NOT_EQUIVALENT_ERROR, new_metrics=None
            )

        metrics = self._facade.calculate_metrics(new_code)
        return ValidationResultDTO(is_valid=True, raw_error_data=None, new_metrics=metrics)
