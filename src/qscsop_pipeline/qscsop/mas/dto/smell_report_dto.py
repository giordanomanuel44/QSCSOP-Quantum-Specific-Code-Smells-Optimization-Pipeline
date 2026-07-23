"""DTO prodotto dal DetectorAgent: esito dell'analisi di rilevamento smell."""

from typing import Optional


class SmellReportDTO:
    """Esito dell'analisi del DetectorAgent: presenza di smell e relativa descrizione."""

    def __init__(
        self,
        has_smells: bool,
        report_details: str,
        detected_smells: Optional[list[str]] = None,
    ) -> None:
        self._has_smells = has_smells
        self._report_details = report_details
        # Default None invece di [] mutabile: evita di condividere la stessa lista tra istanze.
        self._detected_smells: list[str] = (
            list(detected_smells) if detected_smells is not None else []
        )

    def get_has_smells(self) -> bool:
        return self._has_smells

    def set_has_smells(self, value: bool) -> None:
        self._has_smells = value

    def get_report_details(self) -> str:
        return self._report_details

    def set_report_details(self, value: str) -> None:
        self._report_details = value

    def get_detected_smells(self) -> list[str]:
        """Ritorna una copia della lista interna, per non esporre il riferimento mutabile."""
        return list(self._detected_smells)

    def set_detected_smells(self, value: list[str]) -> None:
        """Memorizza una copia di value, per non condividere il riferimento con il chiamante."""
        self._detected_smells = list(value)
