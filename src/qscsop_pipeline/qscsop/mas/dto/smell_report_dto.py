"""DTO prodotto dal DetectorAgent: esito dell'analisi di rilevamento smell."""


class SmellReportDTO:
    """Esito dell'analisi del DetectorAgent: presenza di smell e relativa descrizione."""

    def __init__(self, has_smells: bool, report_details: str) -> None:
        self._has_smells = has_smells
        self._report_details = report_details

    def get_has_smells(self) -> bool:
        return self._has_smells

    def set_has_smells(self, value: bool) -> None:
        self._has_smells = value

    def get_report_details(self) -> str:
        return self._report_details

    def set_report_details(self, value: str) -> None:
        self._report_details = value
