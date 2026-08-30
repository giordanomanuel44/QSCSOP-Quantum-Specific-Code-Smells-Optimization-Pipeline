"""DTO prodotto dal DetectorAgent: esito dell'analisi di rilevamento smell."""

from typing import Optional


class SmellReportDTO:
    """Esito dell'analisi del DetectorAgent: presenza di smell e relativa descrizione."""

    def __init__(
        self,
        has_smells: bool,
        report_details: str,
        detected_smells: Optional[list[str]] = None,
        repairable: bool = True,
    ) -> None:
        self._has_smells = has_smells
        self._report_details = report_details
        # Default True: un circuito e' riparabile finche' il DetectorAgent non dichiara il
        # contrario, cosi' i chiamanti che non conoscono il campo si comportano come prima.
        self._repairable = repairable
        # Default None invece di [] mutabile: evita di condividere la stessa lista tra istanze.
        self._detected_smells: list[str] = (
            list(detected_smells) if detected_smells is not None else []
        )

    def get_repairable(self) -> bool:
        """False quando il Detector dichiara che non c'e' nulla di rimovibile.

        NON e' l'assenza di smell (per quello c'e' has_smells): e' un circuito sopra soglia per
        sola dimensione, in cui ogni operazione contribuisce e nulla si cancella.

        E' REGISTRATO, NON AGITO: il MASEngine lo scrive nella tracciatura ma entra nel ciclo
        comunque. Quando invece lo usava per uscire subito, dei 33 circuiti dichiarati
        irriparabili 15 erano migliorabili e 5 portabili sotto soglia. E' un GIUDIZIO DEL
        MODELLO, non una prova: va riportato come tale, e misurarne l'accuratezza e' un dato.
        """
        return self._repairable

    def set_repairable(self, value: bool) -> None:
        self._repairable = value

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
