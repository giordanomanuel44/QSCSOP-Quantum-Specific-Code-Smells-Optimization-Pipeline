"""Entità di dominio pura: una versione (baseline o refactored) di un circuito quantistico."""

from qscsop_pipeline.qscsop.entities.smell_metrics import SmellMetrics


class CircuitVersion:
    """Sorgente e misura degli smell di una specifica versione di un circuito.

    UNA sola SmellMetrics, dove prima c'erano due CircuitMetrics (astratte e fisiche) piu'
    logical_qubits. Le metriche fisiche sono state rimosse perche' misurate cieche ai refactoring
    della pipeline (docs/misura_metriche_fisiche_pre_rimozione.md); logical_qubits perche' il suo
    unico consumatore era un KPI di Analytics che sotto QSMELL vale sempre zero -- un fix di Idle
    Qubits non rimuove qubit, li tiene occupati.
    """

    def __init__(self, source_code: str, smell_metrics: SmellMetrics) -> None:
        self._source_code = source_code
        self._smell_metrics = smell_metrics

    def get_source_code(self) -> str:
        return self._source_code

    def set_source_code(self, value: str) -> None:
        self._source_code = value

    def get_smell_metrics(self) -> SmellMetrics:
        return self._smell_metrics

    def set_smell_metrics(self, value: SmellMetrics) -> None:
        self._smell_metrics = value

    def to_dict(self) -> dict:
        """Serializza la versione, annidando la misura come dict tramite il suo to_dict()."""
        return {
            "sourceCode": self._source_code,
            "smellMetrics": self._smell_metrics.to_dict(),
        }
