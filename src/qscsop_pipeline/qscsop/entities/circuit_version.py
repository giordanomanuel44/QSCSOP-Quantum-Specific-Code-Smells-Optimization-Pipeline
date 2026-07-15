"""Entità di dominio pura: una versione (baseline o refactored) di un circuito quantistico."""

from qscsop_pipeline.qscsop.entities.circuit_metrics import CircuitMetrics


class CircuitVersion:
    """Sorgente e metriche (astratte e fisiche) di una specifica versione di un circuito."""

    def __init__(
        self,
        source_code: str,
        logical_qubits: int,
        abstract_metrics: CircuitMetrics,
        physical_metrics: CircuitMetrics,
    ) -> None:
        self._source_code = source_code
        self._logical_qubits = logical_qubits
        self._abstract_metrics = abstract_metrics
        self._physical_metrics = physical_metrics

    def get_source_code(self) -> str:
        return self._source_code

    def set_source_code(self, value: str) -> None:
        self._source_code = value

    def get_logical_qubits(self) -> int:
        return self._logical_qubits

    def set_logical_qubits(self, value: int) -> None:
        self._logical_qubits = value

    def get_abstract_metrics(self) -> CircuitMetrics:
        return self._abstract_metrics

    def set_abstract_metrics(self, value: CircuitMetrics) -> None:
        self._abstract_metrics = value

    def get_physical_metrics(self) -> CircuitMetrics:
        return self._physical_metrics

    def set_physical_metrics(self, value: CircuitMetrics) -> None:
        self._physical_metrics = value

    def to_dict(self) -> dict:
        """Serializza la versione, annidando le metriche come dict tramite il loro to_dict()."""
        return {
            "sourceCode": self._source_code,
            "logicalQubits": self._logical_qubits,
            "abstractMetrics": self._abstract_metrics.to_dict(),
            "physicalMetrics": self._physical_metrics.to_dict(),
        }
