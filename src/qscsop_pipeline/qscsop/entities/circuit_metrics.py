"""Entità di dominio pura: metriche di un circuito quantistico (astratte o fisiche)."""


class CircuitMetrics:
    """Coppia di metriche (numero di gate, profondità) associata a una versione di circuito."""

    def __init__(self, gate_count: int, depth: int) -> None:
        self._gate_count = gate_count
        self._depth = depth

    def get_gate_count(self) -> int:
        return self._gate_count

    def set_gate_count(self, value: int) -> None:
        self._gate_count = value

    def get_depth(self) -> int:
        return self._depth

    def set_depth(self, value: int) -> None:
        self._depth = value

    def to_dict(self) -> dict:
        """Serializza le metriche nella forma {"gateCount": int, "depth": int}."""
        return {
            "gateCount": self._gate_count,
            "depth": self._depth,
        }
