"""Entità di dominio pura: aggregato radice di un programma quantistico nel ciclo QSCSOP."""

from typing import Optional

from qscsop_pipeline.qscsop.entities.circuit_version import CircuitVersion
from qscsop_pipeline.qscsop.entities.evaluation_data import EvaluationData


class QuantumProgramEntity:
    """Circuito sotto analisi: identità, baseline, versione refactored ed esito valutazione."""

    def __init__(self, circuit_id: str, dataset_source: str, baseline: CircuitVersion) -> None:
        self._circuit_id = circuit_id
        self._dataset_source = dataset_source
        self._baseline = baseline
        self._refactored: Optional[CircuitVersion] = None
        self._evaluation: EvaluationData = EvaluationData()

    def get_circuit_id(self) -> str:
        return self._circuit_id

    def set_circuit_id(self, value: str) -> None:
        self._circuit_id = value

    def get_dataset_source(self) -> str:
        return self._dataset_source

    def set_dataset_source(self, value: str) -> None:
        self._dataset_source = value

    def get_baseline(self) -> CircuitVersion:
        return self._baseline

    def set_baseline(self, value: CircuitVersion) -> None:
        self._baseline = value

    def get_refactored(self) -> Optional[CircuitVersion]:
        return self._refactored

    def set_refactored(self, value: Optional[CircuitVersion]) -> None:
        self._refactored = value

    def get_evaluation(self) -> EvaluationData:
        return self._evaluation

    def set_evaluation(self, value: EvaluationData) -> None:
        self._evaluation = value

    def set_refactored_version(self, version: CircuitVersion) -> None:
        """Registra la versione refactored ricevuta indietro dal MASEngine."""
        self.set_refactored(version)

    def to_dict(self) -> dict:
        """Serializza l'entità; la chiave "refactored" è presente solo se una versione esiste.

        I circuiti ancora SMELL_FREE (mai passati dal MASEngine) non possiedono un blocco
        refactored nel record, coerentemente con la sezione 1.5.1 della tesi.
        """
        result = {
            "circuitId": self._circuit_id,
            "datasetSource": self._dataset_source,
            "baseline": self._baseline.to_dict(),
            "evaluation": self._evaluation.to_dict(),
        }
        if self._refactored is not None:
            result["refactored"] = self._refactored.to_dict()
        return result
