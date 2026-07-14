"""Implementazione concreta del servizio di validazione e calcolo metriche di QCEP."""

import ast
from typing import Optional

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qcep.interfaces.i_quantum_metrics_service import IQuantumMetricsService


class QuantumMetricsService(IQuantumMetricsService):
    """Valida un record grezzo e ne calcola le metriche baseline tramite IQiskitFacade."""

    def __init__(self, facade: IQiskitFacade) -> None:
        self._facade = facade

    def calculate_metrics(self, raw_record: dict) -> Optional[dict]:
        """Valida sintassi e circuito, ritorna il record arricchito o None se va scartato."""
        source_code = raw_record["sourceCode"]

        if not self._validate_syntax(source_code):
            return None

        # Il dataset proviene da mining di repository reali (codice non fidato): anche a
        # sintassi valida, ogni stadio successivo puo' fallire in modi imprevedibili (gate
        # inesistenti, API Qiskit deprecate, errori runtime nella sandbox). Un'eccezione qui
        # romperebbe il generator a monte in QCEPMain, quindi la si cattura e si scarta il
        # record restituendo None, senza interrompere l'elaborazione degli elementi successivi.
        try:
            qc = self._facade.isolate_circuit(source_code)
            abstract_metrics = self._facade.get_abstract_metrics(qc)
            transpiled_qc = self._facade.transpile_circuit(qc)
            physical_metrics = self._facade.get_physical_metrics(transpiled_qc)
        except Exception:
            return None

        return {
            "circuitId": raw_record["circuitId"],
            "datasetSource": raw_record["datasetSource"],
            "baseline": {
                "sourceCode": source_code,
                "logicalQubits": qc.num_qubits,
                "abstractMetrics": abstract_metrics,
                "physicalMetrics": physical_metrics,
            },
        }

    @staticmethod
    def _validate_syntax(source_code: str) -> bool:
        """Verifica solo la correttezza sintattica tramite ast, senza eseguire il codice.

        SyntaxError copre gia' anche IndentationError (sua sottoclasse in Python).
        """
        try:
            ast.parse(source_code)
        except SyntaxError:
            return False
        return True
