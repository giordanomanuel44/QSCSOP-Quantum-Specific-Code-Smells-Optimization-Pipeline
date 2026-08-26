"""Implementazione concreta del servizio di validazione e calcolo metriche di QCEP."""

import ast
from typing import Optional

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qcep.interfaces.i_quantum_metrics_service import IQuantumMetricsService


class QuantumMetricsService(IQuantumMetricsService):
    """Valida un record grezzo e ne misura gli smell baseline tramite IQiskitFacade."""

    def __init__(self, facade: IQiskitFacade) -> None:
        self._facade = facade

    def calculate_metrics(self, raw_record: dict) -> Optional[dict]:
        """Valida sintassi e circuito, ritorna il record arricchito o None se va scartato.

        Il baseline porta ora la sola misura QSMELL. Le metriche di costo (gateCount/depth
        astratte e fisiche, logicalQubits) sono uscite dal contratto: erano cieche ai refactoring
        della pipeline e il loro unico consumatore era un KPI che valeva sempre zero -- vedi
        docs/misura_metriche_fisiche_pre_rimozione.md. Con loro e' sparita la transpilazione, che
        era l'operazione piu' costosa di questo servizio: calculate_smell_metrics non transpila.
        """
        source_code = raw_record["sourceCode"]

        if not self._validate_syntax(source_code):
            return None

        # Il dataset proviene da mining di repository reali (codice non fidato): anche a
        # sintassi valida, ogni stadio successivo puo' fallire in modi imprevedibili (gate
        # inesistenti, API Qiskit deprecate, errori runtime nella sandbox). Un'eccezione qui
        # romperebbe il generator a monte in QCEPMain, quindi la si cattura e si scarta il
        # record restituendo None, senza interrompere l'elaborazione degli elementi successivi.
        try:
            smell_metrics = self._facade.calculate_smell_metrics(source_code)
        except Exception:
            return None

        return {
            "circuitId": raw_record["circuitId"],
            "datasetSource": raw_record["datasetSource"],
            "baseline": {
                "sourceCode": source_code,
                "smellMetrics": _to_baseline_metrics(smell_metrics),
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


def _to_baseline_metrics(smell_metrics: dict) -> dict:
    """Traduce il payload della facade nella forma persistita del contratto dati.

    La facade ritorna di piu' di quanto il record debba portare: gateError ed
    errorFreeProbability sono la forma con cui il paper PRESENTA la metrica, non misure da
    persistere, e worstQubit e' un puntatore dentro al circuito utile al momento della
    riparazione (lo legge il MAS chiamando la facade), non una proprieta' del risultato.
    Le chiavi qui sotto coincidono con SmellMetrics.to_dict(), che e' cio' che QSCSOP ricostruira'.
    """
    long_circuit = smell_metrics["longCircuit"]
    return {
        "maxOpsPerQubit": long_circuit["maxOpsPerQubit"],
        "maxParallelOps": long_circuit["maxParallelOps"],
        "longCircuit": long_circuit["value"],
        "idleQubits": smell_metrics["idleQubits"]["value"],
    }
