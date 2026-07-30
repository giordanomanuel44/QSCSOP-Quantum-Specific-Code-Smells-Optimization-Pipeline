"""Implementazione concreta della pipeline di verifica deterministica (sezione 1.4.3)."""

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qscsop.mas.dto.failure_reason import FailureReason
from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO
from qscsop_pipeline.qscsop.mas.interfaces.i_validation_service import IValidationService

_NOT_EQUIVALENT_ERROR = "Il circuito refactored non e' funzionalmente equivalente al baseline"

_NOT_IMPROVED_ERROR = (
    "Il refactoring non ha migliorato le metriche fisiche rispetto alla baseline. "
    "Baseline: gateCount={baseline_gate_count}, depth={baseline_depth}, "
    "qubit={baseline_qubits}. Refactored: gateCount={new_gate_count}, "
    "depth={new_depth}, qubit={new_qubits}. E' richiesta una riduzione di almeno una "
    "di queste metriche, senza peggiorarne nessuna: un refactoring che lascia il "
    "circuito sostanzialmente invariato non risolve lo smell."
)


class ValidationService(IValidationService):
    """Verifica in sequenza compilabilità, equivalenza funzionale e metriche di new_code."""

    def __init__(self, facade: IQiskitFacade) -> None:
        self._facade = facade
        # Cache delle metriche di baseline, indicizzata sul baseline_code. calculate_metrics
        # esegue una transpilazione, l'operazione piu' costosa della facade; la baseline non
        # cambia mai tra le iterazioni del ciclo di refactoring del MASEngine, quindi ricalcolarla
        # ad ogni validate() sarebbe uno spreco proporzionale a maxIterations.
        self._baseline_metrics_cache: dict[str, dict] = {}

    def validate(self, baseline_code: str, new_code: str) -> ValidationResultDTO:
        """Compila, verifica l'equivalenza col baseline e infine confronta le metriche.

        L'ordine (compilazione -> equivalenza -> metriche/miglioramento) segue l'Activity
        Diagram di tesi: un circuito che compila ma non e' equivalente non deve arrivare al
        calcolo metriche.
        """
        is_valid, error_message = self._facade.compile_circuit(new_code)
        if not is_valid:
            return ValidationResultDTO(
                is_valid=False,
                raw_error_data=error_message,
                new_metrics=None,
                failure_reason=FailureReason.COMPILATION_FAILED,
            )

        # Da qui in poi ogni passo puo' SOLLEVARE eccezioni deliberate: check_equivalence lancia
        # NotImplementedError sul feedback classico e ValueError oltre il limite partial_trace;
        # sia check_equivalence sia _get_baseline_metrics isolano codice non ancora compilato
        # (baseline_code) e transpilano, operazioni a loro volta fallibili. A differenza di QCEP,
        # QSCSOP non ha una rete a monte: senza questo try/except il record andrebbe perso,
        # violando la garanzia della sezione 1.4.4 (output terminale sempre prodotto). Qualunque
        # fallimento imprevisto deve diventare un DTO terminale, mai propagarsi: il MASEngine deve
        # sempre ricevere un esito interpretabile.
        try:
            equivalent = self._facade.check_equivalence(baseline_code, new_code)
            if not equivalent:
                return ValidationResultDTO(
                    is_valid=False,
                    raw_error_data=_NOT_EQUIVALENT_ERROR,
                    new_metrics=None,
                    failure_reason=FailureReason.NOT_EQUIVALENT,
                )

            metrics = self._facade.calculate_metrics(new_code)
            baseline_metrics = self._get_baseline_metrics(baseline_code)

            if not self._is_improvement(baseline_metrics, metrics):
                # A differenza degli altri due fallimenti (compilazione, equivalenza), qui
                # metrics e' gia' stato calcolato con successo su un circuito compilabile ed
                # equivalente al baseline: scartarlo sarebbe buttare un dato valido. new_metrics
                # popolato E' il segnale strutturale che identifica questo specifico tipo di
                # fallimento ("Migliori?" non superato) rispetto agli altri due, dove resta None
                # per costruzione (non esiste alcun new_code compilato/equivalente da misurare).
                # Nessun consumatore deve pero' interpretare new_metrics!=None come "successo":
                # solo is_valid lo garantisce.
                return ValidationResultDTO(
                    is_valid=False,
                    raw_error_data=self._format_not_improved_error(baseline_metrics, metrics),
                    new_metrics=metrics,
                    failure_reason=FailureReason.METRICS_NOT_IMPROVED,
                )

            return ValidationResultDTO(is_valid=True, raw_error_data=None, new_metrics=metrics)
        except Exception as e:
            return ValidationResultDTO(
                is_valid=False,
                raw_error_data=f"Errore durante la validazione: {type(e).__name__}: {e}",
                new_metrics=None,
                failure_reason=FailureReason.UNEXPECTED_ERROR,
            )

    def _get_baseline_metrics(self, baseline_code: str) -> dict:
        """Ritorna le metriche della baseline, calcolandole una sola volta per baseline_code."""
        if baseline_code not in self._baseline_metrics_cache:
            self._baseline_metrics_cache[baseline_code] = self._facade.calculate_metrics(
                baseline_code
            )
        return self._baseline_metrics_cache[baseline_code]

    @staticmethod
    def _is_improvement(baseline_metrics: dict, new_metrics: dict) -> bool:
        """Applica il criterio Pareto sul nodo "Migliori?" dell'Activity Diagram.

        Migliorativo se e solo se almeno una tra physicalMetrics.gateCount,
        physicalMetrics.depth e logicalQubits e' STRETTAMENTE minore nel new, e nessuna delle
        tre e' MAGGIORE nel new rispetto al baseline.
        """
        baseline_values = ValidationService._pareto_values(baseline_metrics)
        new_values = ValidationService._pareto_values(new_metrics)
        strictly_better = any(new < base for new, base in zip(new_values, baseline_values))
        none_worse = all(new <= base for new, base in zip(new_values, baseline_values))
        return strictly_better and none_worse

    @staticmethod
    def _pareto_values(metrics: dict) -> tuple[int, int, int]:
        """Estrae la terna (physical gateCount, physical depth, logicalQubits) confrontata."""
        physical = metrics["physicalMetrics"]
        return physical["gateCount"], physical["depth"], metrics["logicalQubits"]

    @staticmethod
    def _format_not_improved_error(baseline_metrics: dict, new_metrics: dict) -> str:
        """Compone il messaggio azionabile per il ReviewerAgent coi valori confrontati."""
        baseline_gate_count, baseline_depth, baseline_qubits = ValidationService._pareto_values(
            baseline_metrics
        )
        new_gate_count, new_depth, new_qubits = ValidationService._pareto_values(new_metrics)
        return _NOT_IMPROVED_ERROR.format(
            baseline_gate_count=baseline_gate_count,
            baseline_depth=baseline_depth,
            baseline_qubits=baseline_qubits,
            new_gate_count=new_gate_count,
            new_depth=new_depth,
            new_qubits=new_qubits,
        )
