"""Implementazione concreta della pipeline di verifica deterministica (sezione 1.4.3)."""

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qscsop.mas.dto.failure_reason import FailureReason
from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO
from qscsop_pipeline.qscsop.mas.interfaces.i_validation_service import IValidationService

_NOT_EQUIVALENT_ERROR = "Il circuito refactored non e' funzionalmente equivalente al baseline"

_NOT_IMPROVED_ERROR = (
    "Il refactoring non ha ridotto nessuna delle due metriche di smell rispetto alla baseline. "
    "Baseline: l*c={baseline_lc}, IdQ={baseline_idq}. "
    "Refactored: l*c={new_lc}, IdQ={new_idq}. "
    "E' richiesto che almeno una delle due SCENDA e che nessuna delle due SALGA. "
    "l*c si riduce soltanto RIMUOVENDO operazioni; IdQ si riduce eliminando l'attesa di un "
    "qubit fra due sue operazioni, purche' cio' non allunghi la catena del qubit piu' carico "
    "(altrimenti l*c peggiora e il refactoring viene respinto lo stesso)."
)


class ValidationService(IValidationService):
    """Verifica in sequenza compilabilità, equivalenza funzionale e metriche di new_code."""

    def __init__(self, facade: IQiskitFacade) -> None:
        self._facade = facade
        # Cache della misura di baseline, indicizzata sul baseline_code: la baseline non cambia
        # mai tra le iterazioni del ciclo di refactoring del MASEngine, quindi rimisurarla ad ogni
        # validate() sarebbe uno spreco proporzionale a maxIterations. Il risparmio e' oggi minore
        # di prima -- calculate_smell_metrics non transpila, a differenza del calculate_metrics
        # che sostituisce -- ma isolate_circuit ESEGUE comunque il sorgente, quindi non e' gratis.
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

            metrics = self._facade.calculate_smell_metrics(new_code)
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
        """Ritorna la misura della baseline, calcolandola una sola volta per baseline_code."""
        if baseline_code not in self._baseline_metrics_cache:
            self._baseline_metrics_cache[baseline_code] = self._facade.calculate_smell_metrics(
                baseline_code
            )
        return self._baseline_metrics_cache[baseline_code]

    @staticmethod
    def _is_improvement(baseline_metrics: dict, new_metrics: dict) -> bool:
        """Applica il criterio Pareto sul nodo "Migliori?" dell'Activity Diagram.

        Migliorativo se e solo se almeno una fra l*c e IdQ e' STRETTAMENTE minore nel new, e
        nessuna delle due e' MAGGIORE rispetto al baseline.

        LA FORMA DEL CRITERIO NON E' CAMBIATA, sono cambiati i VALORI confrontati: prima era la
        terna (physicalMetrics.gateCount, physicalMetrics.depth, logicalQubits). Quelle tre
        misurano il COSTO del circuito, non gli smell, mentre il nodo "Migliori?" chiede se lo
        smell e' stato risolto. La differenza non era teorica: il fix canonico di Idle Qubits del
        paper (idq-smelly.py -> idq-fixed.py) porta IdQ da 7 a 0 ma alza depth da 8 a 11, quindi
        veniva RESPINTO; e su una coppia costruita equivalente il riempimento dell'attesa lasciava
        la terna identica (9, 6, 2) con IdQ da 2 a 0, quindi veniva respinto lo stesso. Il ciclo
        del MASEngine non poteva chiudersi con successo su Idle Qubits, mai, per costruzione.
        """
        baseline_values = ValidationService._smell_values(baseline_metrics)
        new_values = ValidationService._smell_values(new_metrics)
        strictly_better = any(new < base for new, base in zip(new_values, baseline_values))
        none_worse = all(new <= base for new, base in zip(new_values, baseline_values))
        return strictly_better and none_worse

    @staticmethod
    def _smell_values(metrics: dict) -> tuple[int, int]:
        """Estrae la coppia (l*c, IdQ) dal payload di calculate_smell_metrics."""
        return metrics["longCircuit"]["value"], metrics["idleQubits"]["value"]

    @staticmethod
    def _format_not_improved_error(baseline_metrics: dict, new_metrics: dict) -> str:
        """Compone il messaggio azionabile per il ReviewerAgent coi valori confrontati.

        Riporta le stesse metriche su cui il verdetto e' stato dato: un messaggio che parlasse
        ancora di gateCount manderebbe il tentativo successivo a ottimizzare una grandezza che non
        decide piu' nulla.
        """
        baseline_lc, baseline_idq = ValidationService._smell_values(baseline_metrics)
        new_lc, new_idq = ValidationService._smell_values(new_metrics)
        return _NOT_IMPROVED_ERROR.format(
            baseline_lc=baseline_lc,
            baseline_idq=baseline_idq,
            new_lc=new_lc,
            new_idq=new_idq,
        )
