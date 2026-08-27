"""Enum dei motivi per cui una validazione (o l'elaborazione di un'entita') puo' fallire."""

from enum import Enum


class FailureReason(str, Enum):
    """Unica fonte di verita' sui motivi di fallimento che portano un'entita' a OPT_FAILED.

    Eredita da str oltre che da Enum affinche' i valori siano direttamente confrontabili
    e serializzabili come stringhe semplici, coerente con QuantumSmellType/EvaluationStatus.

    Popolato SOLO quando la validazione o l'elaborazione fallisce: resta None per
    SMELL_FREE e OPTIMIZED. UNEXPECTED_ERROR copre sia le eccezioni impreviste catturate
    dentro ValidationService (durante check_equivalence/calculate_metrics) sia quelle
    catturate dal blocco try/except esterno di MASEngine, che puo' scattare anche PRIMA
    che una validazione sia mai stata tentata (es. durante detect_smell o refactor).
    """

    # Il ciclo di refactoring non e' MAI stato tentato: il DetectorAgent ha dichiarato che il
    # circuito non contiene nulla di rimovibile (sopra soglia per sola dimensione, ogni operazione
    # contribuisce). Va distinto dagli altri tre, che descrivono un tentativo fallito: qui non
    # c'e' stato tentativo, quindi iterationCount resta 0. Attenzione in fase di analisi: e' un
    # giudizio del modello, mentre la stratificazione per riduzione richiesta e' calcolata.
    NOT_REPAIRABLE = "not_repairable"
    COMPILATION_FAILED = "compilation_failed"
    NOT_EQUIVALENT = "not_equivalent"
    METRICS_NOT_IMPROVED = "metrics_not_improved"
    UNEXPECTED_ERROR = "unexpected_error"
