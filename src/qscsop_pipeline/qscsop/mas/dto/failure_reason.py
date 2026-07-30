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

    COMPILATION_FAILED = "compilation_failed"
    NOT_EQUIVALENT = "not_equivalent"
    METRICS_NOT_IMPROVED = "metrics_not_improved"
    UNEXPECTED_ERROR = "unexpected_error"
