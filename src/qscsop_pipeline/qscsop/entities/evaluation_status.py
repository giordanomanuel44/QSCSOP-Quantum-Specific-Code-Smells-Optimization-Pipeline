"""Enum degli stati del ciclo di vita di EvaluationData."""

from enum import Enum


class EvaluationStatus(str, Enum):
    """Unica fonte di verità sui quattro stati del ciclo di vita di EvaluationData
    (sezione 1.4.3 e Activity Diagram Fig. 1.6 della tesi).

    Eredita da str oltre che da Enum affinché i valori siano direttamente
    confrontabili e serializzabili come stringhe semplici (es. EvaluationStatus.PROCESSING
    == "PROCESSING"), coerente con status: String del class diagram.
    """

    PROCESSING = "PROCESSING"
    SMELL_FREE = "SMELL_FREE"
    OPTIMIZED = "OPTIMIZED"
    OPT_FAILED = "OPT_FAILED"
