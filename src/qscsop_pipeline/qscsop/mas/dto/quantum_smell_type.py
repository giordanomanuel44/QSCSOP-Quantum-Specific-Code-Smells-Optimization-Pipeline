"""Enum dei Quantum Code Smell in scope per questa tesi."""

from enum import Enum


class QuantumSmellType(str, Enum):
    """Unica fonte di verità sui due smell trattati da questa tesi (sezione 1.6).

    Eredita da str oltre che da Enum affinché i valori siano direttamente
    confrontabili e serializzabili come stringhe semplici (es. QuantumSmellType.LONG_CIRCUIT
    == "long_circuit"), coerente con detected_smells: List<String> del class diagram.
    """

    LONG_CIRCUIT = "long_circuit"
    IDLE_QUBITS = "idle_qubits"
