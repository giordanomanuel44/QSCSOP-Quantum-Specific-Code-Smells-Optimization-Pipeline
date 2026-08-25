"""Interfaccia astratta per l'unico punto di contatto ammesso con il framework Qiskit."""

from abc import ABC, abstractmethod
from typing import Optional

from qiskit import QuantumCircuit


class IQiskitFacade(ABC):
    """Astrae isolamento, transpilazione e calcolo metriche dei circuiti Qiskit."""

    @abstractmethod
    def isolate_circuit(self, source_code: str) -> QuantumCircuit:
        """Esegue source_code in sandbox e ritorna l'ultimo QuantumCircuit assegnato."""
        raise NotImplementedError

    @abstractmethod
    def get_abstract_metrics(self, qc: QuantumCircuit) -> dict:
        """Ritorna gateCount e depth del circuito prima della transpilazione."""
        raise NotImplementedError

    @abstractmethod
    def transpile_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Transpila il circuito su un Fake Backend per simulare l'hardware reale."""
        raise NotImplementedError

    @abstractmethod
    def get_physical_metrics(self, qc: QuantumCircuit) -> dict:
        """Ritorna gateCount e depth del circuito dopo la transpilazione."""
        raise NotImplementedError

    @abstractmethod
    def compile_circuit(self, source_code: str) -> tuple[bool, Optional[str]]:
        """Tenta di isolare source_code; ritorna (True, None) o (False, messaggio d'errore)."""
        raise NotImplementedError

    @abstractmethod
    def check_equivalence(self, baseline_code: str, new_code: str) -> bool:
        """Isola entrambi i codici e confronta i rispettivi Statevector a meno di fase globale."""
        raise NotImplementedError

    @abstractmethod
    def is_qubit_idle(self, source_code: str, qubit_index: int) -> bool:
        """Isola il codice e verifica se il qubit indicato resta inerte (fisso e scorrelato)."""
        raise NotImplementedError

    @abstractmethod
    def calculate_smell_metrics(self, code: str) -> dict:
        """Isola code e ritorna le metriche Long Circuit e Idle Qubits secondo QSMELL.

        Forma: {"longCircuit": {"maxOpsPerQubit": int, "maxParallelOps": int, "value": int,
                                "gateError": float, "errorFreeProbability": float},
                "idleQubits": {"value": int, "worstQubit": Optional[int]}}.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_metrics(self, code: str) -> dict:
        """Isola, transpila e ritorna logicalQubits + abstractMetrics + physicalMetrics.

        Forma: {"logicalQubits": int, "abstractMetrics": {...}, "physicalMetrics": {...}}.
        """
        raise NotImplementedError
