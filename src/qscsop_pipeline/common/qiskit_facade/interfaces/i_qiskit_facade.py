"""Interfaccia astratta per l'unico punto di contatto ammesso con il framework Qiskit."""

from abc import ABC, abstractmethod

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
