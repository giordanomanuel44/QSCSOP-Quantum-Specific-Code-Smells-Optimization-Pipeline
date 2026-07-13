"""Implementazione concreta dell'unico punto di contatto ammesso con il framework Qiskit."""

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade


class QiskitFacade(IQiskitFacade):
    """Incapsula isolamento, transpilazione e calcolo metriche tramite Qiskit."""

    # GenericBackendV2 sostituisce i Fake Backend concreti a 5 qubit (FakeManila, FakeAthens),
    # rimossi da qiskit.providers.fake_provider a partire da Qiskit 2.x: genera un backend
    # parametrico con basis gates e coupling map realistici, equivalente ai fini della transpilazione.
    _DEFAULT_NUM_QUBITS = 5
    _BACKEND_SEED = 42

    def isolate_circuit(self, source_code: str) -> QuantumCircuit:
        """Esegue source_code in sandbox e ritorna l'ultimo QuantumCircuit assegnato."""
        namespace: dict = {}
        exec(source_code, namespace)

        circuit = None
        for value in namespace.values():
            if isinstance(value, QuantumCircuit):
                circuit = value

        if circuit is None:
            raise ValueError(
                "Nessun QuantumCircuit trovato tra le variabili assegnate dal codice sorgente."
            )
        return circuit

    def get_abstract_metrics(self, qc: QuantumCircuit) -> dict:
        """Ritorna gateCount e depth del circuito prima della transpilazione."""
        return self._extract_metrics(qc)

    def transpile_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Transpila il circuito su un Fake Backend per simulare l'hardware reale."""
        num_qubits = max(qc.num_qubits, self._DEFAULT_NUM_QUBITS)
        backend = GenericBackendV2(num_qubits=num_qubits, seed=self._BACKEND_SEED)
        return transpile(qc, backend=backend)

    def get_physical_metrics(self, qc: QuantumCircuit) -> dict:
        """Ritorna gateCount e depth del circuito dopo la transpilazione."""
        return self._extract_metrics(qc)

    @staticmethod
    def _extract_metrics(qc: QuantumCircuit) -> dict:
        """Calcola la coppia gateCount/depth condivisa da metriche astratte e fisiche."""
        return {"gateCount": qc.size(), "depth": qc.depth()}
