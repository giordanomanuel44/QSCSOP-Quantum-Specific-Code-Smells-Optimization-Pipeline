"""Implementazione concreta dell'unico punto di contatto ammesso con il framework Qiskit."""

import sys
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector

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

        # source_code proviene da mining di repository reali e puo' contenere print() di
        # caratteri non rappresentabili nell'encoding di default della console (es. cp1252 su
        # Windows, con simboli come i box-drawing del disegno ASCII di un circuito o lettere
        # greche). Senza questo reconfigure, un simile print() farebbe fallire l'intero exec()
        # con UnicodeEncodeError, scartando un circuito altrimenti valido a monte in
        # QuantumMetricsService. errors="replace" sostituisce i caratteri non rappresentabili
        # invece di sollevare eccezione, senza cambiare l'encoding effettivo dello stream.
        with self._tolerant_stdio():
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

    def compile_circuit(self, source_code: str) -> tuple[bool, Optional[str]]:
        """Tenta di isolare source_code; ritorna (True, None) o (False, messaggio d'errore)."""
        try:
            self.isolate_circuit(source_code)
        except Exception:
            return False, traceback.format_exc()
        return True, None

    def check_equivalence(self, baseline_code: str, new_code: str) -> bool:
        """Isola entrambi i codici e confronta i rispettivi Statevector.

        Il chiamante deve aver gia' verificato compile_circuit(new_code) prima di arrivare qui:
        un'eccezione di isolamento non viene gestita, ma propagata.
        """
        baseline_circuit = self.isolate_circuit(baseline_code)
        new_circuit = self.isolate_circuit(new_code)

        baseline_state = Statevector.from_instruction(baseline_circuit)
        new_state = Statevector.from_instruction(new_circuit)

        return baseline_state.equiv(new_state)

    def calculate_metrics(self, code: str) -> dict:
        """Isola, transpila e ritorna {"abstractMetrics": {...}, "physicalMetrics": {...}}."""
        qc = self.isolate_circuit(code)
        abstract_metrics = self.get_abstract_metrics(qc)
        transpiled_qc = self.transpile_circuit(qc)
        physical_metrics = self.get_physical_metrics(transpiled_qc)
        return {
            "abstractMetrics": abstract_metrics,
            "physicalMetrics": physical_metrics,
        }

    @staticmethod
    @contextmanager
    def _tolerant_stdio() -> Generator[None, None, None]:
        """Rende temporaneamente stdout/stderr tolleranti a caratteri non rappresentabili.

        Alcuni stream sostitutivi (es. la cattura di pytest, o pipe senza terminale) non
        espongono reconfigure(): in quel caso non c'e' nulla da tollerare, si prosegue tali e
        quali.
        """
        streams = [s for s in (sys.stdout, sys.stderr) if hasattr(s, "reconfigure")]
        original_errors = [s.errors for s in streams]
        for s in streams:
            s.reconfigure(errors="replace")
        try:
            yield
        finally:
            for s, errors in zip(streams, original_errors, strict=True):
                s.reconfigure(errors=errors)
