"""Implementazione concreta dell'unico punto di contatto ammesso con il framework Qiskit."""

import sys
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ControlFlowOp
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

        # Guardia esplicita PRIMA di toccare qualunque misura: lo Statevector rappresenta lo
        # stato quantistico PURO e non puo' modellare feedback classico. Un circuito con
        # istruzioni condizionate da bit classici (c_if legacy / if_test -> control-flow op) e'
        # fuori scope e non va confrontato silenziosamente.
        self._reject_classical_feedback(baseline_circuit)
        self._reject_classical_feedback(new_circuit)

        baseline_state = Statevector.from_instruction(self._strip_measurements(baseline_circuit))
        new_state = Statevector.from_instruction(self._strip_measurements(new_circuit))

        return baseline_state.equiv(new_state)

    @staticmethod
    def _reject_classical_feedback(circuit: QuantumCircuit) -> None:
        """Solleva se il circuito contiene istruzioni condizionate da bit classici (fuori scope).

        In Qiskit 2.x il feedback classico (c_if / if_test) e' modellato da control-flow op
        (IfElseOp, WhileLoopOp, SwitchCaseOp); l'attributo legacy .condition e' controllato in
        aggiunta per robustezza. Casi del genere non sono confrontabili via Statevector.
        """
        for instruction in circuit.data:
            operation = instruction.operation
            if isinstance(operation, ControlFlowOp) or getattr(operation, "condition", None):
                raise NotImplementedError(
                    "check_equivalence non supporta circuiti con misure intermedie e feedback "
                    "classico (c_if / if_test) — fuori scope per questo progetto."
                )

    @staticmethod
    def _strip_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
        """Ritorna una copia del circuito senza istruzioni measure, per il confronto Statevector.

        Le misure vengono rimosse OVUNQUE si trovino, non solo se "finali" secondo il DAG:
        necessario per pattern come misure per-qubit intervallate da barrier (vedi idq-fixed.py),
        che remove_final_measurements lascerebbe in parte al loro posto. I barrier restano: non
        hanno effetto sullo stato. La guardia _reject_classical_feedback deve essere gia' passata.
        """
        pure_circuit = circuit.copy_empty_like()
        for instruction in circuit.data:
            if instruction.operation.name == "measure":
                continue
            pure_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)
        return pure_circuit

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
