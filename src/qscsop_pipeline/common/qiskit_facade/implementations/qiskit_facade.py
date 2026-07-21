"""Implementazione concreta dell'unico punto di contatto ammesso con il framework Qiskit."""

import sys
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from itertools import combinations
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ControlFlowOp
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade


class QiskitFacade(IQiskitFacade):
    """Incapsula isolamento, transpilazione e calcolo metriche tramite Qiskit."""

    # GenericBackendV2 sostituisce i Fake Backend concreti a 5 qubit (FakeManila, FakeAthens),
    # rimossi da qiskit.providers.fake_provider a partire da Qiskit 2.x: genera un backend
    # parametrico con basis gates e coupling map realistici, equivalente ai fini della transpilazione.
    _DEFAULT_NUM_QUBITS = 5
    _BACKEND_SEED = 42

    # Limite oltre il quale il confronto a dimensioni diverse non viene tentato: la DensityMatrix
    # cresce come 2^N x 2^N complessi, quindi a 12 qubit occupa gia' ~268 MB (4096^2 * 16 byte) e
    # raddoppia di esponente ad ogni qubit in piu', diventando rapidamente proibitiva. I circuiti
    # NISQ di questo dataset stanno ampiamente sotto la soglia: superarla segnala un caso fuori
    # scope, da rifiutare esplicitamente invece che tentare un'allocazione insostenibile.
    _MAX_PARTIAL_TRACE_QUBITS = 12

    # Tolleranza sulla fidelity: due stati identici danno 1.0 a meno dell'errore numerico in
    # virgola mobile accumulato da partial_trace e dalla radice di matrice interna a state_fidelity.
    _EQUIVALENCE_TOLERANCE = 1e-6

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
        """Isola entrambi i codici e verifica se sono funzionalmente equivalenti.

        A parita' di qubit il confronto e' diretto fra Statevector; se il numero di qubit
        differisce (caso tipico di una correzione Idle Qubits riuscita, che ne rimuove uno) si
        passa al confronto via partial_trace, vedi _check_equivalence_across_dimensions.

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

        baseline_pure = self._strip_measurements(baseline_circuit)
        new_pure = self._strip_measurements(new_circuit)

        if baseline_pure.num_qubits != new_pure.num_qubits:
            return self._check_equivalence_across_dimensions(baseline_pure, new_pure)

        baseline_state = Statevector.from_instruction(baseline_pure)
        new_state = Statevector.from_instruction(new_pure)

        return baseline_state.equiv(new_state)

    @classmethod
    def _check_equivalence_across_dimensions(
        cls, first_circuit: QuantumCircuit, second_circuit: QuantumCircuit
    ) -> bool:
        """Confronta due circuiti con numero di qubit diverso tracciando via i qubit in eccesso.

        Uno Statevector ha dimensione 2^N: due circuiti con N diverso non sono confrontabili
        direttamente. Si riduce allora il circuito piu' grande con partial_trace, che risponde
        esattamente alla domanda "ignorando questi qubit, cosa si osserva sul resto?". Se il qubit
        tracciato via era davvero idle (non correlato agli altri) la riduzione restituisce
        esattamente lo stato del circuito piu' piccolo; se era invece entangled, produce uno stato
        MISTO, distinguibile da qualunque stato puro ridotto — il confronto fallisce correttamente,
        senza diventare permissivo solo perche' le dimensioni sono cambiate.
        """
        larger_circuit, smaller_circuit = (
            (first_circuit, second_circuit)
            if first_circuit.num_qubits > second_circuit.num_qubits
            else (second_circuit, first_circuit)
        )
        larger_n = larger_circuit.num_qubits
        smaller_n = smaller_circuit.num_qubits

        # Controllo PRIMA di costruire qualunque DensityMatrix: oltre la soglia l'allocazione
        # sarebbe insostenibile, quindi si rifiuta il confronto invece di tentarlo.
        if larger_n > cls._MAX_PARTIAL_TRACE_QUBITS:
            raise ValueError(
                f"Circuito troppo grande per il confronto di equivalenza via partial_trace "
                f"({larger_n} qubit, limite {cls._MAX_PARTIAL_TRACE_QUBITS}) — "
                f"validazione non eseguita."
            )

        larger_state = DensityMatrix.from_instruction(larger_circuit)
        smaller_state = DensityMatrix.from_instruction(smaller_circuit)

        # SmellReportDTO non porta un indice strutturato di QUALE qubit sia stato rimosso (solo
        # has_smells e testo libero), quindi si prova esaustivamente ogni combinazione di qubit da
        # tracciare via: basta che una produca equivalenza. Per circuiti di questa scala il numero
        # di combinazioni e' minimo (3 per un 3 -> 2 qubit), costo trascurabile.
        for traced_qubits in combinations(range(larger_n), larger_n - smaller_n):
            reduced_state = partial_trace(larger_state, list(traced_qubits))
            # validate=False: partial_trace produce per costruzione uno stato a traccia 1, ma la
            # verifica interna di Qiskit e' esatta e solleverebbe su deviazioni puramente numeriche.
            fidelity = state_fidelity(reduced_state, smaller_state, validate=False)
            if fidelity >= 1.0 - cls._EQUIVALENCE_TOLERANCE:
                return True

        return False

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
