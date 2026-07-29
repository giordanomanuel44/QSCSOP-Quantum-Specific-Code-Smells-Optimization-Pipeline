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
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace, state_fidelity

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

    # Stesso ordine di grandezza del limite sopra, applicato al ramo a stessa dimensione: la
    # matrice unitaria di Operator e' anch'essa 2^N x 2^N complessi (in piu', non solo 2^N come lo
    # Statevector che sostituisce). Prima dell'introduzione di Operator questo ramo non aveva un
    # limite perche' Statevector era economico (O(2^N), non O(4^N)); ora serve la stessa guardia.
    _MAX_OPERATOR_QUBITS = 12

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

        A parita' di qubit il confronto e' RIGOROSO: si costruisce l'Operator (matrice unitaria
        completa) di entrambi i circuiti, valido per qualunque stato di ingresso per costruzione —
        non solo per |0...0>. Se il numero di qubit differisce (caso tipico di una correzione Idle
        Qubits riuscita, che ne rimuove uno) l'operatore unitario completo non si applica (le
        dimensioni non coincidono): si passa al confronto via partial_trace, un campione PRAGMATICO
        di due stati di ingresso (|0...0> e |1...1>), non esaustivo — vedi
        _check_equivalence_across_dimensions per il dettaglio e la motivazione della differenza di
        rigore tra i due rami.

        Il chiamante deve aver gia' verificato compile_circuit(new_code) prima di arrivare qui:
        un'eccezione di isolamento non viene gestita, ma propagata.
        """
        baseline_circuit = self.isolate_circuit(baseline_code)
        new_circuit = self.isolate_circuit(new_code)

        # Guardia esplicita PRIMA di toccare qualunque misura: sia Operator che DensityMatrix
        # rappresentano l'evoluzione di uno stato quantistico PURO e non possono modellare feedback
        # classico. Un circuito con istruzioni condizionate da bit classici (c_if legacy / if_test
        # -> control-flow op) e' fuori scope e non va confrontato silenziosamente.
        self._reject_classical_feedback(baseline_circuit)
        self._reject_classical_feedback(new_circuit)

        baseline_pure = self._strip_measurements(baseline_circuit)
        new_pure = self._strip_measurements(new_circuit)

        if baseline_pure.num_qubits != new_pure.num_qubits:
            return self._check_equivalence_across_dimensions(baseline_pure, new_pure)

        return self._check_equivalence_same_dimension(baseline_pure, new_pure)

    @classmethod
    def _check_equivalence_same_dimension(
        cls, baseline_circuit: QuantumCircuit, new_circuit: QuantumCircuit
    ) -> bool:
        """Confronta due circuiti con lo STESSO numero di qubit via Operator (matrice unitaria).

        A differenza di uno Statevector (che testa solo il comportamento a partire da |0...0>),
        Operator rappresenta l'intera trasformazione applicata dal circuito: due Operator
        equivalenti garantiscono stessa uscita per QUALUNQUE stato di ingresso, non solo il
        default. Questo chiude un falso positivo reale osservato in produzione: due circuiti
        composti esclusivamente da gate condizionati (es. Toffoli) restano entrambi inerti da
        |0...0> indipendentemente da quanto siano strutturalmente diversi, e uno Statevector non
        li avrebbe mai distinti.
        """
        num_qubits = baseline_circuit.num_qubits

        # Controllo PRIMA di costruire qualunque Operator: la matrice unitaria e' 2^N x 2^N
        # complessi, cresce piu' rapidamente della DensityMatrix limitata sopra (stessa soglia,
        # stesso motivo: oltre 12 qubit l'allocazione diventa insostenibile).
        if num_qubits > cls._MAX_OPERATOR_QUBITS:
            raise ValueError(
                f"Circuito troppo grande per il confronto di equivalenza via Operator "
                f"({num_qubits} qubit, limite {cls._MAX_OPERATOR_QUBITS}) — "
                f"validazione non eseguita."
            )

        baseline_op = Operator(baseline_circuit)
        new_op = Operator(new_circuit)

        return baseline_op.equiv(new_op, atol=cls._EQUIVALENCE_TOLERANCE)

    @classmethod
    def _check_equivalence_across_dimensions(
        cls, first_circuit: QuantumCircuit, second_circuit: QuantumCircuit
    ) -> bool:
        """Confronta due circuiti con numero di qubit diverso tracciando via i qubit in eccesso.

        DIFFERENZA DI RIGORE rispetto a _check_equivalence_same_dimension: qui l'operatore
        unitario completo non si applica (le due dimensioni non coincidono), quindi non esiste un
        confronto valido per QUALUNQUE stato di ingresso come per il ramo a stessa dimensione. La
        soluzione adottata e' PRAGMATICA, non esaustiva: si ripete il confronto partial_trace per
        due soli stati di ingresso, |0...0> (il default implicito di from_instruction, comportamento
        storico) e |1...1> (tutti i qubit invertiti via _prepare_flipped_variant). Il risultato
        complessivo e' equivalente solo se ENTRAMBI i campioni lo sono. Due campioni bastano a
        smascherare il caso reale osservato in produzione (gate Toffoli/ccx concatenati, inerti da
        |0...0> ma attivi da |1...1>), ma non costituiscono una prova di equivalenza di canale
        quantistico completa: un futuro miglioramento potrebbe ampliare il campione; la piena
        equivalenza per dimensioni diverse resta fuori scope per questo progetto.

        Per ciascuno stato di ingresso, uno Statevector ha dimensione 2^N: due circuiti con N
        diverso non sono confrontabili direttamente. Si riduce allora il circuito piu' grande con
        partial_trace, che risponde esattamente alla domanda "ignorando questi qubit, cosa si
        osserva sul resto?". Se il qubit tracciato via era davvero idle (non correlato agli altri)
        la riduzione restituisce esattamente lo stato del circuito piu' piccolo; se era invece
        entangled, produce uno stato MISTO, distinguibile da qualunque stato puro ridotto — il
        confronto fallisce correttamente, senza diventare permissivo solo perche' le dimensioni
        sono cambiate.
        """
        return cls._partial_trace_equivalent(
            first_circuit, second_circuit
        ) and cls._partial_trace_equivalent(
            cls._prepare_flipped_variant(first_circuit),
            cls._prepare_flipped_variant(second_circuit),
        )

    @classmethod
    def _partial_trace_equivalent(
        cls, first_circuit: QuantumCircuit, second_circuit: QuantumCircuit
    ) -> bool:
        """Esegue il confronto partial_trace fra due circuiti per il loro stato di ingresso attuale.

        Fattorizzata fuori da _check_equivalence_across_dimensions per essere riusata identica sia
        sui circuiti originali (stato di ingresso |0...0>) sia sulla loro variante con tutti i
        qubit invertiti (stato di ingresso |1...1>), senza duplicare guardia e ciclo.
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
    def _prepare_flipped_variant(circuit: QuantumCircuit) -> QuantumCircuit:
        """Ritorna una copia del circuito con X su ogni qubit anteposto.

        Serve a testare il comportamento del circuito partendo da |1...1> invece di |0...0> (il
        default implicito di from_instruction), per smascherare circuiti composti da gate
        condizionati (es. Toffoli) che restano inerti da |0...0> indipendentemente da quanto siano
        strutturalmente diversi.
        """
        flipped = circuit.copy_empty_like()
        flipped.x(range(circuit.num_qubits))
        flipped.compose(circuit, inplace=True)
        return flipped

    @staticmethod
    def _reject_classical_feedback(circuit: QuantumCircuit) -> None:
        """Solleva se il circuito contiene istruzioni condizionate da bit classici (fuori scope).

        In Qiskit 2.x il feedback classico (c_if / if_test) e' modellato da control-flow op
        (IfElseOp, WhileLoopOp, SwitchCaseOp); l'attributo legacy .condition e' controllato in
        aggiunta per robustezza. Casi del genere non sono confrontabili via Operator/DensityMatrix.
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
        """Ritorna una copia del circuito senza istruzioni measure, per il confronto di equivalenza.

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
        """Isola, transpila e ritorna logicalQubits + abstractMetrics + physicalMetrics.

        Il payload ha forma:
        {"logicalQubits": int, "abstractMetrics": {...}, "physicalMetrics": {...}}.
        """
        qc = self.isolate_circuit(code)
        abstract_metrics = self.get_abstract_metrics(qc)
        transpiled_qc = self.transpile_circuit(qc)
        physical_metrics = self.get_physical_metrics(transpiled_qc)
        # logicalQubits e' letto dal circuito ISOLATO (qc), non dal transpiled_qc: la
        # transpilazione padda il circuito ai qubit del backend (almeno _DEFAULT_NUM_QUBITS = 5),
        # falsando il conteggio logico. E' fratello di abstractMetrics/physicalMetrics, non
        # contenuto in essi: stessa gerarchia del Listing 1.2 della tesi e della classe
        # CircuitVersion, dove logical_qubits e' un campo separato dalle due CircuitMetrics.
        return {
            "logicalQubits": qc.num_qubits,
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
