"""Implementazione concreta dell'unico punto di contatto ammesso con il framework Qiskit."""

import sys
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from itertools import combinations
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    partial_trace,
    state_fidelity,
)

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade


class QiskitFacade(IQiskitFacade):
    """Incapsula isolamento, equivalenza funzionale e misura degli smell tramite Qiskit."""

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

    # Errore di gate usato SOLO per la forma esponenziale (1 - error)^(l*c) con cui il paper
    # QSMELL presenta la metrica Long Circuit nelle sue tabelle. Non entra nel valore della
    # metrica: l'implementazione di riferimento (qsmell/smell/LC.py) ritorna il prodotto l*c
    # grezzo, senza alcuna costante hardware. Il valore adottato e' l'errore massimo di
    # GenericBackendV2(num_qubits=5, seed=42), il backend su cui questa facade transpilava finche'
    # produceva anche le metriche fisiche; e' congelato come costante perche' quel backend lo
    # estrae pseudo-casualmente e lo fa variare con la dimensione richiesta (0.00416 a 3 qubit,
    # 0.00496 a 10), il che renderebbe non confrontabili fra loro circuiti di taglia diversa
    # nello stesso dataset. Da qui in poi e' un numero fissato, non piu' riletto da nulla:
    # la transpilazione e' uscita dal progetto insieme alle metriche di costo.
    _MAX_GATE_ERROR = 0.00485

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

    def calculate_smell_metrics(self, code: str) -> dict:
        """Isola il codice e ritorna le metriche Long Circuit e Idle Qubits secondo QSMELL.

        Metriche definite da Chen et al., "The Smelly Eight: An Empirical Study on the
        Prevalence of Code Smells in Quantum Computing", ICSE 2023
        (DOI 10.1109/ICSE48619.2023.00041), e allineate all'implementazione di riferimento
        github.com/jose/qsmell (moduli qsmell/smell/LC.py e qsmell/smell/IdQ.py).

        Il payload ha forma:
        {"longCircuit": {"maxOpsPerQubit": int, "maxParallelOps": int, "value": int,
                         "gateError": float, "errorFreeProbability": float},
         "idleQubits": {"value": int, "worstQubit": Optional[int]}}.

        "value" di longCircuit e' il prodotto l*c ritornato dallo strumento di riferimento;
        "errorFreeProbability" e' la forma (1 - error)^(l*c) con cui il paper presenta la stessa
        metrica nelle tabelle. Le SOGLIE non compaiono qui: questa e' la misura, il confronto con
        una soglia e' politica di rilevamento e vive nel livello che decide (stessa separazione
        gia' in atto fra detection_thresholds e ValidationService._is_improvement).

        "maxOpsQubits" e' un PUNTATORE, non una misura: gli indici dei qubit che realizzano il
        massimo, cioe' gli unici da cui abbia senso rimuovere operazioni. Serve al DetectorAgent
        per prescrivere una riparazione, e come worstQubit non viene mai persistito -- il
        contratto dati (SmellMetrics) porta solo le misure. Sui 72 circuiti del dataset sintetico
        il massimo e' condiviso da piu' di un qubit in 34 casi: e' una lista perche' togliere
        operazioni da UNO SOLO dei qubit al massimo non abbassa l di un'unita'.

        "operationsPerQubit" e' la sequenza REALE di operazioni di ciascun qubit, una stringa per
        riga della matrice, barrier esclusi. Esiste per chiudere un divario che ha prodotto
        prescrizioni inventate: le metriche descrivono il circuito ESEGUITO (l = 21 operazioni),
        ma il sorgente puo' costruirle con un for di 10 iterazioni su 8 righe. Ricevendo solo i
        conteggi, il DetectorAgent immaginava uno srotolamento inesistente e citava righe che nel
        file non c'erano ("remove lines 2-3 ... 24-25" su un sorgente di 8 righe). La sequenza
        rende visibile cosa il circuito fa davvero, senza doverlo dedurre. Costa ~80 token sul
        circuito piu' grande del dataset ed e' input esatto, non una trascrizione generata dal
        modello -- che era il compito su cui aveva gia' fallito.

        NON ESISTE LA SIMMETRICA PER LE COLONNE, ed e' deliberato. Tre ragioni misurate:
        (1) sapere quali colonne realizzano c non aiuta a scegliere cosa rimuovere -- due
        ridondanze sullo stesso qubit, una in colonna affollata e una scarica, danno lo stesso
        risultato (l*c 24 -> 16 in entrambi i casi), perche' il packing ASAP richiude i buchi;
        (2) l'unica leva DIRETTA su c e' serializzare con barrier, che a gate invariati porta
        l*c da 8 a 6 preservando l'equivalenza: e' un "fix" peggiore dell'originale su hardware
        reale, e non va mostrato al modello; (3) esporre l'intera matrice e' gia' stato provato
        nella generazione sintetica ed e' fallito (degenerazione del modello sul disegno della
        matrice). L'asimmetria e' strutturale: l si conta per riga e la rimozione agisce su una
        riga, c e' una proprieta' emergente del packing che nessuno controlla direttamente.
        """
        circuit = self.isolate_circuit(code)
        matrix = self._execution_matrix(circuit)

        operation_names = [self._real_operations(row) for row in matrix]
        ops_per_qubit = [len(names) for names in operation_names]
        max_ops_per_qubit = max(ops_per_qubit, default=0)
        max_parallel_ops = max(
            (self._count_operations(column) for column in zip(*matrix)), default=0
        )
        product = max_ops_per_qubit * max_parallel_ops
        idle_value, worst_qubit = self._idle_qubits_metric(matrix)

        return {
            "longCircuit": {
                "maxOpsPerQubit": max_ops_per_qubit,
                "maxParallelOps": max_parallel_ops,
                "maxOpsQubits": [
                    index for index, count in enumerate(ops_per_qubit) if count == max_ops_per_qubit
                ]
                if max_ops_per_qubit
                else [],
                "operationsPerQubit": [", ".join(names) for names in operation_names],
                "value": product,
                "gateError": self._MAX_GATE_ERROR,
                "errorFreeProbability": (1 - self._MAX_GATE_ERROR) ** product,
            },
            "idleQubits": {"value": idle_value, "worstQubit": worst_qubit},
        }

    @staticmethod
    def _execution_matrix(circuit: QuantumCircuit) -> list[list[str]]:
        """Costruisce la execution matrix di QSMELL: righe = qubit, colonne = timestamp.

        Porta l'algoritmo di leftqc2matrix (qsmell/utils/quantum_circuit_to_matrix.py): ogni
        istruzione viene collocata al primo livello libero comune a TUTTI i bit che tocca, e
        quei bit avanzano insieme a quel livello. I bit classici NON compaiono fra le righe
        (entrambe le metriche li scartano) ma partecipano al calcolo dei livelli, altrimenti due
        measure che condividono un registro classico finirebbero erroneamente in parallelo.

        I barrier vengono INSERITI in matrice, non filtrati qui: pur non contando come operazioni
        nelle metriche, occupano un livello e agiscono da punto di sincronizzazione, impedendo
        alle istruzioni che li seguono di risalire in colonne precedenti. Rimuoverli in questa
        fase cambierebbe il packing e quindi i valori di entrambe le metriche.

        Le celle contengono il solo nome dell'operazione: le metriche interrogano il contenuto
        unicamente per "cella vuota" e "inizia per barrier", quindi la firma dei parametri che
        lo strumento di riferimento aggiunge al nome sarebbe informazione morta.
        """
        bit_level = dict.fromkeys(list(circuit.qubits) + list(circuit.clbits), 0)
        placements: list[tuple[int, int, str]] = []
        depth = 0

        for instruction in circuit.data:
            involved = list(instruction.qubits) + list(instruction.clbits)
            level = max(bit_level[bit] for bit in involved) + 1
            for bit in involved:
                bit_level[bit] = level
            depth = max(depth, level)
            for qubit in instruction.qubits:
                placements.append(
                    (level, circuit.find_bit(qubit).index, instruction.operation.name)
                )

        matrix = [["" for _ in range(depth)] for _ in range(circuit.num_qubits)]
        for level, qubit_index, operation_name in placements:
            matrix[qubit_index][level - 1] = operation_name
        return matrix

    @staticmethod
    def _real_operations(cells) -> list[str]:
        """Nomi delle operazioni reali di una riga o colonna: vuote e barrier escluse."""
        return [cell for cell in cells if cell and not cell.lower().startswith("barrier")]

    @classmethod
    def _count_operations(cls, cells) -> int:
        """Quante celle sono occupate da un'operazione reale."""
        return len(cls._real_operations(cells))

    @staticmethod
    def _idle_qubits_metric(matrix: list[list[str]]) -> tuple[int, Optional[int]]:
        """Massimo numero di timestamp vuoti fra due operazioni consecutive dello stesso qubit.

        Porta la logica di qsmell/smell/IdQ.py, comprese le sue due convenzioni non ovvie: le
        colonne di barrier sono saltate del tutto (non contano ne' come operazione ne' come
        attesa), e la coda di celle vuote DOPO l'ultima operazione di un qubit non viene
        conteggiata — un qubit che smette di essere usato non accumula attesa fino a fine
        circuito. Ritorna anche l'indice del qubit che realizza il massimo, per permettere al
        RefactorerAgent di sapere su quale qubit intervenire (None se nessuno attende).
        """
        worst_value = 0
        worst_qubit: Optional[int] = None

        for qubit_index, row in enumerate(matrix):
            gap = -1  # -1 = la prima operazione del qubit non e' ancora stata incontrata
            for cell in row:
                if cell.lower().startswith("barrier"):
                    continue
                if not cell:
                    if gap != -1:
                        gap += 1
                elif gap == -1:
                    gap = 0
                else:
                    if gap > worst_value:
                        worst_value, worst_qubit = gap, qubit_index
                    gap = 0

        return worst_value, worst_qubit

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
