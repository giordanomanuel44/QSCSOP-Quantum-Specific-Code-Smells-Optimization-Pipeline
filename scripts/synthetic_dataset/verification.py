"""Verifica strutturale post-generazione dei circuiti sintetici.

Isolamento e compilazione dei circuiti passano SEMPRE per la QiskitFacade reale
(isolate_circuit, compile_circuit) -- zero duplicazione di quella logica, come richiesto.

Eccezione dichiarata: la regola "solo la facade importa qiskit" (CLAUDE.md, punto 4) e' scoped
a src/qscsop_pipeline/; questo modulo vive fuori da quel perimetro (tooling di preparazione dati,
stesso ruolo di scripts/fetch_datasets.py). Statevector viene importato direttamente qui perche'
la facade non espone un metodo pubblico per costruire lo stato di un circuito isolato (i suoi
equivalenti privati sono pensati per confrontare DUE circuiti, non per verificare "e' tutto
|0...0>?" di uno solo). La rimozione delle misure per costruire lo Statevector e' reimplementata
localmente in poche righe: non e' la logica di isolamento/compilazione che va riusata dalla
facade, e' un filtro minimo necessario perche' Statevector non ammette istruzioni measure.
"""

from qiskit.quantum_info import Statevector

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.mas.dto.quantum_smell_type import QuantumSmellType
from scripts.synthetic_dataset.prompts import GeneratedCircuit, BatchTheme, SmellFocus
import re


def structural_idle_check(source_code: str, facade: QiskitFacade) -> dict:
    """Isola il circuito e ritorna quali indici di qubit non hanno ricevuto operazioni reali.

    Conta le operazioni REALI leggendo circuit.data (non il testo sorgente) -- cattura
    automaticamente casi come le chiamate orfane (es. .control() il cui risultato non e' mai
    assegnato: semplicemente non compare in circuit.data). I barrier sono esclusi dal conteggio:
    non hanno effetto sullo stato (stesso principio del docstring di
    QiskitFacade._strip_measurements), quindi un qubit toccato SOLO da un barrier register-wide
    resterebbe altrimenti mascherato come "attivo" senza esserlo davvero.

    LIMITE STRUTTURALE (verificato empiricamente su idq-smelly.py): questo controllo cattura
    solo il caso "zero operazioni in assoluto" (es. la chiamata orfana). NON cattura un qubit
    che riceve operazioni reali il cui effetto NETTO e' pero' nullo -- es. q2 in idq-smelly.py
    riceve due H consecutivi (H*H = identita') seguiti da soli gate di fase prima di una misura
    Z: e' idle nel comportamento risultante (stessa definizione usata da DetectorAgent), ma
    "touched" per questo conteggio, perche' riceve gate reali. Riconoscere la cancellazione
    netta richiede simulare lo stato, non solo contare le operazioni -- fuori scope per un
    controllo puramente strutturale. Quel caso e' coperto da verify_declared_idle_qubits, che
    delega la simulazione dello stato del singolo qubit a QiskitFacade.is_qubit_idle.
    """
    circuit = facade.isolate_circuit(source_code)

    touched_indices: set[int] = set()
    for instruction in circuit.data:
        if instruction.operation.name == "barrier":
            continue
        for qubit in instruction.qubits:
            touched_indices.add(circuit.find_bit(qubit).index)

    idle_indices = [i for i in range(circuit.num_qubits) if i not in touched_indices]
    return {"num_qubits": circuit.num_qubits, "idle_qubit_indices": idle_indices}


def structural_dead_circuit_check(source_code: str, facade: QiskitFacade) -> bool:
    """Isola il circuito e verifica se il suo stato finale coincide con |0...0>.

    Un circuito interamente inerte (es. solo gate condizionati mai attivati) e' il caso
    degenere vietato dal vincolo negativo 3d -- va scartato, non e' rappresentativo di uno smell
    genuino (che presuppone computazione reale con uno spreco incorporato, non "tutto morto").
    """
    circuit = facade.isolate_circuit(source_code)

    pure_circuit = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            continue
        pure_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)

    if pure_circuit.parameters:
        # Il circuito puo' contenere Parameter mai assegnati (visto nel few-shot "chiamata
        # orfana" di prompts.py, dove theta1 resta libero): Statevector non accetta operazioni
        # con parametri liberi. Il verdetto di degenerazione non dipende dal valore specifico
        # del parametro (a meno di coincidenze patologiche), quindi si lega un valore generico
        # e non nullo -- 0 andrebbe evitato perche' coinciderebbe con l'identita' per molti gate
        # parametrici (es. rx(0) = I), falsando il controllo.
        generic_value = 0.4321
        pure_circuit = pure_circuit.assign_parameters(
            {param: generic_value for param in pure_circuit.parameters}
        )

    state = Statevector(pure_circuit)
    zero_state = Statevector.from_label("0" * pure_circuit.num_qubits)
    return state.equiv(zero_state)


def verify_generated_circuit(circuit: GeneratedCircuit, facade: QiskitFacade) -> dict:
    """Orchestra compile_circuit -> structural_dead_circuit_check -> structural_idle_check.

    Diagnostica pura: non solleva eccezioni per circuiti che falliscono la verifica, ritorna un
    dict con l'esito. Il chiamante decide cosa fare (scrivere su disco, scartare, segnalare).

    NOTA IMPORTANTE: la verifica automatica qui e' strutturalmente PIU' DEBOLE per Long Circuit
    che per Idle Qubits. Non esiste un controllo deterministico generale per "ridondanza
    semantica" -- l'identita' SWAP-via-coniugazione-H (Example 3 in prompts.py) lo dimostra:
    riconoscerla richiede algebra di gruppo, non e' enumerabile a priori per ogni circuito
    possibile. Per i circuiti etichettati long_circuit, questa funzione puo' solo confermare che
    il circuito compila e non e' degenere: la verifica di merito passa da
    verify_declared_simplification (che confronta l'originale con la versione piu' corta fornita
    dal generatore) e dal campionamento con DetectorAgent (vedi generate.py).
    """
    compiles, compile_error = facade.compile_circuit(circuit.source_code)
    if not compiles:
        return {"compiles": False, "compile_error": compile_error, "discarded": True}

    if structural_dead_circuit_check(circuit.source_code, facade):
        return {"compiles": True, "dead_circuit": True, "discarded": True}

    idle_result = structural_idle_check(circuit.source_code, facade)
    idle_declared = QuantumSmellType.IDLE_QUBITS in circuit.intended_smells
    idle_declaration_consistent = idle_declared == bool(idle_result["idle_qubit_indices"])

    return {
        "compiles": True,
        "dead_circuit": False,
        "discarded": False,
        "idle_qubit_indices": idle_result["idle_qubit_indices"],
        "idle_declaration_consistent": idle_declaration_consistent,
    }

def verify_declared_simplification(circuit: GeneratedCircuit, facade: QiskitFacade) -> bool | None:
    """Verifica che il circuito semplificato dichiarato sia davvero equivalente all'originale.

    Colma il limite documentato in verify_generated_circuit: la ridondanza semantica non e'
    enumerabile a priori, ma se il generatore FORNISCE la versione piu' corta che sostiene esistere
    (campo simplified_source_code), l'equivalenza fra le due diventa verificabile a macchina —
    riusando lo stesso QiskitFacade.check_equivalence con cui ValidationService valida i
    refactoring nel sistema reale, non un confronto reimplementato qui.

    Ritorna None se il circuito non dichiara long_circuit (niente da verificare), False se lo
    dichiara ma il campo e' assente/vuoto o i due circuiti non sono equivalenti, True se
    l'equivalenza e' confermata. Diagnostica pura come il resto del modulo: qualunque eccezione di
    isolamento o confronto (codice non compilabile, feedback classico, circuito oltre i limiti
    dimensionali della facade) vale come verifica NON superata, non come errore da propagare.
    """
    if QuantumSmellType.LONG_CIRCUIT not in circuit.intended_smells:
        return None

    simplified_code = circuit.simplified_source_code
    if not simplified_code or not simplified_code.strip():
        return False

    try:
        return facade.check_equivalence(circuit.source_code, simplified_code)
    except Exception:
        return False


def verify_declared_idle_qubits(circuit: GeneratedCircuit, facade: QiskitFacade) -> bool | None:
    """Verifica a macchina che almeno un qubit sia davvero idle, non solo dichiarato tale.

    Complementare a structural_idle_check, di cui copre esattamente il limite documentato: qui il
    verdetto passa da QiskitFacade.is_qubit_idle, che simula lo stato del singolo qubit e cattura
    quindi anche il caso "operazioni reali con effetto netto nullo" (due H consecutivi), quello su
    cui il generatore ha sbagliato piu' spesso in pratica.

    Ritorna None se il circuito non dichiara idle_qubits, True se almeno un qubit risulta
    verificato come idle, False altrimenti. Stessa diagnostica pura: un'eccezione della facade vale
    come verifica non superata.
    """
    if QuantumSmellType.IDLE_QUBITS not in circuit.intended_smells:
        return None

    try:
        num_qubits = facade.isolate_circuit(circuit.source_code).num_qubits
        return any(
            facade.is_qubit_idle(circuit.source_code, index) for index in range(num_qubits)
        )
    except Exception:
        return False


def matches_batch_theme(circuit: GeneratedCircuit, theme: BatchTheme) -> bool:
    """Verifica che intended_smells rispetti il focus dichiarato del lotto.

    Cattura esattamente il problema visto: lotti IDLE_QUBITS_ONLY che generavano
    circuiti puliti o addirittura long_circuit, invece del tema richiesto.
    """
    declared = {s.value for s in circuit.intended_smells}
    focus = theme.smell_focus

    if focus == SmellFocus.LONG_CIRCUIT_ONLY:
        return declared == {"long_circuit"}
    if focus == SmellFocus.IDLE_QUBITS_ONLY:
        return declared == {"idle_qubits"}
    if focus == SmellFocus.BOTH:
        return declared == {"long_circuit", "idle_qubits"}
    if focus == SmellFocus.NONE:
        return declared == set()
    return True

def is_near_duplicate(circuit: GeneratedCircuit, already_accepted: list[GeneratedCircuit]) -> str | None:
    """Confronta il circuito con quelli già accettati nello stesso lotto.

    Due livelli: (a) codice sorgente identico dopo aver rimosso spazi/a-capo superflui
    (cattura il caso visto nel lotto both_smells, dove 6 record erano byte-identici a
    parte una costante numerica); (b) stessa identica sequenza di NOMI di gate, ignorando
    quali qubit specifici vengono toccati (cattura duplicati "travestiti" con indici diversi).
    Ritorna l'ID del circuito duplicato se trovato, altrimenti None.
    """
    normalized_new = " ".join(circuit.source_code.split())

    for other in already_accepted:
        normalized_other = " ".join(other.source_code.split())
        if normalized_new == normalized_other:
            return "identical_source"

        # Confronto strutturale: sequenza dei nomi dei gate, ignorando gli indici
        gates_new = re.findall(r"qc\.(\w+)\(", circuit.source_code)
        gates_other = re.findall(r"qc\.(\w+)\(", other.source_code)
        if gates_new == gates_other and len(gates_new) > 2:
            return "same_gate_sequence"

    return None