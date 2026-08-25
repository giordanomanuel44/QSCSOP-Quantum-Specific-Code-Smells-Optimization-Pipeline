"""Verifica post-generazione dei circuiti sintetici.

Isolamento, compilazione e MISURA dei circuiti passano SEMPRE per la QiskitFacade reale
(isolate_circuit, compile_circuit, calculate_smell_metrics) -- zero duplicazione di quella
logica: l'etichetta di un circuito sintetico e' prodotta esattamente dallo stesso codice che
etichettera' i circuiti in produzione.

CAMBIO DI RUOLO rispetto alla versione precedente. Prima questo modulo doveva verificare le
DICHIARAZIONI del generatore (intended_smells, simplified_source_code): esistevano
verify_declared_simplification e verify_declared_idle_qubits proprio per smascherare
dichiarazioni false, ed era un lavoro strutturalmente ingrato -- per Long Circuit inteso come
"ridondanza semantica" non esiste nemmeno un controllo deterministico generale. Ora il generatore
non dichiara piu' nulla e l'etichetta si MISURA: quelle due funzioni sono sparite insieme
all'oggetto della loro verifica, e qui resta solo cio' che una misura non puo' dire da sola --
il circuito compila? assegna un solo circuito? non e' un duplicato? ha la forma
richiesta dal lotto?

Eccezione dichiarata: la regola "solo la facade importa qiskit" (CLAUDE.md, punto 4) e' scoped
a src/qscsop_pipeline/; questo modulo vive fuori da quel perimetro (tooling di preparazione dati,
stesso ruolo di scripts/fetch_datasets.py). Statevector viene importato direttamente qui perche'
la facade non espone un metodo pubblico per costruire lo stato di un circuito isolato (i suoi
equivalenti privati sono pensati per confrontare DUE circuiti, non per verificare "e' tutto
|0...0>?" di uno solo).
"""

import re

from qiskit.quantum_info import Statevector

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from scripts.synthetic_dataset.prompts import (
    LC_PRODUCT_CUTOFF,
    BatchTheme,
    GeneratedCircuit,
    IdleTarget,
)


def measure_shape(source_code: str, facade: QiskitFacade) -> dict:
    """Misura la forma QSMELL del circuito e ne deriva l'etichetta.

    E' il cuore del nuovo contratto: l'etichetta di un circuito sintetico non e' piu' quello che
    il generatore dice di aver scritto, e' quello che la facade misura. Le due soglie applicate
    qui sono le uniche in gioco: l*c >= LC_PRODUCT_CUTOFF per Long Circuit, IdQ > 0 per Idle
    Qubits.
    """
    metrics = facade.calculate_smell_metrics(source_code)
    long_circuit, idle_qubits = metrics["longCircuit"], metrics["idleQubits"]

    product = long_circuit["value"]
    idq = idle_qubits["value"]
    labels = []
    if product >= LC_PRODUCT_CUTOFF:
        labels.append("long_circuit")
    if idq > 0:
        labels.append("idle_qubits")

    return {
        "l": long_circuit["maxOpsPerQubit"],
        "c": long_circuit["maxParallelOps"],
        "lc_product": product,
        "idq": idq,
        "idq_worst_qubit": idle_qubits["worstQubit"],
        "measured_smells": labels,
    }


def structural_idle_check(source_code: str, facade: QiskitFacade) -> dict:
    """Isola il circuito e ritorna quali indici di qubit non hanno ricevuto operazioni reali.

    Conta le operazioni REALI leggendo circuit.data (non il testo sorgente) -- cattura
    automaticamente casi come le chiamate orfane (es. .control() il cui risultato non e' mai
    assegnato: semplicemente non compare in circuit.data). I barrier sono esclusi dal conteggio:
    non hanno effetto sullo stato (stesso principio del docstring di
    QiskitFacade._strip_measurements), quindi un qubit toccato SOLO da un barrier register-wide
    resterebbe altrimenti mascherato come "attivo" senza esserlo davvero.

    NON e' piu' un controllo di coerenza (non c'e' piu' una dichiarazione da confrontare), e'
    METADATO. Registra il punto cieco documentato di QSMELL: un qubit allocato e mai usato ha
    IdQ = 0, quindi la metrica lo considera pulito. Averlo nel ground truth permette di
    quantificare quanto spesso il generatore ci cade e di discuterlo in tesi, invece di
    perderlo.
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

    NON E' PIU' UN MOTIVO DI SCARTO, e' un metadato -- cambiato dopo un falso positivo su un
    circuito reale. data/raw/bugs4q/StackExchange_16_fix.py e' uno dei 9 circuiti affidabili del
    corpus (IdQ = 1 misurato, forma perfettamente valida), ma i suoi tre ccx partono tutti da
    controlli a |0> e non scattano mai: lo stato finale e' |00000> e questo controllo lo
    bocciava.

    Lo scarto aveva senso finche' gli smell erano definiti semanticamente ("qubit il cui effetto
    netto e' nullo"): su un circuito interamente inerte quella definizione degenerava. Le
    metriche QSMELL guardano la FORMA della execution matrix e non lo stato prodotto, quindi un
    circuito inerte ha metriche altrettanto valide di uno che computa. Resta comunque utile
    saperlo, per non costruire un dataset fatto di gusci vuoti: il vincolo e' scritto nel prompt
    e l'esito finisce nel ground truth, ma non decide piu' l'ammissione.
    """
    circuit = facade.isolate_circuit(source_code)

    pure_circuit = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            continue
        pure_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)

    if pure_circuit.parameters:
        # Il circuito puo' contenere Parameter mai assegnati (il caso reale si vede nel
        # controesempio B di prompts.py, dove theta1 resta libero): Statevector non accetta
        # operazioni con parametri liberi. Il verdetto di degenerazione non dipende dal valore
        # specifico del parametro (a meno di coincidenze patologiche), quindi si lega un valore
        # generico e non nullo -- 0 andrebbe evitato perche' coinciderebbe con l'identita' per
        # molti gate parametrici (es. rx(0) = I), falsando il controllo.
        generic_value = 0.4321
        pure_circuit = pure_circuit.assign_parameters(
            {param: generic_value for param in pure_circuit.parameters}
        )

    state = Statevector(pure_circuit)
    zero_state = Statevector.from_label("0" * pure_circuit.num_qubits)
    return state.equiv(zero_state)


def _assigned_circuit_count(source_code: str) -> int:
    """Quanti QuantumCircuit il sorgente lascia nel namespace (deve essere esattamente 1).

    Riproduce sui circuiti sintetici il criterio 3 di
    scripts/diagnostics/corpus_reliability_report.py, che ha scartato 6 circuiti reali su 91:
    isolate_circuit misura l'ULTIMO circuito assegnato, quindi un sorgente che ne assegna due
    fa misurare il pezzo sbagliato. Il vincolo e' scritto nel prompt; qui si verifica.
    """
    from qiskit import QuantumCircuit  # import locale: vedi eccezione dichiarata nel docstring

    namespace: dict = {}
    exec(source_code, namespace)  # noqa: S102 - stessa sandbox di facade.isolate_circuit
    return sum(1 for value in namespace.values() if isinstance(value, QuantumCircuit))


def verify_generated_circuit(circuit: GeneratedCircuit, facade: QiskitFacade) -> dict:
    """Orchestra compile -> circuito unico -> non degenere -> misura della forma.

    Diagnostica pura: non solleva eccezioni per circuiti che falliscono la verifica, ritorna un
    dict con l'esito. Il chiamante decide cosa fare (scrivere su disco, scartare, segnalare).
    """
    compiles, compile_error = facade.compile_circuit(circuit.source_code)
    if not compiles:
        return {"compiles": False, "compile_error": compile_error, "discarded": True}

    try:
        assigned = _assigned_circuit_count(circuit.source_code)
    except Exception:  # noqa: BLE001 - gia' compilato sopra, ma il conteggio resta best-effort
        assigned = 1
    if assigned != 1:
        return {"compiles": True, "assigned_circuits": assigned, "discarded": True}

    shape = measure_shape(circuit.source_code, facade)
    return {
        "compiles": True,
        "assigned_circuits": 1,
        # Metadato, non criterio di scarto: vedi structural_dead_circuit_check.
        "dead_circuit": structural_dead_circuit_check(circuit.source_code, facade),
        "discarded": False,
        "untouched_qubit_indices": structural_idle_check(circuit.source_code, facade)[
            "idle_qubit_indices"
        ],
        **shape,
    }


def matches_batch_theme(shape: dict, theme: BatchTheme) -> bool:
    """Verifica che la forma MISURATA rientri negli intervalli richiesti dal lotto.

    Sostituisce il vecchio confronto fra intended_smells e smell_focus: quello controllava che
    il generatore avesse dichiarato l'etichetta giusta, cioe' verificava una sua affermazione.
    Questo controlla che il circuito ABBIA la forma richiesta -- un fatto, misurato.
    """
    if shape is None:
        return False

    l_min, l_max = theme.l_range
    c_min, c_max = theme.c_range
    if not (l_min <= shape["l"] <= l_max and c_min <= shape["c"] <= c_max):
        return False

    if theme.idle_target == IdleTarget.PRESENT:
        return shape["idq"] > 0
    return shape["idq"] == 0


def is_near_duplicate(
    circuit: GeneratedCircuit, already_accepted: list[GeneratedCircuit]
) -> str | None:
    """Confronta il circuito con quelli gia' accettati nello stesso lotto.

    Due livelli: (a) codice sorgente identico dopo aver rimosso spazi/a-capo superflui
    (cattura il caso visto nel lotto both_smells, dove 6 record erano byte-identici a
    parte una costante numerica); (b) stessa identica sequenza di NOMI di gate, ignorando
    quali qubit specifici vengono toccati (cattura duplicati "travestiti" con indici diversi).
    Ritorna l'ID del circuito duplicato se trovato, altrimenti None.
    """
    normalized_new = " ".join(circuit.source_code.split())
    gates_new = _gate_sequence(circuit.source_code)

    for other in already_accepted:
        if normalized_new == " ".join(other.source_code.split()):
            return "identical_source"

        if gates_new == _gate_sequence(other.source_code) and len(gates_new) > 2:
            return "same_gate_sequence"

    return None


# Il nome della variabile del circuito non e' vincolato dal prompt (i circuiti reali usano qc,
# qc_bad, circuit, ...), quindi il pattern cattura qualunque ricevente: ancorarlo a "qc." come
# faceva la versione precedente rendeva il confronto cieco su tutti gli altri nomi.
_GATE_CALL_PATTERN = re.compile(r"\b\w+\.(\w+)\(")

# Metodi che non sono gate: comparirebbero nella sequenza senza dire nulla sulla struttura.
_NON_GATE_METHODS = frozenset({"draw", "copy", "bind_parameters", "assign_parameters", "append"})


def _gate_sequence(source_code: str) -> list[str]:
    """Sequenza dei nomi di metodo invocati sul circuito, ignorando gli indici dei qubit."""
    return [
        name for name in _GATE_CALL_PATTERN.findall(source_code) if name not in _NON_GATE_METHODS
    ]
