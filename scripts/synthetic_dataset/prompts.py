"""Few-shot reali e istruzioni di lotto per la generazione sintetica.

Il testo che finisce nel prompt e' in inglese, stessa scelta motivata gia' fatta per
DetectorAgent (vedi detector_agent.py): i modelli coder sono piu' affidabili su prompt tecnici
in inglese.

IL MODELLO SCRIVE SOLO CODICE. E' il terzo e ultimo restringimento del suo compito, e nasce da
una diagnosi sui 55 record del primo giro di generazione (data/interim/synthetic_ground_truth.jsonl).

  - Prima versione: il modello dichiarava l'etichetta (intended_smells) e la motivava. Caduta
    perche' le dichiarazioni erano spesso false e smascherarle era strutturalmente ingrato.
  - Seconda versione: niente etichetta, ma il modello doveva prima DISEGNARE la execution matrix
    bersaglio (matrix_sketch) e contarne l e c. Caduta in modo molto piu' netto:
      * il lotto lc_wide_shallow ha riportato `l` = numero di qubit in tutti e 8 i record
        (6 qubit -> "l=7", 11 qubit -> "l=11"), cioe' il modello contava le RIGHE invece delle
        operazioni dentro una riga;
      * il lotto idq_long_wait ha riportato la stessa riga di conteggi byte-identica in tutti e
        10 i record, mentre i valori misurati andavano da l=5 a l=8 e da IdQ=3 a IdQ=15;
      * il primo lotto e' degenerato in un loop di ripetizione DENTRO matrix_sketch (centinaia
        di '.' consecutivi, poi deriva su testo estraneo), bruciando venti minuti;
      * `theme_consistent` era False su 50 circuiti compilanti su 50, fuori dal lotto pulito.

La conclusione e' che il modello non sa contare `l` e non modella il packing ASAP. Chiedergli
una FORMA metrica era chiedergli l'impossibile, e lo schizzo -- introdotto per aiutarlo -- era
esso stesso la principale fonte di allucinazione, oltre a non essere letto da nessuno a valle.

Quindi ora: un solo campo, `source_code`, e un solo requisito, che giri. La forma non si chiede
piu' in termini di metriche ma in termini STRUTTURALI, che il modello ha dimostrato di saper
seguire anche quando sbagliava i numeri (il lotto idq_long_wait ha prodotto blocchi sequenziali
piu' measure_all finale in tutti e 10 i casi, cioe' esattamente la struttura richiesta). Le
metriche le misura la facade a valle: l'etichetta e' una CONSEGUENZA, non una richiesta.

OTTO few-shot su dieci vengono dai circuiti reali di data/raw/: la base sono i 9 classificati
affidabili da scripts/diagnostics/corpus_reliability_report.py -- eseguibili senza ritocchi,
deterministici, con un solo QuantumCircuit assegnato e senza blocchi compositi opachi. Sono
mostrati come CODICE E BASTA: niente matrice disegnata, niente valori di l/c/IdQ accanto, niente
spiegazione delle metriche. Al modello non servono per il compito che gli resta, e ogni riga in
piu' e' superficie per allucinare.

I DUE RESTANTI (_DENSE_LAYERS, _DEEP_PAIR) li abbiamo COSTRUITI noi, e il commento accanto a
_DENSE_LAYERS spiega perche': coprono la sola forma che ne' il corpus reale ne' il modello
producono, cioe' un circuito grande con tutti i qubit occupati fino alla stessa colonna finale.
Nel prompt la loro provenienza non e' dichiarata -- l'intestazione dei few-shot e' neutra e non
afferma nulla su nessuno dei dieci.

UNA ECCEZIONE alla regola "solo circuiti che girano senza ritocchi" fra i reali, motivata accanto
a _REAL_DEEP_CHAIN: e' l'unico esempio reale della forma profonda-e-stretta.

I lotti non vedono tutti gli stessi esempi: vedi build_batch_prompt.
"""

from dataclasses import dataclass

from pydantic import BaseModel


class GeneratedCircuit(BaseModel):
    """Un circuito generato dall'LLM: solo il sorgente, nient'altro.

    Campo unico e deliberato. Ogni campo aggiuntivo chiesto al modello in passato (l'etichetta,
    il ragionamento, lo schizzo della matrice) e' stato una fonte di errore che non serviva a
    valle: la pipeline misura il codice, non legge quello che il modello dice del codice.
    """

    source_code: str


class GenerationBatch(BaseModel):
    """Output strutturato atteso per un intero lotto di generazione."""

    circuits: list[GeneratedCircuit]


# ---------------------------------------------------------------------------------------------
# Few-shot: circuiti reali, mostrati come solo codice.
#
# Le didascalie descrivono la STRUTTURA (chi lavora, quando, come finisce), mai le metriche:
# nominare l, c o IdQ qui rimetterebbe nel prompt proprio la nozione che il modello non sa
# maneggiare. I nomi delle costanti sono cambiati di conseguenza -- _REAL_BOTH_SMELLS e simili
# descrivevano l'etichetta, che qui non e' piu' un concetto in gioco.
# ---------------------------------------------------------------------------------------------

# data/raw/bugs4q/Terra-0-4000_10_fix.py
_REAL_LAYERED_ENTANGLING = """from qiskit import *
q = QuantumRegister(5,name='q')
c = ClassicalRegister(5, name='c')
qc_bad = QuantumCircuit(q, c)

qc_bad.x(q[4])
for kk in range(5):
    qc_bad.h(q[kk])
qc_bad.barrier(q)
qc_bad.cx(q[2], q[4])
qc_bad.cx(q[3], q[4])

qc_bad.barrier(q)
qc_bad.cx(q[2], q[0])
qc_bad.h(q[2])
qc_bad.h(q[0])
qc_bad.cx(q[2], q[0])
qc_bad.h(q[2])
qc_bad.h(q[0])
qc_bad.cx(q[2], q[0])
"""

# data/raw/thesmellyeight/idq/idq-smelly.py (Listing 6 del paper)
_REAL_SEQUENTIAL_BLOCKS = """from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
qc = QuantumCircuit(qreg_q, creg_c)
qc.h(qreg_q)

qc.p(pi / 2, qreg_q[0])
qc.z(qreg_q[0])
qc.s(qreg_q[0])

qc.barrier()

qc.p(pi / 4, qreg_q[1])
qc.z(qreg_q[1])
qc.s(qreg_q[1])

qc.barrier()
qc.h(qreg_q[2])
qc.p(pi / 8, qreg_q[2])
qc.z(qreg_q[2])
qc.s(qreg_q[2])
qc.measure_all(add_bits=False)
"""

# data/raw/bugs4q/StackExchange_16_fix.py
_REAL_DISJOINT_SUBSETS = """from qiskit import *
q = QuantumRegister(5)
c = ClassicalRegister(5)
qc = QuantumCircuit(q,c)

qc.ccx(q[0],q[1],q[3])
qc.ccx(q[2],q[3],q[4])
qc.ccx(q[0],q[1],q[3])
"""

# data/raw/bugs4q/Terra-0-4000_6_Fixed.py, SEZIONE DI COSTRUZIONE citata verbatim.
#
# UNICA ECCEZIONE alla regola "solo circuiti che girano senza ritocchi", e vale la pena
# motivarla. Quella regola nasce per il DATO: un circuito che entra nel dataset dev'essere
# esattamente quello che il suo autore ha scritto. Un ESEMPIO nel prompt non entra nel dataset e
# non viene mai etichettato: il suo unico compito e' mostrare una struttura.
#
# Il file originale non gira per due righe che NON toccano la costruzione del circuito: un
# `from qiskit.providers.aer import QasmSimulator` (modulo rimosso) in cima e una chiamata a
# `execute(...)` (rimossa in Qiskit 1.0) in coda. Qui e' citata la porzione fra le due, verbatim.
#
# E' l'unico esempio reale della forma profonda-e-stretta: nessun circuito misurabile del corpus
# costruisce una catena lunga su un solo qubit. Il suo loop e' pero' da MILLE iterazioni, e nel
# primo giro di generazione il modello ha tentato di copiarne la magnitudine: l'istruzione del
# lotto deep_chain indica quindi un numero di iterazioni esplicito e molto piu' piccolo.
_REAL_DEEP_CHAIN = """from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister

q = QuantumRegister(16)
c = ClassicalRegister(16)

qc = QuantumCircuit(q, c)

def pad_QId(circuit,N,q):
    for ii in range(N):
        circuit.barrier(q)
        circuit.id(q)
    return circuit

qc.x(q[0])
qc = pad_QId(qc, 1000, q[0])
qc.measure(q[0], c[0])
"""

# data/raw/thesmellyeight/idq/idq-fixed.py
_REAL_PER_QUBIT_MEASURE = """from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
qc = QuantumCircuit(qreg_q, creg_c)

qc.h(qreg_q[0])
qc.p(pi / 2, qreg_q[0])
qc.z(qreg_q[0])
qc.s(qreg_q[0])
qc.measure(qreg_q[0], creg_c[0])
qc.barrier()
qc.h(qreg_q[1])
qc.p(pi / 4, qreg_q[1])
qc.z(qreg_q[1])
qc.s(qreg_q[1])
qc.measure(qreg_q[1], creg_c[1])
qc.barrier()
qc.h(qreg_q[2])
qc.p(pi / 8, qreg_q[2])
qc.z(qreg_q[2])
qc.s(qreg_q[2])
qc.measure(qreg_q[2], creg_c[2])
"""

# data/raw/bugs4q/Terra-0-4000_3_Fixed.py
_REAL_COMPACT = """from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.cx(0, 1, ctrl_state='0')
qc.ccx(0, 1, 2, ctrl_state='00')
"""

# data/raw/thesmellyeight/lc/lc-smelly.py (Listing 4 del paper)
_REAL_TINY = """from qiskit import QuantumCircuit

qc = QuantumCircuit(1)

qc.h(0)
qc.z(0)
qc.h(0)
"""

# data/raw/bugs4q/Terra-11000-15000_12801_Fixed_fixed_version.py
_REAL_PARAMETRIC = """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

theta1 = Parameter('theta1')

qc = QuantumCircuit(5)
qc.ry(theta1, 4)
"""

# ---------------------------------------------------------------------------------------------
# I DUE ESEMPI COSTRUITI DA NOI. Non vengono da data/raw/: li abbiamo scritti noi, e questa e' la
# sola coppia del modulo che non sia un circuito reale.
#
# PERCHE' ESISTONO. Il corpus reale non contiene un solo circuito con questa forma -- grande ma
# con TUTTI i qubit occupati fino alla stessa colonna finale -- e nemmeno il modello la produce
# spontaneamente: nel giro di generazione precedente 16 circuiti su 16 sopra la soglia portavano
# anche attesa, senza eccezioni. Il meccanismo e' identificato: ogni circuito grande chiudeva con
# measure_all(), che crea una colonna finale condivisa da tutti i qubit, e chiunque avesse finito
# prima restava fermo ad aspettarla. Produrre questa forma richiede di ragionare sul packing ASAP,
# che il modello non sa fare; mostrargliela gia' fatta si'.
#
# I valori accanto sono MISURATI con QiskitFacade.calculate_smell_metrics, non stimati, e sono
# fissati da un test (test_the_two_constructed_examples_have_the_shape_they_are_there_for): se
# qualcuno li ritocca e perdono la forma, l'unica ragione per cui esistono sparisce in silenzio.
#
# Nel PROMPT non compare nulla sulla loro provenienza: l'intestazione dei few-shot e' neutra, non
# dichiara ne' "reali" ne' "costruiti". Questi commenti non entrano mai nel prompt.

# Misurato: l=5, c=5, l*c=25, IdQ=0, non inerte. Ogni qubit riceve un'operazione in OGNI colonna,
# e le measure finali sono per-qubit: nessuno resta indietro ad aspettare una colonna condivisa.
_DENSE_LAYERS = """from qiskit import QuantumCircuit
from numpy import pi
qc = QuantumCircuit(5, 5)
for layer in range(2):
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.h(4)
    for q in range(5):
        qc.p(pi / 4, q)
for q in range(5):
    qc.measure(q, q)
"""

# Misurato: l=11, c=2, l*c=22, IdQ=0, non inerte. La forma opposta: due soli qubit, catena lunga,
# ma ENTRAMBI lavorano a ogni iterazione -- e' quello che tiene l'attesa a zero.
_DEEP_PAIR = """from qiskit import QuantumCircuit
from numpy import pi
qc = QuantumCircuit(2, 2)
for i in range(5):
    qc.h(0)
    qc.rz(pi / 3, 1)
    qc.cx(0, 1)
qc.measure(0, 0)
qc.measure(1, 1)
"""


# Chiave -> (didascalia strutturale, codice). La didascalia dice cosa GUARDARE nell'esempio, in
# termini di struttura del codice: nessun numero, nessuna metrica, nessun giudizio.
_EXAMPLES: dict[str, tuple[str, str]] = {
    "dense_layers": (
        "every qubit receives a gate in every single layer, and each one is measured at the end "
        "with no qubit ever pausing in between",
        _DENSE_LAYERS,
    ),
    "deep_pair": (
        "a long loop over just two qubits, with BOTH of them worked on every iteration",
        _DEEP_PAIR,
    ),
    "layered_entangling": (
        "layers across all qubits, then entangling gates between a few of them",
        _REAL_LAYERED_ENTANGLING,
    ),
    "sequential_blocks": (
        "one qubit worked at a time, in its own block, with a shared measure_all at the end",
        _REAL_SEQUENTIAL_BLOCKS,
    ),
    "disjoint_subsets": (
        "multi-qubit gates acting on different subsets of the qubits, one after another",
        _REAL_DISJOINT_SUBSETS,
    ),
    "deep_chain": (
        "a LOOP appending gates to a single qubit -- note the loop, ignore its size",
        _REAL_DEEP_CHAIN,
    ),
    "per_qubit_measure": (
        "the same work as the sequential example, but each qubit is measured right after its own "
        "block instead of all together at the end",
        _REAL_PER_QUBIT_MEASURE,
    ),
    "compact": (
        "a couple of multi-qubit gates over three qubits, with open controls",
        _REAL_COMPACT,
    ),
    "tiny": (
        "three gates on one qubit and nothing else",
        _REAL_TINY,
    ),
    "parametric": (
        "an unbound Parameter used as a rotation angle",
        _REAL_PARAMETRIC,
    ),
}


# Il preambolo obbligatorio sostituisce due regole della versione precedente che erano formulate
# come DIVIETI ("un circuito costruito senza bit classici crasha", "ogni import dev'essere
# presente") e che il modello ha violato 14 volte su 48. La diagnosi e' che una regola negativa
# letta a meta' prompt perde contro un'istruzione positiva letta alla fine: un lotto intero (8 su
# 8) e' morto proprio sulla regola dei bit classici, perche' la sua istruzione chiedeva una
# measure per qubit e il modello ha raggiunto la forma sbagliata. Qui la stessa informazione e'
# data come TEMPLATE DA COPIARE, che e' cio' che un modello sa seguire.
#
# Imporre un'unica forma di costruzione elimina per costruzione entrambi i guasti: i bit classici
# ci sono sempre, e pi e' sempre importato.
_MANDATORY_PREAMBLE = """MANDATORY PREAMBLE -- every snippet begins with exactly these three lines, in this order:

    from qiskit import QuantumCircuit
    from numpy import pi
    qc = QuantumCircuit(<n>, <n>)

Replace <n> with the number of qubits you want, and pass it TWICE. The second <n> creates the classical bits that `qc.measure(i, i)` writes into: a circuit built as `QuantumCircuit(n)` has none, and every measure on it crashes with 'Index 0 out of range for size 0'.
Use no other import, and do not use QuantumRegister or ClassicalRegister -- `qc` is the only name you need. Always call the circuit `qc`."""


_HARD_RULES = """HARD RULES (a snippet that breaks one is discarded)

1. Assign EXACTLY ONE QuantumCircuit in the snippet. Helper sub-circuits assigned to their own variable are forbidden: the pipeline measures the last circuit assigned, so a helper would be measured instead of yours.
2. Every two-qubit or three-qubit gate needs DIFFERENT indices. `qc.cx(0, i)` inside a loop starting at i = 0 crashes on the first iteration with 'duplicate bit arguments'. Start such loops at 1, or skip the coinciding index.
3. Use only these gate calls: h, x, y, z, s, sdg, t, tdg, p, rx, ry, rz, u, sx, cx, cy, cz, ch, cp, crx, cry, crz, swap, ccx, cswap, id, barrier, measure. No composite or library blocks, no `.to_gate()`, no `.control()`, no `.compose()` of another circuit. These were REMOVED from Qiskit and crash: `execute(...)`, `qiskit.Aer`, `qiskit.providers.aer`, `qc.c_if(...)`, `qc.bind_parameters(...)`.
4. No randomness: no `random_circuit`, no unseeded numpy random. The snippet must build the identical circuit on every run.
5. The snippet must be COMPLETE and RUNNABLE: no undefined name, no placeholder comment standing in for code.
6. Loops are welcome: `for q in range(n): qc.h(q)` is how real code is written."""


# La checklist chiude il prompt, DOPO l'istruzione del lotto. E' deliberato: la causa comune di
# tre guasti su cinque e' che l'ultima cosa letta pesa di piu' della regola letta prima, quindi
# qui lo stesso effetto viene sfruttato a favore invece di subirlo. Le quattro voci sono i quattro
# guasti misurati, non prudenza generica.
_FINAL_CHECKLIST = """BEFORE YOU ANSWER, check every snippet one by one:
- does it start with the three preamble lines, with the qubit count passed TWICE?
- does every two- or three-qubit gate get DIFFERENT indices?
- does every qubit you allocated receive at least one gate?
- does at least one qubit get an `h` or an `x` BEFORE any controlled gate? A circuit made only of controlled gates starting from all-zero does nothing at all, because no control ever fires."""


_OUTPUT_FORMAT_INSTRUCTIONS = """OUTPUT FORMAT
Respond with a structured object containing a single field "circuits": a list of exactly \
{count} circuit objects. Each circuit object has exactly ONE field, `source_code`: the complete, \
runnable Qiskit snippet, as plain Python source with no markdown fences around it.
Do not include any text outside the structured object. Do not add comments explaining the \
circuit, do not describe it, do not classify it, do not count anything about it."""


@dataclass(frozen=True)
class BatchTheme:
    """Un lotto di generazione, definito da una STRUTTURA di codice.

    Cosa e' sparito rispetto alla versione precedente, e perche': l_range, c_range e idle_target
    descrivevano un bersaglio METRICO che il modello ha mancato su 50 circuiti compilanti su 50
    (vedi il docstring del modulo). Chiederlo non produceva circuiti a bersaglio, produceva
    scarti: verification.matches_batch_theme li bocciava tutti e il lotto restava vuoto. Restano
    il numero di qubit -- l'unico vincolo numerico che il modello ha effettivamente rispettato --
    e un'istruzione che descrive la struttura in termini di codice.

    example_keys sceglie quali few-shot il lotto vede, in ordine: in un prompt lungo la posizione
    pesa, e l'esempio piu' vicino alla struttura richiesta non deve finire in fondo.
    """

    theme: str
    qubit_range: tuple[int, int]
    instruction: str
    example_keys: tuple[str, ...]
    count: int


# Sei lotti, 48 circuiti. Non c'e' piu' una previsione di etichetta per lotto -- l'etichetta si
# misura a valle e basta -- ma le strutture sono scelte perche' lo SPAZIO delle forme risulti
# coperto: catene profonde e strati larghi tendono a circuiti grandi, blocchi sequenziali e gate
# su sottoinsiemi disgiunti tendono a produrre attese, i circuiti piccoli e compatti tendono a
# restare puliti. "Tendono": nessuna di queste e' una garanzia, ed e' esattamente il motivo per
# cui la misura resta a valle.
#
# count = 8 uniforme e piu' basso di prima (era fino a 12). Due ragioni misurate sul primo giro:
# il lotto da 12 ha prodotto 11 duplicati identici su 12, e la lunghezza dell'output e' il
# fattore che ha innescato la degenerazione. Lotti corti costano una chiamata in piu' e riducono
# entrambi i rischi.
BATCH_THEMES: list[BatchTheme] = [
    BatchTheme(
        theme="deep_chain",
        qubit_range=(2, 3),
        instruction=(
            "Write circuits in which ONE qubit receives a long chain of gates, applied by a "
            "loop of 8 to 12 iterations. Inside the loop body, apply EXACTLY ONE gate to that "
            "chain qubit -- not two, not three -- plus one gate touching the other qubits, so "
            "that none of them is left allocated and untouched. Vary which gate goes into the "
            "chain across iterations instead of repeating a single one. Close with "
            "`for q in range(n): qc.measure(q, q)` and never with `qc.measure_all()`."
        ),
        example_keys=("deep_pair", "deep_chain", "per_qubit_measure"),
        count=8,
    ),
    BatchTheme(
        theme="wide_layers",
        qubit_range=(4, 7),
        instruction=(
            "Write circuits made of 3 to 5 successive LAYERS, where in every single layer EVERY "
            "qubit receives exactly one gate. Build each layer by pairing up the qubits with an "
            "entangling gate (cx, cz, swap) on different pairs each time, and giving a "
            "single-qubit gate to any qubit left over -- then a full round of single-qubit "
            "gates across all of them. The point of this batch is that NO qubit ever sits out: "
            "every qubit works in every layer, from the first to the last, and none stops and "
            "comes back later. Close with `for q in range(n): qc.measure(q, q)`. Never use "
            "`qc.measure_all()` here: it makes every qubit that finished earlier wait for one "
            "shared final step, which is exactly what this batch must avoid. The first two "
            "reference circuits show this shape in its wide and its narrow form."
        ),
        example_keys=("dense_layers", "deep_pair", "layered_entangling"),
        count=8,
    ),
    BatchTheme(
        theme="sequential_blocks",
        qubit_range=(3, 5),
        instruction=(
            "Write circuits where the qubits are worked ONE AT A TIME: the first qubit gets a "
            "block of 3 to 5 gates, then the second gets its own block, then the third, and so "
            "on, so that while one qubit is being worked the others receive nothing. Close the "
            "circuit with a single shared `qc.measure_all(add_bits=False)` at the very end, "
            "after all the blocks. Vary the gates used in each block and how many qubits you "
            "allocate."
        ),
        example_keys=("sequential_blocks", "per_qubit_measure", "layered_entangling"),
        count=8,
    ),
    BatchTheme(
        theme="disjoint_pauses",
        qubit_range=(4, 5),
        instruction=(
            "Open with an `h` or an `x` on two or three of the qubits, so that the controlled "
            "gates that follow actually fire. Then write the body out of multi-qubit gates -- "
            "cx, cz, ccx, swap -- each acting on a DIFFERENT subset of the qubits, so that "
            "between two gates touching the same qubit there are gates that do not touch it at "
            "all. Keep the whole snippet under about twelve gate calls, use no barriers, and "
            "leave no allocated qubit untouched. This batch is about the ORDER of the gates, "
            "not the size: the circuits must stay small."
        ),
        example_keys=("disjoint_subsets", "compact", "layered_entangling"),
        count=8,
    ),
    BatchTheme(
        theme="large_mixed",
        qubit_range=(5, 8),
        instruction=(
            "Write LARGER circuits, 5 to 8 qubits and 20 to 35 gate calls, in which the qubits "
            "are NOT all treated the same: some of them work from start to finish, while others "
            "receive a few gates early, then nothing for a good stretch while the rest of the "
            "circuit works, then come back later. Mix single-qubit gates and entangling gates, "
            "and close with `qc.measure_all(add_bits=False)`."
        ),
        example_keys=("layered_entangling", "sequential_blocks", "deep_chain"),
        count=8,
    ),
    BatchTheme(
        theme="small_compact",
        qubit_range=(1, 4),
        instruction=(
            "Write SMALL circuits: 1 to 4 qubits and at most 8 gate calls. Every qubit must be "
            "busy in consecutive steps, with no stretch where it receives nothing while other "
            "qubits are being worked, and each qubit is measured right after its own work "
            "rather than all together at the end. Vary the qubit count across the batch and "
            "make the circuits look like purposeful quantum code rather than filler -- "
            "superposition, entanglement, phase kickback, small rotations."
        ),
        example_keys=("compact", "tiny", "parametric", "per_qubit_measure"),
        count=8,
    ),
]


def _examples_block(theme: BatchTheme) -> str:
    """I few-shot del lotto, in ordine, come solo codice con una didascalia strutturale."""
    blocks = []
    for position, key in enumerate(theme.example_keys, start=1):
        caption, code = _EXAMPLES[key]
        blocks.append(f"""Example {position} -- {caption}:
```python
{code}```""")
    return "\n\n".join(blocks)


def build_batch_prompt(theme: BatchTheme) -> str:
    """Assembla il prompt completo (preambolo + vincoli + few-shot + struttura + checklist).

    I few-shot NON sono gli stessi per tutti i lotti: sono dieci circuiti con strutture molto
    diverse, e mostrarli tutti a ogni lotto significherebbe annegare quello pertinente. Ogni lotto
    vede i tre o quattro piu' vicini alla struttura che deve produrre, nell'ordine dichiarato in
    example_keys.

    L'ORDINE DELLE SEZIONI E' PARTE DEL PROGETTO. L'istruzione del lotto sta in fondo e la
    checklist ancora piu' in fondo: nel giro precedente un lotto intero e' morto perche' la sua
    istruzione (letta per ultima) chiedeva una measure per qubit e ha prevalso sulla regola dei
    bit classici (letta a meta' prompt). La posizione pesa, quindi cio' che deve prevalere sta
    alla fine.

    L'intestazione dei few-shot non dichiara la provenienza dei circuiti: otto vengono da
    data/raw/, due li abbiamo costruiti noi (vedi il commento accanto a _DENSE_LAYERS). Al modello
    la distinzione non serve -- guarda la struttura, non l'origine -- e cosi' il prompt non
    afferma nulla di falso.
    """
    minimum_qubits, maximum_qubits = theme.qubit_range

    return f"""You are writing small Qiskit programs for a research dataset. Your ONLY job is to \
produce Python snippets that RUN. Nothing else is asked of you: do not analyse the circuits, do \
not classify them, do not count anything about them, do not explain them. Working code, and that \
is all.

{_MANDATORY_PREAMBLE}

{_HARD_RULES}

REFERENCE CIRCUITS -- look at how these are STRUCTURED; you are not asked to reproduce them.

{_examples_block(theme)}

WHAT TO WRITE IN THIS BATCH
{theme.instruction}

Use between {minimum_qubits} and {maximum_qubits} qubits, varying the number across the batch.

Generate exactly {theme.count} circuits. Every circuit in this batch must be STRUCTURALLY \
DISTINCT from every other circuit in the same batch: do not repeat the same gate sequence with \
only cosmetic changes (different qubit indices, different variable names, a different subset of \
qubits touched by an otherwise identical block). Vary the gates, their order, the number of \
qubits, and the way the circuit is put together. Do not copy the reference circuits above.

{_FINAL_CHECKLIST}

{_OUTPUT_FORMAT_INSTRUCTIONS.format(count=theme.count)}"""
