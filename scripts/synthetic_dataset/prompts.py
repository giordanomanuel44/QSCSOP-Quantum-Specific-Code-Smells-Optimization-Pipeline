"""Definizioni QSMELL, few-shot reali e istruzioni di lotto per la generazione sintetica.

Il testo che finisce nel prompt e' in inglese, stessa scelta motivata gia' fatta per
DetectorAgent (vedi detector_agent.py): i modelli coder sono piu' affidabili su prompt tecnici
in inglese.

DUE CAMBIAMENTI DI FONDO rispetto alla versione precedente di questo modulo.

1. Le definizioni non sono piu' SEMANTICHE ma METRICHE. Prima Long Circuit voleva dire
   "sequenza di gate riducibile" e Idle Qubits "qubit il cui effetto netto e' l'identita'":
   nozioni entrambe estranee a QSMELL (Chen et al., ICSE 2023, DOI 10.1109/ICSE48619.2023.00041),
   la fonte su cui la tesi dichiara di basarsi. Ora si genera verso una FORMA misurabile della
   execution matrix -- l (operazioni sul qubit piu' carico), c (operazioni in parallelo), IdQ
   (colonne di attesa) -- calcolata da QiskitFacade.calculate_smell_metrics. Vedi
   docs/qsmell_definizioni_e_esempi_prompt.md.

2. Il modello NON DICHIARA PIU' NULLA. Sparisce intended_smells, e con esso reasoning,
   line_by_line_expansion, qubit_operation_analysis e simplified_source_code: erano tutta
   impalcatura costruita per rendere verificabile una dichiarazione del generatore, e la
   dichiarazione era il punto debole (in revisione manuale si trovavano reasoning che
   descrivevano un circuito diverso da quello scritto sotto). L'etichetta ora si MISURA a valle,
   non si chiede: il generatore puo' sbagliare la forma, non puo' mentire sull'etichetta.

I few-shot vengono dai circuiti reali di data/raw/, non da esempi costruiti per l'occasione, e
ogni valore riportato nei commenti e' misurato con la facade, non stimato. La base sono i 9
circuiti classificati affidabili da scripts/diagnostics/corpus_reliability_report.py --
eseguibili senza ritocchi, deterministici, con un solo QuantumCircuit assegnato e senza blocchi
compositi opachi.

UNA SOLA ECCEZIONE, motivata per esteso accanto a _REAL_LC_DEEP_NARROW: quei 9 circuiti non
contengono un Long Circuit PURO (l'unico LC del corpus affidabile porta anche Idle Qubits) ne'
alcun circuito profondo-e-stretto, cioe' proprio la forma che tre lotti su sette devono produrre.
L'esempio che copre quel buco e' la sezione di costruzione di Terra-0-4000_6_Fixed.py citata
verbatim -- un file che non gira per un import e una chiamata obsoleti, entrambi ESTERNI alla
costruzione del circuito. La distinzione e' quella fra dato ed esempio: il criterio "gira senza
ritocchi" protegge il DATO dall'inquinamento, mentre a un esempio nel prompt (che non entra nel
dataset e non viene mai etichettato) si chiede solo che il codice mostrato produca davvero i
numeri dichiarati accanto -- e qui e' verificato.

I lotti non vedono tutti gli stessi esempi: vedi build_batch_prompt.
"""

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from qscsop_pipeline.qscsop.mas.detection_thresholds import LC_PRODUCT_CUTOFF, is_long_circuit


class GeneratedCircuit(BaseModel):
    """Un singolo circuito generato dall'LLM, preceduto dallo schizzo della sua matrice.

    L'ordine dei campi conta: in generazione autoregressiva ogni campo e' condizionato solo su
    quelli che lo precedono, quindi l'ordine qui sotto e' l'ordine di ragionamento imposto al
    modello. Qui il PIANO precede l'artefatto -- l'opposto della versione precedente, dove
    source_code veniva per primo per necessita' (i campi di analisi dovevano riferirsi a un
    codice gia' scritto) e l'analisi finiva spesso a razionalizzare a posteriori codice
    sbagliato.
    """

    # 1. matrix_sketch e' PRIMO: la forma target va decisa e contata PRIMA di scrivere il codice.
    #    Il modello disegna righe = qubit e colonne = timestamp, poi conta l e c dallo schizzo.
    #    Scrivere il codice per primo significherebbe misurarne la forma dopo averla subita.
    matrix_sketch: str
    # 2. source_code REALIZZA lo schizzo appena disegnato. Nessun campo lo segue: non c'e' piu'
    #    nulla da dichiarare, l'etichetta si misura con la facade.
    source_code: str


class GenerationBatch(BaseModel):
    """Output strutturato atteso per un intero lotto di generazione."""

    circuits: list[GeneratedCircuit]


# ---------------------------------------------------------------------------------------------
# Definizioni degli smell, in forma metrica.
# ---------------------------------------------------------------------------------------------

_EXECUTION_MATRIX = """THE EXECUTION MATRIX

Both smells are measured on one structure: a matrix whose ROWS are qubits and whose COLUMNS are \
timestamps. It is built like this:

1. Each instruction is placed at the first level that is free on ALL the bits it touches, and \
those bits all advance to that level (as-soon-as-possible, left-justified packing).
2. Classical bits take part in the level computation but are not rows: two measures sharing a \
classical register do NOT end up in the same column.
3. Barriers OCCUPY a level and synchronise, but do NOT count as operations."""

_SMELL_DEFINITIONS = f"""SMELL DEFINITIONS (measured on the execution matrix, not judged by eye)

1. LONG CIRCUIT (LC)
Long Circuit is about accumulated hardware error, not about redundancy. A circuit is Long when \
it is deep and/or wide enough that the probability of running it without any gate error becomes \
too low. It says nothing about whether the circuit could be rewritten more concisely: a circuit \
with no redundant gate at all is still Long if it is big enough, and a circuit full of \
cancelling gates is not Long if it is small.

  l = the largest number of operations on a single ROW (one qubit), barriers excluded
  c = the largest number of operations in a single COLUMN (in parallel), barriers excluded
  LC metric = l * c, and the circuit is LONG CIRCUIT when l * c >= {LC_PRODUCT_CUTOFF}

2. IDLE QUBITS (IdQ)
Idle Qubits is about decoherence while waiting. A qubit is idle when it is used, then left \
untouched while the rest of the circuit works on other qubits, then used again: it loses quantum \
information during the wait. It is NOT about qubits that do nothing useful, and it is NOT about \
gates that cancel each other out.

For each row, left to right: skip every barrier column; start counting only AFTER that qubit's \
first operation; each empty column between two operations increments the counter; at the next \
operation the counter is compared against the maximum and reset; empty columns AFTER a qubit's \
LAST operation do not count.

  IdQ metric = the maximum over all qubits, and the circuit is IDLE QUBITS when IdQ > 0

THREE CONSEQUENCES THAT ARE EASY TO GET WRONG
- A qubit that is NEVER used contributes IdQ = 0. It has no two uses to wait between. Allocating \
qubits you never touch is waste, but it is not this smell.
- Gates that cancel each other out (two consecutive H) do NOT make a qubit idle. Such a qubit is \
BUSY -- it has operations in its row, which is all the metric looks at.
- Barriers do not create idling: the metric skips them. Waiting comes from DEPENDENCIES -- \
multi-qubit gates acting on disjoint subsets, or a section of the circuit working elsewhere."""

# ---------------------------------------------------------------------------------------------
# Few-shot: i 9 circuiti reali affidabili. Ogni valore e' misurato con
# QiskitFacade.calculate_smell_metrics, ogni matrice e' prodotta da _execution_matrix.
# ---------------------------------------------------------------------------------------------

# data/raw/bugs4q/Terra-0-4000_10_fix.py -- misurato l=7, c=5, l*c=35, IdQ=3 (q0).
# L'UNICO Long Circuit del corpus affidabile, e porta entrambi gli smell insieme.
_REAL_BOTH_SMELLS = """from qiskit import *
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

_REAL_BOTH_SMELLS_MATRIX = """  q0:   h    .   barr   .    .   barr  cx    h    cx    h    cx
  q1:   h    .   barr   .    .   barr   .    .     .    .     .
  q2:   h    .   barr  cx    .   barr  cx    h    cx    h    cx
  q3:   h    .   barr   .   cx   barr   .    .     .    .     .
  q4:   x    h   barr  cx   cx   barr   .    .     .    .     ."""

_REAL_BOTH_SMELLS_EXPLANATION = """l = 7 (rows q0 and q2, which work all the way to the end), \
c = 5 (the first column, where all five qubits get a gate) -> l * c = 35, LONG CIRCUIT.
IdQ = 3 on q0: after its h in column 1, three columns work only on q2/q3/q4 before q0 comes \
back at the first cx. Note that barrier columns are skipped and do not count towards the 3.
Both smells at once, produced by different parts of the same circuit: LC by the overall size, \
IdQ by the dependency structure."""

# data/raw/thesmellyeight/idq/idq-smelly.py -- misurato l=6, c=3, l*c=18, IdQ=7 (q0).
# Listing 6 del paper. IdQ puro a DUE unita' dalla soglia LC: l'esempio migliore del confine.
_REAL_IDQ_LONG_WAIT = """from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
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

_REAL_IDQ_LONG_WAIT_MATRIX = """  q0:   h    p    z    s   barr   .    .    .   barr   .    .    .    .   barr  meas
  q1:   h    .    .    .   barr   p    z    s   barr   .    .    .    .   barr  meas
  q2:   h    .    .    .   barr   .    .    .   barr   h    p    z    s   barr  meas"""

_REAL_IDQ_LONG_WAIT_EXPLANATION = """l = 6 (q2: h, h, p, z, s, measure), c = 3 (the first \
column, where the register-wide qc.h(qreg_q) touches all three qubits) -> l * c = 18, which is \
BELOW 20: this circuit is NOT Long Circuit. It is pure IDLE QUBITS.
IdQ = 7 on q0: q0 finishes its work at column 4, then seven empty columns follow in its row \
(three while q1 works, four while q2 works) before the final measure_all brings it back.
Read the two counterintuitive points against this circuit. q2 receives TWO h gates -- the \
broadcast one and its own -- which cancel algebraically, yet q2 is the BUSIEST row here and is \
not the idle qubit. The idle qubit is q0, which has exactly one h. And the barriers are not the \
cause of the waiting: the metric skips them; the waiting comes from the sequential structure.
The trailing structure matters too: a final measure_all is a reliable IdQ generator, because \
every qubit that finishes early waits for that shared column."""

# data/raw/bugs4q/StackExchange_16_fix.py -- misurato l=3, c=3, l*c=9, IdQ=1 (q0).
_REAL_IDQ_SHORT_WAIT = """from qiskit import *
q = QuantumRegister(5)
c = ClassicalRegister(5)
qc = QuantumCircuit(q,c)

qc.ccx(q[0],q[1],q[3])
qc.ccx(q[2],q[3],q[4])
qc.ccx(q[0],q[1],q[3])
"""

_REAL_IDQ_SHORT_WAIT_MATRIX = """  q0:  ccx    .   ccx
  q1:  ccx    .   ccx
  q2:    .  ccx     .
  q3:  ccx  ccx   ccx
  q4:    .  ccx     ."""

_REAL_IDQ_SHORT_WAIT_EXPLANATION = """l = 3 (q3, the only qubit in all three gates), c = 3 \
(every column holds one ccx, which occupies three rows) -> l * c = 9, no Long Circuit.
IdQ = 1 on q0: the middle ccx acts on a DISJOINT subset (q2, q3, q4), so q0 and q1 sit still \
for exactly one column and then resume.
This is the cheapest possible way to produce Idle Qubits: no barrier, no redundancy, no wasted \
qubit -- just a multi-qubit gate on a disjoint subset in the middle. Note also that q2 and q4 \
are NOT idle: their first operation is in column 2, and columns before a qubit's first \
operation do not count."""

# data/raw/bugs4q/Terra-0-4000_6_Fixed.py, SEZIONE DI COSTRUZIONE citata verbatim.
# Misurato sul testo esattamente com'e' riportato qui sotto: l=1002, c=1, l*c=1002, IdQ=0.
#
# UNICA ECCEZIONE alla regola "solo circuiti che girano senza ritocchi", e vale la pena
# motivarla. Quella regola nasce per il DATO: un circuito che entra nel dataset dev'essere
# esattamente quello che il suo autore ha scritto, altrimenti l'etichetta misura qualcosa che
# nessuno ha mai scritto. Un ESEMPIO nel prompt non entra nel dataset e non viene mai etichettato:
# il suo unico compito e' mostrare una forma, quindi il requisito e' un altro, cioe' che il codice
# mostrato produca davvero i numeri dichiarati accanto.
#
# Il file originale non gira per due righe che NON toccano la costruzione del circuito: un
# `from qiskit.providers.aer import QasmSimulator` (modulo rimosso da Qiskit) in cima, e una
# chiamata a `execute(...)` (rimossa in Qiskit 1.0) in coda. Fra le due, ogni riga e' Qiskit
# valido e moderno. Qui e' citata quella porzione, verbatim: nessuna riga riscritta, nessun
# valore cambiato, solo import morto e coda morta non riportati -- e le metriche sono misurate
# su questo testo, non sul file intero.
#
# Copre il buco piu' grosso del corpus affidabile: e' l'unico LONG CIRCUIT PURO disponibile (gli
# altri 8 circuiti affidabili sono 5 puliti, 2 IdQ puri e 1 con entrambi gli smell), ed e' anche
# l'unico esempio della forma profonda-e-stretta, che nessun circuito misurabile del repo
# realizza.
_REAL_LC_DEEP_NARROW = """from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister

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

_REAL_LC_DEEP_NARROW_MATRIX = """  q0:   x   barr  id   barr  id   barr  id   ... (1000 identities) ...  meas
  q1:   .    .    .    .    .    .    .    ...                     ...    .
  ...
  q15:  .    .    .    .    .    .    .    ...                     ...    ."""

_REAL_LC_DEEP_NARROW_EXPLANATION = """l = 1002 (q0: one x, then 1000 id gates, then the measure), \
c = 1 (nothing ever runs in parallel) -> l * c = 1002, LONG CIRCUIT. IdQ = 0.
This is the OPPOSITE extreme from Example 1, and it is pure Long Circuit: no idling anywhere.
Two things make it work, and both are worth copying:
- The depth comes from a LOOP appending to ONE qubit. That is how real code produces a long \
chain; writing a thousand lines by hand is not.
- The barriers do not break q0's row: the metric skips barrier columns entirely, so q0's \
operations stay contiguous and IdQ stays 0 even though a barrier sits between every pair.
And note what does NOT happen: q1 to q15 are allocated and never touched, and they contribute \
nothing -- not to l, not to c, not to IdQ. Fifteen wasted qubits, metric zero.
MAGNITUDE WARNING: this real circuit reaches l = 1002, far beyond anything your batch asks for. \
Copy the MECHANISM (a loop building depth on one qubit), not the size -- generate the l your \
batch instruction specifies."""

# data/raw/thesmellyeight/idq/idq-fixed.py -- misurato l=5, c=1, l*c=5, IdQ=0.
# Il fix canonico del paper per l'esempio sopra: STESSO numero di operazioni, riordinate.
_REAL_CLEAN_SEQUENTIAL = """from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
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

_REAL_CLEAN_SEQUENTIAL_MATRIX = """  q0:   h    p    z    s   meas  barr   .    .    .    .    .   barr   .    .    .    .    .
  q1:   .    .    .    .     .   barr   h    p    z    s   meas barr   .    .    .    .    .
  q2:   .    .    .    .     .   barr   .    .    .    .    .   barr   h    p    z    s   meas"""

_REAL_CLEAN_SEQUENTIAL_EXPLANATION = """l = 5, c = 1 -> l * c = 5, and IdQ = 0. CLEAN.
This is the SAME 15 operations as the idle-qubits example above, only reordered: nothing was \
removed. Each qubit now gets one contiguous block ending in its own measure, and then leaves the \
stage. The emptiness before a qubit's first operation and after its last one does not count, so \
all that emptiness is free.
Use this pattern when you need a clean circuit that is not trivially small: per-qubit measures \
instead of a final measure_all."""

# data/raw/bugs4q/Terra-0-4000_3_Fixed.py -- misurato l=2, c=3, l*c=6, IdQ=0.
_REAL_CLEAN_COMPACT = """from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.cx(0, 1, ctrl_state='0')
qc.ccx(0, 1, 2, ctrl_state='00')
"""

_REAL_CLEAN_COMPACT_MATRIX = """  q0:  cx_o0  ccx_o00
  q1:  cx_o0  ccx_o00
  q2:     .   ccx_o00"""

_REAL_CLEAN_COMPACT_EXPLANATION = """l = 2, c = 3 -> l * c = 6, IdQ = 0. CLEAN.
The other way to keep a circuit clean: keep every qubit busy in consecutive columns. q2 starts \
late, but a late start is free -- only gaps BETWEEN two operations of the same qubit count."""

# data/raw/thesmellyeight/lc/lc-smelly.py -- misurato l=3, c=1, l*c=3, IdQ=0.
# Listing 4 del paper: l'esempio con cui il paper ILLUSTRA Long Circuit, che con la metrica del
# paper stesso non e' Long Circuit. Nel prompt vale come contro-esempio esplicito.
_REAL_COUNTEREXAMPLE_REDUNDANT = """from qiskit import QuantumCircuit

qc = QuantumCircuit(1)

qc.h(0)
qc.z(0)
qc.h(0)
"""

_REAL_COUNTEREXAMPLE_REDUNDANT_EXPLANATION = """Matrix: one row, three columns -- h, z, h.
l = 3, c = 1 -> l * c = 3, IdQ = 0. This circuit is CLEAN by the metric.
It is the textbook redundant circuit: H-Z-H on one qubit is algebraically a single X gate, three \
gates where one suffices. Under a "redundancy" reading of Long Circuit it would be the flagship \
example. Under the metric it is not Long Circuit at all, because it is tiny.
This is the single most important thing to internalise: REDUNDANCY IS NOT THE SMELL. SIZE IS. Do \
not generate cancelling gate pairs and expect them to count as Long Circuit -- they will only \
inflate l slightly, and if the circuit stays small the label will come back clean."""

# data/raw/bugs4q/Terra-11000-15000_12801_Fixed_fixed_version.py -- misurato l=1, c=1, IdQ=0.
_REAL_COUNTEREXAMPLE_UNUSED = """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

theta1 = Parameter('theta1')

qc = QuantumCircuit(5)
qc.ry(theta1, 4)
"""

_REAL_COUNTEREXAMPLE_UNUSED_EXPLANATION = """Matrix: five rows, one column. Only q4 has anything \
in it; q0 to q3 are entirely empty.
l = 1, c = 1 -> l * c = 1, and IdQ = 0. This circuit is CLEAN by the metric.
Four out of five allocated qubits do nothing at all, and the Idle Qubits metric is still zero: \
those qubits have no two uses to wait between. Wasting allocated qubits is a real problem, but \
it is not this smell and none of the other seven QSMELL smells covers it either.
So: do NOT produce Idle Qubits by leaving a qubit untouched. It produces exactly nothing."""

# ---------------------------------------------------------------------------------------------
# Vincoli negativi. I primi tre derivano dai criteri di affidabilita' applicati al corpus reale
# (scripts/diagnostics/corpus_reliability_report.py): un circuito sintetico che li violasse
# sarebbe scartato dallo stesso metro con cui sono stati scartati 82 circuiti reali su 91.
# ---------------------------------------------------------------------------------------------

_NEGATIVE_CONSTRAINTS = """HARD CONSTRAINTS (these are rules, not stylistic advice -- a snippet \
that breaks one is discarded)

- Assign EXACTLY ONE QuantumCircuit in the snippet. Helper sub-circuits assigned to their own \
variable are forbidden: the pipeline measures the last circuit assigned, so a helper would be \
measured instead of your circuit.
- Use only ELEMENTARY, FLAT gate calls: qc.h(0), qc.cx(0, 1), qc.ccx(0, 1, 2), qc.rz(theta, 0), \
qc.measure(...), qc.barrier(). Do NOT use library or composite blocks -- no QFT, no \
EfficientSU2, no PauliEvolutionGate, no .to_gate(), no .control(), no .compose() of another \
circuit. A composite block occupies ONE cell of the matrix no matter how many gates it hides, so \
it makes the measurement meaningless.
- No classical feedback (c_if, if_test) and no randomness (random_circuit, unseeded numpy \
random): the circuit must measure identically on every run.
- The snippet must be complete and runnable on its own: every import present, no undefined name.
- Do not generate a circuit that is entirely inert from |0...0> (e.g. only gates that never \
trigger). Real computation with the target shape is wanted, not an empty shell.
- Loops (for kk in range(N): qc.h(kk)) are fine and encouraged for the deep shapes -- they are \
how real code produces long chains."""

# ---------------------------------------------------------------------------------------------
# Procedura: si disegna la forma, si contano l e c dallo schizzo, POI si scrive il codice.
# ---------------------------------------------------------------------------------------------

_GENERATION_PROCEDURE = """GENERATION PROCEDURE (two fields, in this exact order, for EVERY \
circuit)

STEP 1 -- MATRIX SKETCH (fill matrix_sketch): BEFORE writing any code, draw the execution matrix \
you are aiming for, in the same layout as the examples above: one line per qubit, labelled \
'q0:', 'q1:', ..., with each column holding either a gate name, a '.' for an empty cell, or \
'barr' for a barrier. Then, on a final line, write the counts you read off your own sketch, in \
the form: 'l=<value> c=<value> l*c=<value> IdQ=<value>'.
Count them from the sketch, honestly: l is the fullest row (barriers not counted), c is the \
fullest column (barriers not counted), IdQ is the longest run of '.' between two gates within a \
single row (barrier columns skipped, leading and trailing runs ignored).
If your counts do not hit the target shape for this batch, ADJUST THE SKETCH and recount before \
going on. This is cheap here and expensive later.

STEP 2 -- CODE (fill source_code): write the Qiskit snippet that produces exactly the matrix you \
just sketched. Remember the packing rule: an instruction slides as far LEFT as it can go. Two \
gates on disjoint qubits written on consecutive lines land in the SAME column, not in two. To \
force a gate into a later column you must create a dependency (a shared qubit) or place a \
barrier.

You are not asked to label the circuit, and you must not try. The pipeline measures l, c and IdQ \
from your code with the reference implementation and assigns the label itself. Your job is only \
to hit the requested shape."""

_OUTPUT_FORMAT_INSTRUCTIONS = """OUTPUT FORMAT
Respond with a structured object containing a single field "circuits": a list of exactly \
{count} circuit objects. Each circuit object has exactly two fields, in this order: \
matrix_sketch (the drawn matrix plus the final counts line) and source_code (the complete, \
runnable Qiskit snippet realising it). Do not include any text outside the structured object."""


class IdleTarget(str, Enum):
    """Cosa deve valere la metrica IdQ nei circuiti di un lotto."""

    NONE = "none"  # IdQ deve valere 0
    PRESENT = "present"  # IdQ deve valere > 0


@dataclass(frozen=True)
class BatchTheme:
    """Un lotto di generazione, definito da una FORMA target invece che da un'etichetta.

    l_range e c_range non sono decorativi: sono scelti in modo che il prodotto l*c cada
    interamente da un lato di LC_PRODUCT_CUTOFF, cosi' che l'etichetta risultante sia una
    conseguenza della forma richiesta e non un caso fortunato. La verifica a valle
    (verification.matches_batch_theme) confronta la forma MISURATA con questi intervalli.
    """

    theme: str
    qubit_range: tuple[int, int]
    l_range: tuple[int, int]
    c_range: tuple[int, int]
    idle_target: IdleTarget
    instruction: str
    count: int

    @property
    def expects_long_circuit(self) -> bool:
        """True se ogni prodotto l*c ammesso dagli intervalli raggiunge la soglia LC."""
        return is_long_circuit(self.l_range[0] * self.c_range[0])

    @property
    def targets_clean_circuits(self) -> bool:
        """True per il solo lotto che chiede circuiti privi di entrambi gli smell.

        Serve a build_batch_prompt per NON mostrare esempi puliti ai lotti che devono produrre
        smell: i few-shot pesano sull'output, e un prompt in cui la maggioranza dei circuiti
        mostrati e' pulita spinge verso circuiti puliti.
        """
        return not self.expects_long_circuit and self.idle_target == IdleTarget.NONE


# Sette lotti, 64 circuiti: 56 con almeno uno smell, 8 puliti. Lo sbilanciamento e' voluto e ha
# due motivi. Primo, il dataset serve a valutare un DETECTOR: i circuiti puliti sono la classe
# negativa, senza la quale precisione e falsi positivi non sono calcolabili e il ramo SMELL_FREE
# del MASEngine non viene mai esercitato -- servono, ma pochi bastano. Secondo, il corpus reale
# affidabile ne fornisce gia' 6 di suo (su 9 circuiti totali), quindi il sintetico non deve
# ripagare quel debito: deve coprire cio' che manca, cioe' gli smell.
#
# I conteggi non sono uniformi: lc_deep_narrow e idq_long_wait ne chiedono di piu' perche' hanno
# il bersaglio piu' stretto (l >= 20 il primo; l*c <= 18 CON IdQ >= 4 il secondo, una finestra di
# due unita' sotto la soglia LC) e ci si aspetta che una quota maggiore manchi la forma e venga
# scartata da matches_batch_theme.
#
# La copertura e' scelta contro i buchi misurati sul corpus reale
# (docs/qsmell_definizioni_e_esempi_prompt.md, sezione 6): l'unico Long Circuit affidabile e'
# bilanciato (7x5) e porta anche IdQ, quindi LC puro e le due forme estreme di LC -- profonda e
# stretta, larga e bassa -- non hanno alcun esempio reale e vanno specificate numericamente.
BATCH_THEMES: list[BatchTheme] = [
    BatchTheme(
        theme="lc_deep_narrow",
        qubit_range=(1, 3),
        l_range=(20, 40),
        c_range=(1, 2),
        idle_target=IdleTarget.NONE,
        instruction=(
            "Target shape: a DEEP, NARROW circuit. One qubit must carry a long chain of "
            "operations (l between 20 and 40), while at most two operations ever happen in the "
            "same column (c of 1 or 2). Use 1 to 3 qubits. This reaches Long Circuit through "
            "DEPTH alone, and Example 1 above is exactly this shape -- follow its mechanism (a "
            "loop appending gates to one qubit) but stop at the l this batch asks for, not at "
            "its 1002. Vary which gates go into the chain: do not repeat one gate identically. "
            "Every qubit you allocate must work in consecutive columns so that IdQ stays 0."
        ),
        count=10,
    ),
    BatchTheme(
        theme="lc_wide_shallow",
        qubit_range=(6, 10),
        l_range=(4, 6),
        c_range=(5, 8),
        idle_target=IdleTarget.NONE,
        instruction=(
            "Target shape: a WIDE, SHALLOW circuit. No qubit does more than 6 operations "
            "(l between 4 and 6), but 5 to 8 operations happen in the SAME column (c between 5 "
            "and 8). Use 6 to 10 qubits. This reaches Long Circuit through WIDTH alone: the "
            "exact opposite of Example 1, which gets there with c = 1 and enormous depth. No "
            "example below has this shape -- no measurable circuit in the real corpus is wide "
            "and shallow -- but the MECHANISM is visible in the very first column of the "
            "LONG CIRCUIT + IDLE QUBITS example: an x on q4 followed by a loop putting an h on "
            "q0..q4 lands FIVE operations in that one column, because they act on disjoint "
            "qubits and each slides as far left as it can. That is c = 5 out of four lines of "
            "code. Widen that mechanism to 6-10 qubits and keep the depth short, instead of "
            "stacking gates on one qubit as Example 1 does. Keep every qubit busy in "
            "consecutive columns so IdQ stays 0."
        ),
        count=8,
    ),
    BatchTheme(
        theme="lc_balanced",
        qubit_range=(4, 6),
        l_range=(5, 8),
        c_range=(4, 6),
        idle_target=IdleTarget.NONE,
        instruction=(
            "Target shape: BALANCED. l between 5 and 8, c between 4 and 6, on 4 to 6 qubits -- "
            "the shape of the LONG CIRCUIT + IDLE QUBITS example above, but WITHOUT its "
            "Idle Qubits. That is the hard part: that circuit has IdQ = 3 because some qubits "
            "stop working halfway through. Here every qubit must keep receiving operations "
            "across the whole circuit, with no gap left between two of its operations."
        ),
        count=8,
    ),
    BatchTheme(
        theme="idq_short_wait",
        qubit_range=(3, 5),
        l_range=(2, 4),
        c_range=(2, 4),
        idle_target=IdleTarget.PRESENT,
        instruction=(
            "Target shape: SMALL with a SHORT WAIT. l between 2 and 4 and c between 2 and 4, so "
            "that l * c stays well under 20 and the circuit is NOT Long Circuit, but IdQ must "
            "be at least 1. Use 3 to 5 qubits. Build the wait the way the ccx example above "
            "does: a multi-qubit gate acting on a subset DISJOINT from the qubits that must "
            "wait. Do not build it by leaving a qubit untouched (that gives IdQ = 0) and do not "
            "build it with cancelling gates (that gives a busy qubit)."
        ),
        count=8,
    ),
    BatchTheme(
        theme="idq_long_wait",
        qubit_range=(3, 5),
        l_range=(4, 6),
        c_range=(2, 3),
        idle_target=IdleTarget.PRESENT,
        instruction=(
            "Target shape: a LONG WAIT with no Long Circuit. l between 4 and 6, c between 2 and "
            "3 -- so l * c stays at 18 at the very most, just under the cutoff -- and IdQ of 4 "
            "or more. Use 3 to 5 qubits. This is the tightest constraint of all the batches: "
            "the long wait naturally inflates the circuit, and going one operation too far "
            "flips it into Long Circuit as well. The real example with IdQ = 7 sits at "
            "l * c = 18, two units below the cutoff -- study its shape. Sequential per-qubit "
            "sections followed by a shared final measure_all is the pattern that produces long "
            "waits at low c."
        ),
        count=10,
    ),
    BatchTheme(
        theme="both_smells",
        qubit_range=(4, 8),
        l_range=(6, 10),
        c_range=(4, 8),
        idle_target=IdleTarget.PRESENT,
        instruction=(
            "Target shape: BOTH smells. l between 6 and 10, c between 4 and 8 (so l * c is "
            "comfortably above 20), AND IdQ of at least 2. Use 4 to 8 qubits. Here the two "
            "smells should come from different features of the same circuit, as in the real "
            "five-qubit example above: the overall size gives Long Circuit, while some qubits "
            "stopping and restarting gives Idle Qubits."
        ),
        count=12,
    ),
    BatchTheme(
        theme="clean_mixed",
        qubit_range=(1, 8),
        l_range=(1, 5),
        c_range=(1, 3),
        idle_target=IdleTarget.NONE,
        instruction=(
            "Target shape: CLEAN. l at most 5 and c at most 3, so l * c stays at 15 at the very "
            "most, and IdQ exactly 0. Vary the qubit count across the batch, anywhere from 1 to "
            "8. Being clean is easy to achieve trivially and that is the trap: these must still "
            "look like purposeful quantum code, not filler. The two clean examples above show "
            "the two ways to do it -- per-qubit contiguous blocks each ending in their own "
            "measure, or every qubit busy in consecutive columns. Vary which one you use."
        ),
        count=8,
    ),
]


# I due errori che affondano un lotto SMELLY, in forma compatta: servono come guardrail, non
# come modelli da imitare, quindi niente matrice disegnata e niente trattazione estesa. La
# versione lunga (con matrice) resta nella variante per il lotto pulito, dove un circuito senza
# smell E' il bersaglio e vale la pena mostrarlo per esteso.
_SMELLY_BATCH_GUARDRAILS = f"""TWO WAYS TO MISS THE TARGET (do not do these -- both produce a \
CLEAN circuit, and the batch is wasted)

A. Redundancy is not the smell; SIZE is. This real circuit is the textbook redundant one -- \
H-Z-H on a single qubit is algebraically just an X, three gates where one suffices:
```python
{_REAL_COUNTEREXAMPLE_REDUNDANT}```
Measured: l = 3, c = 1 -> l * c = 3, IdQ = 0. CLEAN. Piling up cancelling gate pairs will not \
reach Long Circuit unless the circuit is genuinely big.

B. An untouched qubit is not an Idle Qubit. This real circuit allocates five qubits and touches \
only one:
```python
{_REAL_COUNTEREXAMPLE_UNUSED}```
Measured: l = 1, c = 1, IdQ = 0. CLEAN -- q0 to q3 have no TWO uses to wait between. Idling is \
produced by a qubit that works, stops while others work, and then works again."""


# Le quattro forme reali che portano uno smell, una per profilo. La chiave e' il profilo, cosi'
# _smelly_examples puo' ordinarle mettendo per prima quella del lotto: in un prompt lungo la
# posizione conta, e l'esempio piu' vicino al bersaglio non deve finire in fondo.
_SMELLY_EXAMPLES_BY_PROFILE: dict[str, tuple[str, str, str, str]] = {
    "lc_only": (
        "LONG CIRCUIT alone, deep and narrow",
        _REAL_LC_DEEP_NARROW,
        _REAL_LC_DEEP_NARROW_MATRIX,
        _REAL_LC_DEEP_NARROW_EXPLANATION,
    ),
    "both": (
        "LONG CIRCUIT + IDLE QUBITS together",
        _REAL_BOTH_SMELLS,
        _REAL_BOTH_SMELLS_MATRIX,
        _REAL_BOTH_SMELLS_EXPLANATION,
    ),
    "idq_long": (
        "IDLE QUBITS alone, long wait, just below the Long Circuit cutoff",
        _REAL_IDQ_LONG_WAIT,
        _REAL_IDQ_LONG_WAIT_MATRIX,
        _REAL_IDQ_LONG_WAIT_EXPLANATION,
    ),
    "idq_short": (
        "IDLE QUBITS alone, minimal wait",
        _REAL_IDQ_SHORT_WAIT,
        _REAL_IDQ_SHORT_WAIT_MATRIX,
        _REAL_IDQ_SHORT_WAIT_EXPLANATION,
    ),
}


def _smelly_examples(theme: BatchTheme) -> str:
    """Le quattro forme reali che PORTANO uno smell, con quella del lotto in prima posizione."""
    if theme.expects_long_circuit and theme.idle_target == IdleTarget.PRESENT:
        order = ["both", "lc_only", "idq_long", "idq_short"]
    elif theme.expects_long_circuit:
        order = ["lc_only", "both", "idq_long", "idq_short"]
    else:
        order = ["idq_long", "idq_short", "both", "lc_only"]

    blocks = []
    for position, key in enumerate(order, start=1):
        title, code, matrix, explanation = _SMELLY_EXAMPLES_BY_PROFILE[key]
        blocks.append(
            f"""Example {position} -- {title}:
```python
{code}```
Its execution matrix:
```
{matrix}
```
{explanation}"""
        )
    return "\n\n".join(blocks)


def _clean_examples() -> str:
    """Le due forme reali senza smell, per esteso: bersaglio del solo lotto pulito."""
    return f"""Example 1 -- CLEAN, sequential:
```python
{_REAL_CLEAN_SEQUENTIAL}```
Its execution matrix:
```
{_REAL_CLEAN_SEQUENTIAL_MATRIX}
```
{_REAL_CLEAN_SEQUENTIAL_EXPLANATION}

Example 2 -- CLEAN, compact:
```python
{_REAL_CLEAN_COMPACT}```
Its execution matrix:
```
{_REAL_CLEAN_COMPACT_MATRIX}
```
{_REAL_CLEAN_COMPACT_EXPLANATION}

Example 3 -- CLEAN, and it is redundant: this is the textbook H-Z-H circuit (algebraically a \
single X), which under a "redundancy" reading of Long Circuit would be the flagship example:
```python
{_REAL_COUNTEREXAMPLE_REDUNDANT}```
{_REAL_COUNTEREXAMPLE_REDUNDANT_EXPLANATION}

Example 4 -- CLEAN, with four of its five qubits untouched:
```python
{_REAL_COUNTEREXAMPLE_UNUSED}```
{_REAL_COUNTEREXAMPLE_UNUSED_EXPLANATION}"""


# Cio' che il lotto PULITO deve evitare: gli smell, in una riga ciascuno.
_SMELLY_CONTRAST_FOR_CLEAN = """FOR CONTRAST, the two shapes you must NOT produce in this batch, \
both measured on real circuits:
- l = 7, c = 5 -> l * c = 35: too big, that is LONG CIRCUIT.
- three qubits working one after another, each waiting while the others go, then a shared final \
measure_all -> IdQ = 7, that is IDLE QUBITS. A final measure_all is the most common accidental \
way to produce it: every qubit that finishes early waits for that shared column. Prefer per-qubit \
measures placed right after each qubit's own work."""


def build_batch_prompt(theme: BatchTheme) -> str:
    """Assembla il prompt completo (definizioni + few-shot reali + vincoli + forma target).

    I few-shot NON sono gli stessi per tutti i lotti. I circuiti reali affidabili sono 9 e solo
    3 portano uno smell: mostrarli tutti a ogni lotto significherebbe che, in un prompt che
    chiede circuiti smelly, la maggioranza dei circuiti mostrati e' pulita -- e i few-shot
    pesano sull'output. Ogni lotto vede quindi per esteso solo gli esempi del proprio bersaglio,
    piu' il contro-bersaglio in forma compatta.
    """
    if theme.targets_clean_circuits:
        examples = _clean_examples()
        contrast = _SMELLY_CONTRAST_FOR_CLEAN
    else:
        examples = _smelly_examples(theme)
        contrast = _SMELLY_BATCH_GUARDRAILS

    return f"""You are generating synthetic Qiskit circuits for a dataset used to evaluate the \
detection of two Quantum Code Smells. The smells are defined by MEASUREMENTS on a matrix, not by \
appearance, so your task is to hit a requested SHAPE.

{_EXECUTION_MATRIX}

{_SMELL_DEFINITIONS}

WORKED EXAMPLES -- all of these are real circuits mined from public repositories, and every \
number below was measured with the reference implementation, not estimated. They are the shape \
you are aiming for in this batch.

{examples}

{contrast}

{_NEGATIVE_CONSTRAINTS}

{_GENERATION_PROCEDURE}

BATCH INSTRUCTIONS FOR THIS REQUEST
{theme.instruction}

Generate exactly {theme.count} circuits. Every circuit in this batch must be STRUCTURALLY \
DISTINCT from every other circuit in the same batch: do not repeat the same gate sequence with \
only cosmetic changes (different qubit indices, different variable names, a different subset of \
qubits touched by an otherwise identical block). Vary the gates used, their order, the number of \
qubits within the allowed range, and the way the target shape is reached.

{_OUTPUT_FORMAT_INSTRUCTIONS.format(count=theme.count)}"""
