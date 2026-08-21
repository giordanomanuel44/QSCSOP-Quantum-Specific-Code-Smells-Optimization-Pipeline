"""Definizioni degli smell, few-shot ed istruzioni di lotto per la generazione sintetica.

Il testo che finisce nel prompt (definizioni, few-shot, istruzioni) e' in inglese, stessa
scelta motivata gia' fatta per DetectorAgent (vedi detector_agent.py): qwen2.5-coder e' un
modello orientato al codice, piu' affidabile su prompt tecnici in inglese. Le definizioni qui
sono riformulate per la GENERAZIONE ("crea rispettando la definizione"), non sono una copia del
prompt di DetectorAgent (scritto per "riconoscere") -- ma restano concettualmente equivalenti,
stesso rigore.
"""

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from qscsop_pipeline.qscsop.mas.dto.quantum_smell_type import QuantumSmellType


class GeneratedCircuit(BaseModel):
    """Un singolo circuito generato dall'LLM, con lo smell dichiarato e la motivazione.

    L'ordine dei campi conta: in generazione autoregressiva ogni campo e' condizionato solo su
    quelli che lo precedono, quindi l'ordine qui sotto e' l'ordine di ragionamento imposto al
    modello, non solo la forma dello schema -- stesso principio gia' validato empiricamente in
    DetectorAgent._SmellDetectionSchema (vedi detector_agent.py).
    """

    # 1. source_code e' PRIMO per necessita' (non per lo stesso motivo del Detector): qui il
    #    codice e' l'OUTPUT da generare, non un input dato -- deve esistere prima che qualunque
    #    campo successivo possa riferirsi ad esso.
    source_code: str
    # 2. line_by_line_expansion e' SECONDO: trascrizione MECCANICA (non un'analisi) di ogni riga
    #    del source_code appena scritto in operazioni elementari "gate -> qubit_index", con le
    #    chiamate register-wide espanse esplicitamente in una entry per qubit. Ancora il modello
    #    ai fatti del proprio codice invece di permettergli di descriverne uno diverso.
    line_by_line_expansion: str
    # 3. qubit_operation_analysis e' TERZO cosi' il modello raggruppa per qubit SOLO a partire
    #    dalla trascrizione appena prodotta (non tornando al codice), prima di qualunque verdetto.
    qubit_operation_analysis: str
    # 4. reasoning resta, ma ora e' la conclusione ARTICOLATA a partire dall'analisi sopra, non
    #    piu' la prima cosa scritta -- elimina il rischio diagnosticato empiricamente di un
    #    reasoning che descrive un pattern diverso dal codice realmente scritto (es. record 2/7
    #    del lotto idle_qubits: dichiarato "doppio H su q2", il codice conteneva un solo H).
    reasoning: str
    # 5. intended_smells formalizza la conclusione di reasoning senza contraddirla (lista vuota
    #    se il circuito e' pulito), stessa logica di detected_smell_types nel Detector.
    intended_smells: list[QuantumSmellType]
    # 6. simplified_source_code e' ULTIMO perche' e' una CONSEGUENZA della dichiarazione appena
    #    fatta: si scrive solo dopo aver deciso che il circuito e' long_circuit, e serve a rendere
    #    quella dichiarazione verificabile a macchina (verification.verify_declared_simplification
    #    confronta i due circuiti via QiskitFacade.check_equivalence) invece che affidata alla
    #    prosa di reasoning, piu' volte trovata falsa in revisione manuale. None per i circuiti
    #    che non dichiarano long_circuit.
    simplified_source_code: str | None = None


class GenerationBatch(BaseModel):
    """Output strutturato atteso per un intero lotto di generazione."""

    circuits: list[GeneratedCircuit]


# ---------------------------------------------------------------------------------------------
# 3a. Definizioni degli smell riformulate per la generazione.
# ---------------------------------------------------------------------------------------------

_SMELL_DEFINITIONS = """SMELL DEFINITIONS (you must CREATE circuits that genuinely satisfy these \
definitions, not just look like they do)

1. LONG CIRCUIT
The circuit must contain a sequence of gates that produces a result obtainable more directly, \
with fewer operations. This does not have to be a literal cancellation (like H-Z-H collapsing to \
X) -- it can be a less obvious circuit identity, such as a sequence that implements a known \
operation through an indirect path when a direct, shorter, equivalent path exists (for example, \
building a SWAP out of three CNOTs conjugated by Hadamards instead of calling the SWAP gate \
directly).

2. IDLE QUBITS
The circuit must contain at least one qubit that, in the circuit's REAL behavior (not the \
surface appearance of the source code), contributes nothing observable to the result -- either \
because it never receives a genuine operation, or because the operations it does receive cancel \
out / leave it in a fixed, predictable state (e.g. two consecutive Hadamards that cancel back to \
identity, or a method call whose result is never assigned or used)."""

# ---------------------------------------------------------------------------------------------
# 3b. Few-shot SMELLY -- quattro coppie complete (smelly + fixed + spiegazione meccanicistica).
# ---------------------------------------------------------------------------------------------

_LC_SMELLY_CODE = """from qiskit import QuantumCircuit
from numpy import pi

qc = QuantumCircuit(1)


qc.h(0)
qc.z(0)
qc.h(0)



# ------------------------------------------------------------------------------

from qiskit import transpile

# Transpile
qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id'], optimization_level=0)

# Draw
"""

_LC_FIXED_CODE = """from qiskit import QuantumCircuit
from numpy import pi

qc = QuantumCircuit(1)





qc.x(0)

# ------------------------------------------------------------------------------

from qiskit import transpile

# Transpile
qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id'], optimization_level=0)

# Draw
"""

_LC_PAIR_EXPLANATION = """Why the smelly version is smelly: qc.h(0); qc.z(0); qc.h(0) applied to \
the same qubit is algebraically equivalent to a single X gate (H*Z*H = X). Three operations are \
used where one suffices -- gate count and depth are inflated for no benefit.
Why the fix is valid: the fixed version replaces the three redundant gates with the single \
equivalent X gate. The unitary effect is IDENTICAL, only the redundant path is removed."""

_IDQ_SMELLY_CODE = """from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
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


# ------------------------------------------------------------------------------

from qiskit import transpile

# Transpile
qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id'], optimization_level=0)

# Draw
"""

_IDQ_FIXED_CODE = """from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
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

# ------------------------------------------------------------------------------

from qiskit import transpile

# Transpile
qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id'], optimization_level=0)

# Draw
"""

_IDQ_PAIR_EXPLANATION = """Why the smelly version is smelly: every qubit receives a broadcast \
Hadamard (qc.h(qreg_q) touches q0, q1, q2), but q2 then receives ANOTHER Hadamard specifically \
(qc.h(qreg_q[2])). Two consecutive H gates on the same qubit cancel exactly (H*H = I), so q2 is \
back in |0>. The phase gates applied afterward (p, z, s) act on that fixed |0> state and carry \
no information -- q2 is an Idle Qubit, not because it receives zero gates, but because its net \
sequence of gates amounts to the identity. q0 and q1 receive only a SINGLE H each (real \
superposition) followed by their own phase gates, so they stay informative.
Why the fix is valid: the fixed version gives each qubit exactly one H (its own, no broadcast + \
duplicate), so q2 also stays in genuine superposition. All three qubits now contribute to the \
measured outcome."""

_LC_IDQ_COMBINED_SMELLY_CODE = """from qiskit import *
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

qc_bad.draw(output='text')
"""

_LC_IDQ_COMBINED_FIXED_CODE = """from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

q = QuantumRegister(4, name='q')
c = ClassicalRegister(4, name='c')
qc = QuantumCircuit(q, c)

qc.x(q[3])
qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.h(q[3])
qc.barrier(q)
qc.cx(q[1], q[3])
qc.cx(q[2], q[3])

qc.barrier(q)
qc.swap(q[0], q[1])
"""

_LC_IDQ_COMBINED_EXPLANATION = """Why the smelly version is smelly (empirically verified with \
Qiskit: fidelity 1.0 against the fixed version, every metric improved):
- IDLE QUBITS: q1 only ever receives the broadcast H from the `for kk in range(5): qc_bad.h(q[kk])` \
loop and is never touched again by anything -- it never becomes entangled and never influences \
the final result. Removed entirely (register shrinks 5 -> 4 qubits).
- LONG CIRCUIT: the seven-gate block `cx(2,0); h(2); h(0); cx(2,0); h(2); h(0); cx(2,0)` conjugates \
a CNOT with Hadamards on BOTH qubits, which inverts control and target \
(H tensor H * CX(a,b) * H tensor H = CX(b,a)). The whole sequence therefore computes \
CX(2,0)*CX(0,2)*CX(2,0) -- the textbook construction of a SWAP out of 3 CNOTs, used here as an \
indirect path to an operation Qiskit exposes directly.
Why the fix is valid: q1 is removed (Idle Qubits), and the 7-gate CNOT/Hadamard block is replaced \
by a single qc.swap(q[0], q[1]) call, the direct equivalent (Long Circuit fixed). Both smells \
were present on different qubits/sub-sequences of the same circuit."""

_IDQ_ORPHAN_SMELLY_CODE = """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

theta1 = Parameter('theta1')

qc = QuantumCircuit(5)
qc.ry(theta1, 4)
qc.control(4, ctrl_state=0, annotated=True)
qc.draw()
"""

_IDQ_ORPHAN_FIXED_CODE = """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

theta1 = Parameter('theta1')
qc = QuantumCircuit(1)
qc.ry(theta1, 0)
"""

_IDQ_ORPHAN_EXPLANATION = """Why the smelly version is smelly (empirically verified with Qiskit: \
fidelity 1.0 against the fixed version): qc.control(...) in Qiskit does NOT mutate the circuit it \
is called on -- it RETURNS a new controlled-gate object. Here the return value is never assigned \
or reused, so the call has zero effect on qc. The circuit's real instruction list contains only \
ry(theta1, 4); qubits 0-3 never receive any operation at all -- an extreme case of Idle Qubits \
(four out of five qubits are entirely untouched, not merely left in a cancelled-out state).
Why the fix is valid: the fixed version keeps only the one real instruction and shrinks the \
register to the single qubit that is actually used (5 -> 1 qubit), with identical behavior."""

# ---------------------------------------------------------------------------------------------
# 3c. Few-shot PULITI -- quattro esempi, nessuno smell da spiegare.
# ---------------------------------------------------------------------------------------------

_CLEAN_EXAMPLE_1 = _LC_FIXED_CODE
_CLEAN_EXAMPLE_2 = _IDQ_FIXED_CODE

_CLEAN_EXAMPLE_3 = """from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.cx(0, 1, ctrl_state='0')
qc.ccx(0, 1, 2, ctrl_state='00')
"""

_CLEAN_EXAMPLE_4 = """from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

iqft = QFT(3, inverse=True)
reversed_bits_QFT = iqft.reverse_bits()

circuit = QuantumCircuit(3)
circuit.compose(reversed_bits_QFT, inplace=True)
"""

# ---------------------------------------------------------------------------------------------
# 3d. Vincoli negativi espliciti.
# ---------------------------------------------------------------------------------------------

_NEGATIVE_CONSTRAINTS = """NEGATIVE CONSTRAINTS (explicit instructions, not examples):
- Do NOT generate circuits where the ENTIRE circuit is inert / does nothing starting from \
|0...0> (e.g. only gates conditioned on a classical value that is never triggered). This is a \
degenerate, non-representative case: real computation with an embedded waste is wanted, not \
"everything is dead".
- Do NOT use nested composite-gate constructions (sub-circuits converted to a gate and \
re-nested into another circuit) -- this is an unusual style relative to the rest of the dataset. \
Prefer direct, flat calls (qc.h(0), qc.cx(0, 1), etc.), consistent with the prevailing style \
observed in the real dataset.
- For CLEAN circuits: they must contain neither problematic pattern, but must still resemble \
plausible quantum code (not trivial filler circuits)."""

# ---------------------------------------------------------------------------------------------
# 3e. Passo di auto-verifica: ora una PROCEDURA che riempie i campi obbligati dello schema, non
# piu' solo un'istruzione in prosa facilmente ignorata -- la verifica e' un campo dell'output,
# non un'esortazione.
# ---------------------------------------------------------------------------------------------

_SELF_VERIFICATION_STEP = """GENERATION PROCEDURE (fill each field in this exact order, for \
EVERY circuit, after writing its source_code)

STEP 1 -- LINE-BY-LINE EXPANSION (fill line_by_line_expansion): go through the source_code you \
just wrote, ONE LINE AT A TIME, in order. For each line that applies a gate, write one explicit \
entry of the form 'gate_name -> qubit_index'. CRITICAL RULE for register-wide calls: if a line \
applies a gate to a whole register without an index (e.g. qc.h(qreg)), you MUST write ONE \
separate entry for EACH qubit index in that register (qc.h(qreg) on a 3-qubit register expands \
to 'h -> 0', 'h -> 1', 'h -> 2'). This is a mechanical transcription of the code you just wrote, \
not an analysis -- do not skip, merge, or summarize lines.

STEP 2 -- PER-QUBIT GROUPING (fill qubit_operation_analysis): using ONLY the expansion you just \
produced in STEP 1 (not the source_code text itself), group the entries by qubit index and list, \
for each qubit, its full ordered sequence of gates. Note any gate immediately cancelled by its \
own inverse (e.g. two consecutive H on the same qubit), and what state that leaves the qubit in.

STEP 3 -- CONCLUSION (fill reasoning): only after completing qubit_operation_analysis, articulate \
a technical explanation of which smell(s), if any, are genuinely present given the per-qubit \
analysis above, or why the circuit is clean. This must be a conclusion FROM the analysis you just \
wrote, not a restatement of what you intended when you started writing the code.

STEP 4 -- FORMALIZATION (fill intended_smells): populate intended_smells so it EXACTLY MATCHES \
the conclusion you just wrote in reasoning -- empty list if reasoning says the circuit is clean, \
"long_circuit" and/or "idle_qubits" if reasoning describes them. If STEP 2/3 reveal that the \
circuit you wrote does NOT actually exhibit the smell this batch requires (or accidentally \
exhibits the other one), REWRITE source_code before finalizing this entry, then redo STEPS 1-4 \
against the corrected code -- do not leave a mismatch between reasoning and intended_smells, and \
do not leave source_code out of sync with the analysis that follows it.

STEP 5 -- SIMPLIFIED VERSION (fill simplified_source_code): ONLY IF intended_smells contains \
"long_circuit", write a SECOND, complete, runnable Qiskit snippet that is the shorter version of \
the circuit you claim exists -- the same computation reached with fewer operations, with the \
redundant or indirect sequence replaced by its direct equivalent (as the "Fixed" snippets in the \
examples above do). It must be functionally EQUIVALENT to source_code (same qubits, in the same \
order, unless the circuit also exhibits Idle Qubits and the fix removes an unused one) and must \
assign its circuit to a variable exactly like source_code does. This snippet is \
compared to source_code by an automated equivalence check: a claim of redundancy that you cannot \
back with an actually equivalent shorter circuit is a WRONG claim -- if you cannot write one, the \
circuit does not exhibit Long Circuit, so go back to STEP 4 and remove "long_circuit" from \
intended_smells (or rewrite source_code so that it genuinely does). Leave this field null when \
intended_smells does not contain "long_circuit".

WARNING: the risk here is the mirror image of a known failure mode in circuit DETECTION -- citing \
a pattern from one of the few-shot examples above instead of verifying the specific source_code \
you just generated. Your line_by_line_expansion and qubit_operation_analysis must describe THIS \
circuit's actual code, not restate a pattern from Example A/B/C or the combined example just \
because it looks similar. If you catch yourself writing an expansion that does not match the \
source_code above it line-for-line, fix the mismatch before moving on."""

_OUTPUT_FORMAT_INSTRUCTIONS = """OUTPUT FORMAT
Respond with a structured object containing a single field "circuits": a list of exactly \
{total_circuits} circuit objects. Each circuit object has six fields, in this order: \
source_code (a complete, runnable Qiskit Python snippet that assigns the circuit to a variable, \
in the same flat, direct style as the examples above), line_by_line_expansion (mechanical \
transcription of every gate line of source_code into 'gate_name -> qubit_index' entries, with \
register-wide calls expanded into one entry per qubit), qubit_operation_analysis (per-qubit \
grouping of the operations listed in line_by_line_expansion, noting any cancellations and the \
resulting state), reasoning (a technical explanation, grounded in qubit_operation_analysis, of \
where the declared smell(s) are in the circuit's real behavior, or why the circuit is clean), and \
intended_smells (a list containing zero, one, or both of "long_circuit" / "idle_qubits" -- empty \
for clean circuits, exactly matching the conclusion in reasoning), and simplified_source_code (a \
complete, runnable Qiskit snippet with the shorter circuit equivalent to source_code -- REQUIRED \
whenever intended_smells contains "long_circuit", null otherwise). Do not include any text \
outside the structured object."""


class SmellFocus(str, Enum):
    """Quale smell un lotto di generazione deve enfatizzare."""

    LONG_CIRCUIT_ONLY = "long_circuit_only"
    IDLE_QUBITS_ONLY = "idle_qubits_only"
    BOTH = "both"
    NONE = "none"  # lotti di circuiti puliti


@dataclass(frozen=True)
class BatchTheme:
    """Un lotto di generazione: tema testuale, fascia di qubit, enfasi smell, mix richiesto."""

    theme: str
    qubit_range: tuple[int, int]
    smell_focus: SmellFocus
    instruction: str
    count_smelly: int
    count_clean: int

    @property
    def total_circuits(self) -> int:
        return self.count_smelly + self.count_clean


# 3f. Temi dei lotti: 45 smelly + 15 puliti = 60 circuiti totali, su 6 lotti. Copre le fasce
# 1-3 / 4-6 / 7-10 qubit (quest'ultima non coperta da nessun few-shot, estrapolazione esplicita)
# e le tre enfasi di smell (solo long_circuit, solo idle_qubits, entrambi insieme).
BATCH_THEMES: list[BatchTheme] = [
    BatchTheme(
        theme="qubit_1_3_long_circuit",
        qubit_range=(1, 3),
        smell_focus=SmellFocus.LONG_CIRCUIT_ONLY,
        instruction=(
            "Generate circuits using between 1 and 3 qubits. Every circuit must exhibit the "
            "LONG CIRCUIT smell (and NOT Idle Qubits) -- focus on redundant or algebraically "
            "simplifiable gate sequences on a small number of qubits, similar in spirit to (but "
            "not copied from) the H-Z-H example, and also explore less obvious circuit "
            "identities (e.g. a gate sequence that implements a known single- or two-qubit "
            "operation through an indirect path)."
        ),
        count_smelly=25,
        count_clean=0,
    ),
    BatchTheme(
        theme="qubit_1_3_idle_qubits",
        qubit_range=(1, 3),
        smell_focus=SmellFocus.IDLE_QUBITS_ONLY,
        instruction=(
            "Generate circuits using between 1 and 3 qubits. Every circuit must exhibit the "
            "IDLE QUBITS smell (and NOT Long Circuit) -- include at least one qubit whose real "
            "behavior contributes nothing to the outcome, either because it receives no genuine "
            "operation or because its operations cancel out into a fixed, predictable state."
        ),
        count_smelly=5,
        count_clean=0,
    ),
    BatchTheme(
        theme="qubit_4_6_both_smells",
        qubit_range=(4, 6),
        smell_focus=SmellFocus.BOTH,
        instruction=(
            "Generate circuits using between 4 and 6 qubits. Every circuit must exhibit BOTH "
            "the Long Circuit and Idle Qubits smells together, ideally on different qubits "
            "within the same circuit (e.g. one qubit wasted through cancellation, another "
            "involved in a redundant/indirect gate sequence)."
        ),
        count_smelly=15,
        count_clean=0,
    ),
    BatchTheme(
        theme="qubit_4_6_idle_qubits",
        qubit_range=(4, 6),
        smell_focus=SmellFocus.IDLE_QUBITS_ONLY,
        instruction=(
            "Generate circuits using between 4 and 6 qubits. Every circuit must exhibit the "
            "IDLE QUBITS smell (and NOT Long Circuit), using registers large enough to "
            "plausibly hide an unused or cancelled-out qubit among genuinely active ones."
        ),
        count_smelly=10,
        count_clean=0,
    ),
    BatchTheme(
        theme="qubit_7_10_extrapolation",
        qubit_range=(7, 10),
        smell_focus=SmellFocus.BOTH,
        instruction=(
            "Generate circuits using between 7 and 10 qubits -- an extrapolation beyond any "
            "qubit count shown in the examples above. Mix all three smell profiles across the "
            "batch: some circuits with only Long Circuit, some with only Idle Qubits, some with "
            "both together. At this qubit count, the redundancy or waste should still be "
            "locally recognizable (a specific sub-sequence, or a specific qubit), not something "
            "that requires reasoning about the whole register at once."
        ),
        count_smelly=10,
        count_clean=0,
    ),
    BatchTheme(
        theme="clean_circuits_mixed_range",
        qubit_range=(1, 10),
        smell_focus=SmellFocus.NONE,
        instruction=(
            "Generate CLEAN circuits (no Long Circuit, no Idle Qubits) using qubit counts "
            "spanning the same overall range as the smelly batches above (anywhere from 1 to "
            "10 qubits, vary it across the batch). Every allocated qubit must be genuinely used "
            "and every gate sequence must be non-redundant -- but the circuits must still "
            "resemble plausible, purposeful quantum code, not minimal filler."
        ),
        count_smelly=0,
        count_clean=0,
    ),
]


def build_batch_prompt(theme: BatchTheme) -> str:
    """Assembla il prompt completo (definizioni + few-shot + vincoli + istruzione di lotto)."""
    return f"""You are generating synthetic Qiskit circuits to expand a training/evaluation \
dataset for two Quantum Code Smells. Follow the definitions and constraints below precisely.

{_SMELL_DEFINITIONS}

EXAMPLES OF SMELLY CIRCUITS PAIRED WITH THEIR FIX

Example 1 -- LONG CIRCUIT:
Smelly:
```python
{_LC_SMELLY_CODE}```
Fixed:
```python
{_LC_FIXED_CODE}```
{_LC_PAIR_EXPLANATION}

Example 2 -- IDLE QUBITS:
Smelly:
```python
{_IDQ_SMELLY_CODE}```
Fixed:
```python
{_IDQ_FIXED_CODE}```
{_IDQ_PAIR_EXPLANATION}

Example 3 -- LONG CIRCUIT + IDLE QUBITS TOGETHER:
Smelly:
```python
{_LC_IDQ_COMBINED_SMELLY_CODE}```
Fixed:
```python
{_LC_IDQ_COMBINED_FIXED_CODE}```
{_LC_IDQ_COMBINED_EXPLANATION}

Example 4 -- IDLE QUBITS (extreme case, orphaned method call):
Smelly:
```python
{_IDQ_ORPHAN_SMELLY_CODE}```
Fixed:
```python
{_IDQ_ORPHAN_FIXED_CODE}```
{_IDQ_ORPHAN_EXPLANATION}

EXAMPLES OF CLEAN CIRCUITS (no smell to explain, just circuits without either pattern)

Clean example 1:
```python
{_CLEAN_EXAMPLE_1}```

Clean example 2:
```python
{_CLEAN_EXAMPLE_2}```

Clean example 3:
```python
{_CLEAN_EXAMPLE_3}```

Clean example 4:
```python
{_CLEAN_EXAMPLE_4}```

{_NEGATIVE_CONSTRAINTS}

{_SELF_VERIFICATION_STEP}

BATCH INSTRUCTIONS FOR THIS REQUEST
{theme.instruction}
Generate exactly {theme.count_smelly} smelly circuit(s) (non-empty intended_smells, consistent \
with the batch focus above) and exactly {theme.count_clean} clean circuit(s) (empty \
intended_smells).Every circuit in this batch must be STRUCTURALLY DISTINCT from every other \
circuit in the same batch -- do not repeat the same gate sequence with only cosmetic changes \
(different qubit indices, different variable names, or a different subset of qubits touched by \
an otherwise identical block). Vary the actual gates used, their order, and/or the specific \
identity/pattern being exploited across the batch.

{_OUTPUT_FORMAT_INSTRUCTIONS.format(total_circuits=theme.total_circuits)}"""
