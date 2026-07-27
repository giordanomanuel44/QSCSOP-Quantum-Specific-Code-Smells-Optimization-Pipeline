"""DetectorAgent: primo agente del MAS, rileva Long Circuit e Idle Qubits via CrewAI.

Il prompt (definizioni, few-shot, istruzioni di formato) e' scritto in inglese: qwen2.5-coder
e' un modello orientato al codice e reso in modo piu' affidabile su prompt tecnici in inglese;
il report_details generato ne eredita la lingua. Scelta documentata anche in CLAUDE.md/report.
"""

from crewai import Agent, BaseLLM, Crew, Process, Task
from pydantic import BaseModel

from qscsop_pipeline.qscsop.mas.dto.quantum_smell_type import QuantumSmellType
from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO
from qscsop_pipeline.qscsop.mas.interfaces.i_detector_agent import IDetectorAgent


class _SmellDetectionSchema(BaseModel):
    """Schema Pydantic interno: solo tramite per l'output strutturato di CrewAI (non esposto)."""

    # L'ordine dei campi conta: in generazione autoregressiva ogni campo e' condizionato solo su
    # quelli che lo precedono, quindi l'ordine qui sotto e' l'ordine di ragionamento imposto al
    # modello, non solo la forma dello schema.
    # 1. line_by_line_expansion e' PRIMO: trascrizione MECCANICA (non un'analisi) di ogni riga di
    #    codice in operazioni elementari "gate -> qubit_index", con le chiamate register-wide
    #    espanse esplicitamente in una entry per qubit. Cambia il compito cognitivo richiesto al
    #    modello da "inferisci quali qubit tocca questa riga" (fallito due volte, vedi
    #    docs/report_qscsop_refactoring_equivalence.md) a "riscrivi meccanicamente cosa fa ogni
    #    riga" -- lo scaffolding sta nella trascrizione stessa, non in una regola astratta. Campo
    #    interno: non viene mai propagato in SmellReportDTO.
    line_by_line_expansion: str
    # 2. qubit_operation_analysis e' SECONDO cosi' il modello raggruppa per qubit SOLO a partire
    #    dalla trascrizione appena prodotta (non tornando al codice originale), prima di impegnarsi
    #    su qualunque verdetto. Campo interno: non viene mai propagato in SmellReportDTO.
    qubit_operation_analysis: str
    # 3. report_details e' TERZO: il modello articola qui la conclusione discorsiva (quale smell,
    #    se presente, e perche') basandosi sull'analisi appena scritta, prima di dover scegliere i
    #    valori formali della lista.
    report_details: str
    # 4. detected_smell_types e' ULTIMO: deve limitarsi a formalizzare in valori vincolati di
    #    QuantumSmellType la conclusione gia' articolata in report_details (lista vuota se nessuno
    #    smell), non essere un giudizio indipendente. has_smells non e' chiesto separatamente, ma
    #    DERIVATO da questa lista in detect_smell.
    detected_smell_types: list[QuantumSmellType]


# Few-shot POSITIVI (detected_smell_types non vuoto): contenuto integrale dei due esempi del
# dataset TheSmellyEight, uno per smell in scope.
_LC_SMELLY_EXAMPLE = """from qiskit import QuantumCircuit
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

_IDQ_SMELLY_EXAMPLE = """from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
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

# Few-shot NEGATIVO (detected_smell_types vuoto): versione corretta reale dello stesso circuito
# di _LC_SMELLY_EXAMPLE, per insegnare al modello a distinguere il "prima" dal "dopo" sullo stesso
# caso ed evitare il bias verso il rilevamento indotto da soli esempi positivi.
_LC_FIXED_EXAMPLE = """from qiskit import QuantumCircuit
from numpy import pi

qc = QuantumCircuit(1)






qc.x(0)

# ------------------------------------------------------------------------------

from qiskit import transpile

# Transpile
qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id'], optimization_level=0)

# Draw
"""

_TASK_DESCRIPTION_TEMPLATE = """You are analyzing a Qiskit quantum circuit for exactly TWO \
Quantum Code Smells. Ignore any other issue.

SMELL DEFINITIONS

1. LONG CIRCUIT
The circuit contains redundant or algebraically simplifiable gate sequences that artificially \
inflate gate count and depth without need. For example, an H-Z-H sequence applied to the same \
qubit is equivalent to a single X gate, yet uses three operations instead of one. Look for: \
gate sequences that cancel out or collapse into a shorter operation, unnecessary repetitions, \
and operations that do not meaningfully contribute to the circuit's logic.

2. IDLE QUBITS
The circuit declares/allocates qubits that are never involved in any meaningful operation \
(no gate and no measurement), or whose net effect on the final result is negligible \
("wasted" qubits that occupy hardware resources without contributing to the computation).

IMPORTANT: if the circuit contains NO measurement instruction at all, judge Idle Qubits ONLY \
by whether a qubit receives ANY gate. A qubit that receives at least one gate (even a single \
one) is NOT idle, regardless of whether the circuit ever measures it -- the absence of a \
measurement is a property of the whole circuit, not evidence that a specific qubit is unused. \
Only flag a qubit as idle if it receives literally zero gates.

DISTINGUISHING LONG CIRCUIT FROM IDLE QUBITS WHEN GATES CANCEL:
A cancellation of gates (like two H gates in a row) is not itself a smell label -- it is a fact \
you must interpret. Ask what the cancellation RESULTS IN for that qubit:
- If the redundant/cancelling gates leave the qubit STILL PARTICIPATING in the computation (it \
is entangled with others, or its final state still carries information that affects the \
outcome), and the redundancy is merely wasteful length that could be simplified to a shorter \
equivalent sequence -- this is LONG CIRCUIT. The qubit stays useful; only the gate sequence is \
unnecessarily long.
- If the cancelling gates leave the qubit INERT -- returned to a fixed state (e.g. |0>) with no \
superposition and no entanglement, so that its measurement outcome is deterministic and \
contributes nothing to the computation -- this is IDLE QUBITS. The cancellation is not just \
wasteful length; it is the REASON the qubit ends up doing nothing.
Key question: after the cancellation, does the qubit still do useful work (Long Circuit) or is \
it left contributing nothing (Idle Qubits)? A qubit whose gates cancel to the identity and is \
then only followed by phase gates before a Z-basis measurement ends up deterministic and \
non-contributing -- Idle Qubits.
Note: the same circuit can in principle exhibit both smells on different qubits. Judge each \
qubit on its own resulting role.

EXAMPLES OF CIRCUITS WITH A SMELL (detected_smell_types is non-empty)

Example A -- LONG CIRCUIT:
```python
{lc_example}```
Why it is a smell: the sequence qc.h(0); qc.z(0); qc.h(0) on the same qubit is exactly \
equivalent to a single X gate. Three operations are used where one suffices, inflating gate \
count and depth for no benefit.

Example B -- IDLE QUBITS:
```python
{idq_example}```
Why it is a smell: qubit 2 receives TWO Hadamard gates in a row (qc.h(qreg_q) applies H to all \
qubits including q2, then qc.h(qreg_q[2]) applies a second H to q2 specifically). Two \
consecutive H gates on the same qubit cancel out exactly, leaving q2 in state |0> -- \
deterministically, with no superposition. Every phase-type gate applied to q2 afterward (p, z, \
s) acts on a state that is already |0>, so q2's measurement outcome is fixed and 100% \
predictable: it carries zero information about the computation. This is what makes q2 an Idle \
Qubit -- not merely that its phase gates are 'wasted' in isolation (that is true of phase gates \
before a Z-basis measurement in general, on any qubit, smelly or not), but that q2's final \
outcome is deterministic and uninformative because of the redundant double-Hadamard. \
IMPORTANT DISTINCTION: 'deterministic outcome' alone is NOT the criterion for Idle Qubits. A \
single meaningful gate (like a lone X, or any single non-self-cancelling gate) also produces a \
deterministic outcome, yet it performs a real, non-trivial operation on the qubit -- it is NOT \
idle. What makes q2 idle here is specifically that its net sequence of gates is EQUIVALENT TO \
THE IDENTITY: two consecutive H gates cancel out exactly, so q2 ends up in exactly the same \
state it started in, as if untouched. When judging Idle Qubits, ask: does this qubit's sequence \
of gates, taken together, amount to doing NOTHING NET (a no-op, like H followed by H, or any \
gate immediately undone by its own inverse) -- that is idle. A qubit receiving a single gate \
that performs a real transformation (flips the state, rotates it into superposition, etc.) is \
NEVER idle, even though its resulting state may be perfectly predictable.

CONTRAST WITH EXAMPLE A/C: the X gate in Example C (the fixed version of Example A) is the \
OPPOSITE case: a single, meaningful, non-cancelling gate. Do not confuse it with q2's \
double-Hadamard case above -- a lone X gate is real work, not redundancy. If you find yourself \
citing the double-Hadamard reasoning to justify flagging a DIFFERENT circuit that does not \
itself contain two cancelling gates on the same qubit, you are misapplying the example: \
re-check the actual gate sequence of the qubit you are judging, in the circuit under analysis, \
not in a previous example.

EXAMPLE OF A CLEAN CIRCUIT (detected_smell_types is empty)

Example C -- SAME CIRCUIT FAMILY AS EXAMPLE A, AFTER THE SMELL WAS FIXED:
```python
{lc_fixed_example}```
Why it is clean: this is the corrected version of Example A. The redundant H-Z-H sequence has \
been replaced by the single equivalent X gate, so there is no redundancy left to simplify and \
the one qubit is actively used. There is no remaining anomaly to report: detected_smell_types \
must be an empty list. Do not flag a circuit as smelly just because earlier examples were smelly \
-- judge each circuit on its own merits.

NOTE: the presence of phase-type gates (p, z, s) right before a measurement is NEVER by itself \
evidence of Idle Qubits -- it is normal and appears in both smelly and correctly-fixed circuits \
of this dataset. Judge Idle Qubits only by whether the qubit's final outcome is deterministic \
due to redundancy, as explained above.

CIRCUIT TO ANALYZE
```python
{code}```

REASONING PROCEDURE (fill line_by_line_expansion FIRST, then qubit_operation_analysis, before \
deciding)

STEP 1 -- LINE-BY-LINE EXPANSION (fill line_by_line_expansion first): Go through the circuit \
code ONE LINE AT A TIME, in order. For each line that applies a gate, write one explicit entry \
of the form 'gate_name -> qubit_index'. CRITICAL RULE for register-wide calls: if a line applies \
a gate to a whole register without an index (e.g. qc.h(qreg_q)), you MUST write ONE separate \
entry for EACH qubit index in that register. Example transcription:
  Code line 'qc.h(qreg_q)' with a 3-qubit register expands to:
    h -> 0
    h -> 1
    h -> 2
  Code line 'qc.h(qreg_q[2])' expands to:
    h -> 2
Do NOT skip, merge, or summarize lines. Transcribe every gate line literally and completely, \
expanding every register-wide call into one entry per qubit. This expansion is a mechanical \
transcription task, not an analysis -- just faithfully rewrite what each line does.

STEP 2 -- PER-QUBIT GROUPING (fill qubit_operation_analysis): Using ONLY the expansion you just \
produced in STEP 1 (not the original code), group the entries by qubit index and list, for each \
qubit, its full ordered sequence of gates. Count how many times each gate appears per qubit -- \
if a qubit shows the same gate twice in a row (e.g. two h entries), note that they may cancel. \
If they cancel, DO NOT automatically label it Long Circuit: determine what the qubit does AFTER \
the cancellation. If the qubit is left inert (back to |0>, only phase gates before a Z-basis \
measurement, no entanglement, deterministic outcome), classify it as IDLE QUBITS, not Long \
Circuit -- use the disambiguation rule in the smell definitions above. Then apply the Idle \
Qubits / Long Circuit criteria as defined above. A qubit may be declared Idle ONLY if, after this \
explicit grouping, it turns out to have zero meaningful operations. Do not infer that a qubit is \
unused without having first listed its operations: if you find even a single gate or measurement \
involving it, it is NOT an idle qubit.

Only after completing qubit_operation_analysis, write report_details: a concise technical \
explanation of which smell(s) you found and where, or, if the circuit is clean, why it is clean. \
THEN, as the very last step, populate detected_smell_types so that it EXACTLY MATCHES the \
conclusion you just wrote in report_details: if report_details says the circuit is clean or that \
no anomaly was found, detected_smell_types MUST be an empty list. If report_details describes a \
Long Circuit, include "long_circuit". If it describes Idle Qubits, include "idle_qubits". Never \
emit a detected_smell_types value that contradicts your own report_details -- the list is a \
formalization of the verdict you just articulated, not an independent judgment. Respond ONLY \
with the required structured object, with no extra text outside the requested format."""

_EXPECTED_OUTPUT = (
    "A structured object with four fields, in this order: line_by_line_expansion (mechanical "
    "transcription of every gate line into 'gate_name -> qubit_index' entries, with "
    "register-wide calls expanded into one entry per qubit), qubit_operation_analysis "
    "(step-by-step listing of every qubit and the operations involving it, grouped from the "
    "expansion above), report_details (string describing the detected smell(s) or why the "
    "circuit is clean), and detected_smell_types (a list containing zero, one, or both of the "
    'allowed values "long_circuit" and "idle_qubits", exactly matching the conclusion in '
    "report_details)."
)


class DetectorAgent(IDetectorAgent):
    """Rileva Long Circuit e Idle Qubits incapsulando un Agent CrewAI a output strutturato."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm
        self._agent = Agent(
            role="Quantum Code Smell Detector",
            goal=(
                "Detect whether a Qiskit circuit exhibits the Long Circuit or Idle Qubits "
                "smell, and report the finding precisely."
            ),
            backstory=(
                "You are a specialist in quantum circuit optimization who reviews Qiskit code "
                "for two specific inefficiencies: redundant gate sequences (Long Circuit) and "
                "allocated-but-unused qubits (Idle Qubits)."
            ),
            llm=llm,
            verbose=False,
        )

    def detect_smell(self, code: str) -> SmellReportDTO:
        """Analizza code e mappa l'esito strutturato dell'agente in un SmellReportDTO."""
        schema = self._run_detection_crew(code)
        # has_smells e' DERIVATO dalla lista, non richiesto separatamente al modello: elimina per
        # costruzione la possibilita' di una risposta internamente contraddittoria (es.
        # has_smells=True con lista vuota, o viceversa).
        has_smells = bool(schema.detected_smell_types)
        detected_smells = [smell_type.value for smell_type in schema.detected_smell_types]
        return SmellReportDTO(
            has_smells=has_smells,
            report_details=schema.report_details,
            detected_smells=detected_smells,
        )

    def _run_detection_crew(self, code: str) -> _SmellDetectionSchema:
        """Esegue il Crew su code e ritorna l'output strutturato; isola la chiamata all'LLM.

        Punto di mock nei test unitari: tutta la logica di mapping/validazione a valle passa
        di qui, cosi' i test non istanziano mai un vero Agent/Crew/LLM.
        """
        task = Task(
            description=_TASK_DESCRIPTION_TEMPLATE.format(
                lc_example=_LC_SMELLY_EXAMPLE,
                idq_example=_IDQ_SMELLY_EXAMPLE,
                lc_fixed_example=_LC_FIXED_EXAMPLE,
                code=code,
            ),
            expected_output=_EXPECTED_OUTPUT,
            agent=self._agent,
            output_pydantic=_SmellDetectionSchema,
        )
        crew = Crew(agents=[self._agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()

        parsed = result.pydantic
        if not isinstance(parsed, _SmellDetectionSchema):
            raise RuntimeError(
                "Il DetectorAgent non ha prodotto un output conforme a _SmellDetectionSchema. "
                f"Output grezzo ricevuto dal modello: {result.raw!r}"
            )
        return parsed
