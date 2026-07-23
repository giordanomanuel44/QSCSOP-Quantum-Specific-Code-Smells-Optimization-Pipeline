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

    # L'ordine dei campi conta: qubit_operation_analysis e' PRIMO cosi' il modello genera il
    # ragionamento (chain-of-thought) prima di impegnarsi sul verdetto, non a posteriori.
    # Questo campo resta interno: non viene mai propagato in SmellReportDTO.
    qubit_operation_analysis: str
    # Il modello sceglie zero, uno o entrambi i valori vincolati di QuantumSmellType (lista vuota
    # se nessuno smell): has_smells non e' piu' chiesto separatamente, ma DERIVATO in detect_smell.
    detected_smell_types: list[QuantumSmellType]
    report_details: str


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
Why it is a smell: each of the three qubits receives, after an initial Hadamard, only \
phase-type rotations (p, z=p(pi), s=p(pi/2)) and is then measured directly in the computational \
basis. Phase rotations applied right before a Z-basis measurement do not change the measurement \
outcome distribution, so those operations are effectively wasted: the qubits are allocated and \
touched but contribute no observable information to the result, which is the essence of the \
Idle Qubits smell.

EXAMPLE OF A CLEAN CIRCUIT (detected_smell_types is empty)

Example C -- SAME CIRCUIT FAMILY AS EXAMPLE A, AFTER THE SMELL WAS FIXED:
```python
{lc_fixed_example}```
Why it is clean: this is the corrected version of Example A. The redundant H-Z-H sequence has \
been replaced by the single equivalent X gate, so there is no redundancy left to simplify and \
the one qubit is actively used. There is no remaining anomaly to report: detected_smell_types \
must be an empty list. Do not flag a circuit as smelly just because earlier examples were smelly \
-- judge each circuit on its own merits.

CIRCUIT TO ANALYZE
```python
{code}```

REASONING PROCEDURE (fill qubit_operation_analysis FIRST, before deciding)

Before giving any verdict, in the qubit_operation_analysis field list EVERY qubit of the circuit \
one by one (by index), and for each one explicitly write ALL the operations that involve it \
(gates and measurements), reading the ENTIRE code carefully from start to end, including the \
final lines and those separated by blank lines or barriers. A qubit may be declared Idle ONLY \
if, after this explicit listing, it turns out to have zero meaningful operations. Do not infer \
that a qubit is unused without having first listed its operations: if you find even a single \
gate or measurement involving it, it is NOT an idle qubit.

Only after completing qubit_operation_analysis, populate detected_smell_types with the smells \
you found: choose zero, one, or both of the ONLY two allowed values -- "long_circuit" and \
"idle_qubits". Use an empty list if the circuit is clean, ["long_circuit"] or ["idle_qubits"] \
for a single smell, and both values if the circuit exhibits both. Then write in report_details a \
concise technical explanation of which smell(s) you found and where (or, if none, why the circuit \
is clean). Respond ONLY with the required structured object, with no extra text outside the \
requested format."""

_EXPECTED_OUTPUT = (
    "A structured object with three fields, in this order: qubit_operation_analysis "
    "(step-by-step listing of every qubit and the operations involving it), "
    "detected_smell_types (a list containing zero, one, or both of the allowed values "
    '"long_circuit" and "idle_qubits"), and report_details (string describing the detected '
    "smell(s) or why the circuit is clean)."
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
