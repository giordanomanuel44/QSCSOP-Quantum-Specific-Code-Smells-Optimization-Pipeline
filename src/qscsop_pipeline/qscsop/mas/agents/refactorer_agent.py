"""RefactorerAgent: secondo agente del MAS, riscrive i circuiti per eliminare gli smell via CrewAI.

Stesso pattern del DetectorAgent (Agent CrewAI incapsulato, Task per chiamata, output strutturato),
ma con schema di output SEMPLICE: nessun campo di ragionamento (chain-of-thought). Lo si aggiungera'
solo se l'e2e mostrera' problemi concreti, stessa metodologia seguita per il DetectorAgent.

Il prompt e' in inglese: qwen2.5-coder e' orientato al codice e piu' affidabile su prompt tecnici
in inglese.

DEVIAZIONE sul few-shot Idle Qubits: l'esempio "corretto" NON e' piu' quello reale del dataset
(idq-fixed.py). La "correzione" ufficiale del dataset non rimuove il qubit inutile, lo ridefinisce
con un nuovo comportamento mantenendo 3 qubit -- strategia opposta a quella attesa dalla sezione
1.6 della tesi, che si aspetta che correggere un Idle Qubit RIDUCA il numero di qubit logici.
idq-fixed.py resta nel prompt solo come riferimento storico ("non fare cosi'"); la strategia
target e' insegnata da un esempio COSTRUITO ad hoc (qubit idle rimosso, 3 -> 2 qubit). Il few-shot
di Long Circuit resta invece quello reale del dataset (li' il problema non si presenta).
"""

from crewai import Agent, BaseLLM, Crew, Process, Task
from pydantic import BaseModel

from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO
from qscsop_pipeline.qscsop.mas.interfaces.i_refactorer_agent import IRefactorerAgent


class _RefactorSchema(BaseModel):
    """Schema Pydantic interno: solo tramite per l'output strutturato di CrewAI (non esposto)."""

    refactored_code: str


# Few-shot smelly -> fixed. Long Circuit e la coppia "storica" Idle Qubits sono il contenuto
# integrale reale dei file del dataset TheSmellyEight (correzioni applicate dagli autori dello
# studio); i due esempi ILLUSTRATIVI di Idle Qubits piu' sotto sono invece COSTRUITI ad hoc, per
# insegnare la strategia corretta di rimozione del qubit (vedi docstring del modulo).
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

_IDQ_FIXED_EXAMPLE = """from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
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

# Esempio ILLUSTRATIVO (NON tratto dal dataset): insegna la strategia corretta per Idle Qubits,
# cioe' RIMUOVERE il qubit inutile riducendo il numero di qubit dichiarati (3 -> 2), invece di
# ridefinirlo come fa la "correzione" reale di idq-fixed.py.
_IDQ_ILLUSTRATIVE_SMELLY_EXAMPLE = """from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.h(2)
qc.h(2)
qc.measure_all()
"""

_IDQ_ILLUSTRATIVE_FIXED_EXAMPLE = """from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
"""

# Blocco iniettato solo quando review_feedback non e' vuoto (iterazioni successive alla prima).
_REVIEW_FEEDBACK_SECTION = """
PREVIOUS ATTEMPT FEEDBACK

The previous refactoring attempt did NOT pass validation for this reason:
{review_feedback}
Specifically fix this problem in the new attempt, while still removing the original smell and \
keeping the circuit functionally equivalent to the input.
"""

_TASK_DESCRIPTION_TEMPLATE = """You are refactoring a Qiskit quantum circuit to remove exactly TWO \
Quantum Code Smells, and nothing else. You must preserve the circuit's exact behavior.

HOW TO FIX EACH SMELL

1. LONG CIRCUIT
Identify redundant or algebraically simplifiable gate sequences (for example, an H-Z-H sequence \
on the same qubit is exactly equivalent to a single X gate) and replace them with the shortest \
equivalent form, preserving the exact behavior of the circuit.

2. IDLE QUBITS
Identify qubits that are allocated but never involved in any meaningful operation (or whose \
effect is negligible). For the Idle Qubits smell, if a qubit does not contribute in any \
observable way to the result, the correct fix is to REMOVE it entirely from the circuit \
(reducing the number of declared qubits), NOT to modify its gates to give it a new purpose. \
The number of qubits in the corrected circuit MAY be lower than in the original when this \
happens, and that is expected.

FIX EXAMPLES

Example 1 -- LONG CIRCUIT (real correction applied by the authors of the study the dataset is \
taken from), before:
```python
{lc_smelly}```
after the fix:
```python
{lc_fixed}```

Example 2 -- IDLE QUBITS. This is what a circuit with this smell looks like (from the dataset):
```python
{idq_smelly}```
Note: the official correction from that study (shown here as a HISTORICAL REFERENCE only) does \
NOT remove the useless qubit, it redefines it with a new behavior while keeping 3 qubits -- this \
is NOT the strategy to follow for this task:
```python
{idq_fixed}```
The CORRECT strategy for Idle Qubits (illustrative example, NOT taken from the original dataset), \
before:
```python
{idq_illustrative_smelly}```
after (the idle qubit is REMOVED and the qubit count is reduced from 3 to 2):
```python
{idq_illustrative_fixed}```
Why: qubit 2 receives two consecutive H gates that cancel out (it returns exactly to its initial \
|0> state), so it never contributes to the observable result -- it must be REMOVED from the \
circuit (not redefined with new gates), reducing the number of declared qubits from 3 to 2 and \
updating measure_all() accordingly.

CIRCUIT TO REFACTOR NOW
```python
{code}```

WHAT THE DETECTOR REPORTED ABOUT THIS CIRCUIT
{report_details}
{review_feedback_section}
EQUIVALENCE CONSTRAINT (non-negotiable)

The refactored circuit MUST remain FUNCTIONALLY EQUIVALENT to the original on the qubits that \
carry actual information: the fix concerns only structural efficiency (fewer/shorter gates, fewer \
idle qubits). Removing a genuinely idle qubit -- one that never affects the observable result -- \
is allowed and expected even though it reduces the qubit count. Apart from that, do not add, \
remove or reorder anything that would change the computed state.

OUTPUT FORMAT

Return ONLY the refactored Python code in the refactored_code field: pure Python source, exactly \
as it would appear in a .py file. No Markdown code fences (no ```python), no explanatory text \
before or after, no prose comments describing what you changed."""

_EXPECTED_OUTPUT = (
    "A structured object with a single field refactored_code containing ONLY the refactored "
    "Qiskit Python source code, with no Markdown fences and no explanatory text."
)


class RefactorerAgent(IRefactorerAgent):
    """Riscrive i circuiti per eliminare Long Circuit e Idle Qubits incapsulando un Agent CrewAI."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm
        self._agent = Agent(
            role="Quantum Circuit Refactoring Specialist",
            goal=(
                "Rewrite a Qiskit circuit to remove the Long Circuit and Idle Qubits smells "
                "while preserving its exact functional behavior."
            ),
            backstory=(
                "You are an expert in quantum circuit optimization who refactors Qiskit code to "
                "eliminate redundant gate sequences and allocated-but-unused qubits, always "
                "keeping the circuit functionally equivalent to the original."
            ),
            llm=llm,
            verbose=False,
        )

    def refactor(self, code: str, smell_report: SmellReportDTO, review_feedback: str) -> str:
        """Riscrive code eliminando gli smell e ritorna il solo codice refattorizzato."""
        result = self._run_refactor_crew(code, smell_report, review_feedback)
        return result.refactored_code

    def _run_refactor_crew(
        self, code: str, smell_report: SmellReportDTO, review_feedback: str
    ) -> _RefactorSchema:
        """Esegue il Crew di refactoring e ritorna l'output strutturato; isola la chiamata all'LLM.

        Punto di mock nei test unitari: cosi' i test non istanziano mai un vero Agent/Crew/LLM.
        """
        # review_feedback vuoto (prima iterazione) -> la sezione feedback e' del tutto omessa.
        review_feedback_section = (
            _REVIEW_FEEDBACK_SECTION.format(review_feedback=review_feedback)
            if review_feedback
            else ""
        )
        task = Task(
            description=_TASK_DESCRIPTION_TEMPLATE.format(
                lc_smelly=_LC_SMELLY_EXAMPLE,
                lc_fixed=_LC_FIXED_EXAMPLE,
                idq_smelly=_IDQ_SMELLY_EXAMPLE,
                idq_fixed=_IDQ_FIXED_EXAMPLE,
                idq_illustrative_smelly=_IDQ_ILLUSTRATIVE_SMELLY_EXAMPLE,
                idq_illustrative_fixed=_IDQ_ILLUSTRATIVE_FIXED_EXAMPLE,
                code=code,
                report_details=smell_report.get_report_details(),
                review_feedback_section=review_feedback_section,
            ),
            expected_output=_EXPECTED_OUTPUT,
            agent=self._agent,
            output_pydantic=_RefactorSchema,
        )
        crew = Crew(agents=[self._agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()

        parsed = result.pydantic
        if not isinstance(parsed, _RefactorSchema):
            raise RuntimeError(
                "Il RefactorerAgent non ha prodotto un output conforme a _RefactorSchema. "
                f"Output grezzo ricevuto dal modello: {result.raw!r}"
            )
        return parsed
