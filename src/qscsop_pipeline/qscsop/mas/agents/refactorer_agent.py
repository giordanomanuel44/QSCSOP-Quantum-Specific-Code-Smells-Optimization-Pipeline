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

# Blocco iniettato solo quando review_feedback non e' vuoto (iterazioni successive alla prima).
_REVIEW_FEEDBACK_SECTION = """
PREVIOUS ATTEMPT FEEDBACK

The previous refactoring attempt did NOT pass validation for this reason:
{review_feedback}
Specifically fix this problem in the new attempt, while still removing the original smell and \
keeping the circuit functionally equivalent to the input.
"""

_TASK_DESCRIPTION_TEMPLATE = """You are applying a repair to a Qiskit quantum circuit. The analysis has ALREADY been done for you and is given below: your job is to EXECUTE it, not to redo it. Do not look for problems the report does not mention, and do not second-guess the ones it does.

CIRCUIT TO REFACTOR
```python
{code}```

WHAT THE DETECTOR PRESCRIBED FOR THIS CIRCUIT
{report_details}
{review_feedback_section}
HOW TO APPLY THE PRESCRIPTION

- Change exactly what the prescription names, at the lines it names. Nothing else.
- If the prescription says there is NO removable redundancy, return the circuit UNCHANGED. Do not invent a change to look productive: a circuit that is over the threshold purely because of its size cannot be brought under it without altering what it computes, and returning it intact is the correct answer.
- Removing a qubit is NEVER the fix for a waiting qubit. A qubit that is never used at all does not wait between two of its own operations, so removing one does not reduce the wait -- it is the wait that must go, by reordering the operations or by filling it.
- If you fill a wait, the operations you add must not make the busiest qubit's chain longer than it already is: that would make the circuit larger and the repair would be rejected even though the wait is gone.

FIX EXAMPLE

A circuit whose qubits are worked one at a time, so that each one sits idle while the others are used, before:
```python
{idq_smelly}```
after the fix -- the SAME operations, reordered so that each qubit runs its own block and is measured immediately, with no waiting in between. Nothing was removed and no qubit was dropped:
```python
{idq_fixed}```

EQUIVALENCE CONSTRAINT (non-negotiable)

The refactored circuit MUST remain FUNCTIONALLY EQUIVALENT to the original: the repair concerns only the structure, never what the circuit computes. Removing gates that genuinely cancel out is allowed because it preserves behaviour by definition. Apart from that, do not add, remove or reorder anything that would change the computed state.

OUTPUT FORMAT

Return ONLY the refactored Python code in the refactored_code field: pure Python source, exactly as it would appear in a .py file. No Markdown code fences (no ```python), no explanatory text before or after, no prose comments describing what you changed."""

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
                idq_smelly=_IDQ_SMELLY_EXAMPLE,
                idq_fixed=_IDQ_FIXED_EXAMPLE,
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
