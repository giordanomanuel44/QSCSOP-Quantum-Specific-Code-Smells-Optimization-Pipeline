"""DetectorAgent: rilevamento IBRIDO di Long Circuit e Idle Qubits.

LA FACADE MISURA E LE SOGLIE DECIDONO; L'LLM PRESCRIVE. E' il terzo assetto di questo agente e
nasce da un fallimento documentato: alle versioni precedenti si chiedeva di CLASSIFICARE, cioe'
in ultima analisi di contare operazioni su una matrice di esecuzione, e il modello non ne e'
capace. Nella generazione sintetica, messo davanti allo stesso compito, ha riportato l = numero
di qubit in 8 record su 8 e ha degenerato sul disegno della matrice bruciando venti minuti.

Il compito e' quindi diviso secondo chi lo sa fare:

  - `IQiskitFacade.calculate_smell_metrics` misura l, c, l*c e IdQ. Esatto per costruzione.
  - `detection_thresholds` decide `has_smells` e `detected_smells`. Zero falsi negativi, che
    erano il fallimento peggiore: un falso negativo esclude un circuito smelly dal ciclo di
    ottimizzazione senza che nessuno se ne accorga.
  - L'LLM riceve codice E numeri misurati, e produce il solo `report_details`: la PRESCRIZIONE
    di cosa cambiare, che finisce testualmente nel prompt del RefactorerAgent.

Su un circuito pulito l'LLM non viene invocato affatto.

CONSEGUENZA SUL PROMPT. Sono spariti la sezione SMELL DEFINITIONS (definizioni semantiche
abbandonate con l'allineamento a QSMELL: il suo esempio portante, H-Z-H, misura l*c = 3 ed e'
PULITO), la sezione che distingueva i due smell in caso di gate che si cancellano (problema
inesistente: con le definizioni metriche i due smell sono ortogonali per costruzione) e la
REASONING PROCEDURE con line_by_line_expansion e qubit_operation_analysis, che era scaffolding
per far contare le operazioni per qubit al modello -- lavoro che ora fa la facade, esatto.

Il prompt e' in inglese, stessa scelta motivata degli altri agenti: i modelli coder sono piu'
affidabili su prompt tecnici in inglese.
"""

from crewai import Agent, BaseLLM, Crew, Process, Task
from pydantic import BaseModel

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qscsop.mas.detection_thresholds import (
    IDLE_QUBITS_CUTOFF,
    LC_PRODUCT_CUTOFF,
    has_idle_qubits,
    is_long_circuit,
)
from qscsop_pipeline.qscsop.mas.dto.quantum_smell_type import QuantumSmellType
from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO
from qscsop_pipeline.qscsop.mas.interfaces.i_detector_agent import IDetectorAgent


class _SmellPrescriptionSchema(BaseModel):
    """Schema Pydantic interno: un solo campo, la prescrizione.

    Le versioni precedenti chiedevano quattro campi, di cui tre erano impalcatura di
    ragionamento e uno era il verdetto (detected_smell_types). Il verdetto ora non si chiede
    piu': lo decidono le soglie sulla misura della facade, quindi non e' rappresentabile un
    output del modello che contraddica la misura.
    """

    report_details: str


# ---------------------------------------------------------------------------------------------
# La regola meccanica, misurata. E' il contenuto centrale del prompt: senza, il modello prescrive
# rimozioni che non spostano la metrica.
# ---------------------------------------------------------------------------------------------

_MECHANICS = f"""HOW THE TWO METRICS ACTUALLY MOVE

LONG CIRCUIT is l * c, where l is the largest number of operations on any single qubit and c is \
the largest number of operations happening in one time step. It is smelly at l * c >= \
{LC_PRODUCT_CUTOFF}.

To bring l * c down you must bring **l** down. These are measured facts, not guidelines:

- Removing operations lowers l. It does NOT reliably lower c: c only falls as a side effect when \
the removal happens to compact the columns, and you cannot aim for it.
- l falls ONLY if you remove operations from EVERY qubit that currently holds the maximum. \
Removing from a qubit that is not at the maximum changes NOTHING: measured, a circuit at \
(l=4, c=2, l*c=8) stays at exactly (4, 2, 8) after a genuine redundancy is removed from a \
non-maximum qubit. The work is real and the metric does not move.
- The same applies when several qubits share the maximum: removing from one of them alone leaves \
l unchanged. You must remove from all of them.

The list of qubits currently holding the maximum is given to you below. Use it.

IDLE QUBITS is the longest wait a qubit spends between two of its own operations. It is smelly \
at IdQ > {IDLE_QUBITS_CUTOFF}. It is repaired by REORDERING or by filling the wait, never by \
removing the qubit -- removing a qubit does not lower IdQ, and a qubit that is never used at all \
has IdQ = 0 by definition.

One constraint on filling a wait: the operations you add must NOT make the busiest qubit's chain \
longer than it already is, otherwise l grows, l * c gets worse, and the refactoring is rejected \
even though IdQ improved."""


# ---------------------------------------------------------------------------------------------
# Few-shot. Nessuno inventato: sono i casi che abbiamo misurato, e i due centrali sono le
# trappole -- senza, il modello produce prescrizioni oneste e inutili.
# ---------------------------------------------------------------------------------------------

_EXAMPLES = """WORKED EXAMPLES

Example 1 -- removal on the qubit that holds the maximum (this is what works)
```python
qc = QuantumCircuit(2)
qc.x(0)
qc.x(0)
qc.h(0)
qc.h(1)
qc.cx(0, 1)
```
Measured l=4, c=2, l*c=8; the maximum is held by q0 alone. The two consecutive x on q0 cancel. \
Removing them takes q0 from 4 operations to 2, so l becomes 2 and l*c becomes 4.
Prescription: "Remove the two consecutive qc.x(0) calls (lines 2-3). They cancel out, and q0 is \
the qubit holding l=4."

Example 2 -- removal on a qubit that is NOT at the maximum (this does nothing)
```python
qc = QuantumCircuit(2)
qc.h(0)
qc.x(0)
qc.x(0)
qc.h(1)
qc.z(1)
qc.s(1)
qc.t(1)
```
Measured l=4, c=2, l*c=8; the maximum is held by q1, which has four operations. The two x on q0 \
DO cancel, and removing them is a legitimate simplification -- but q0 only has three operations, \
so l stays at 4 and l*c stays at 8. Measured before and after: identical.
Prescription: "The redundancy on q0 (two consecutive qc.x(0)) does not affect the metric: q0 is \
not at the maximum. l is held by q1, which carries no removable redundancy."

Example 3 -- the maximum is shared (remove from all of them, or from none)
```python
qc = QuantumCircuit(2)
qc.x(0)
qc.x(0)
qc.h(0)
qc.x(1)
qc.x(1)
qc.h(1)
qc.cx(0, 1)
```
Measured l=4, c=2, l*c=8; the maximum is held by BOTH q0 and q1. Removing the cancelling pair \
from q0 alone leaves l=4 and l*c=8, measured. Removing it from both brings l to 2 and l*c to 4.
Prescription: "Remove the cancelling x pair from q0 (lines 2-3) AND the one from q1 (lines 5-6). \
Both qubits hold l=4; removing from only one leaves the metric unchanged."

Example 4 -- no removable redundancy (say so)
```python
qc = QuantumCircuit(4, 4)
for layer in range(3):
    qc.cx(0, 1)
    qc.cz(2, 3)
    qc.h(0)
    qc.x(1)
    qc.y(2)
    qc.z(3)
    for q in range(4):
        qc.p(pi / 4, q)
for q in range(4):
    qc.measure(q, q)
```
Measured l=10, c=4, l*c=40. Every gate is distinct and contributes; nothing cancels. Reaching \
l * c < 20 would require l <= 4, that is removing 6 of the 10 operations from every qubit at the \
maximum, which cannot be done while preserving the circuit's behaviour.
Prescription: "No removable redundancy. Every operation contributes and nothing cancels; this \
circuit is above the threshold by size alone and cannot be brought under it without changing \
what it computes."
"""


_TASK_DESCRIPTION_TEMPLATE = """You are prescribing a repair for a Qiskit quantum circuit that \
has ALREADY been measured and classified. The classification is not your job and is not in \
question: it was computed exactly, from the circuit's execution matrix. Your job is to say \
WHAT TO CHANGE, precisely enough that another agent can apply it without re-analysing anything.

{mechanics}

{examples}

CIRCUIT TO REPAIR
```python
{code}```

MEASURED VALUES FOR THIS CIRCUIT (exact, do not recompute or second-guess them)
{measurements}

WHAT TO WRITE

Write report_details: a precise, actionable prescription addressed to the agent that will edit \
this code. It must name concrete lines and concrete operations, not general advice.

For LONG CIRCUIT, state: the target (which value of l brings l * c under the threshold), which \
qubits hold the maximum, how many operations must go from each of them, and WHICH specific \
operations are redundant, quoting the lines. Identifying the redundancy is the part only you can \
do -- the measurement above cannot tell which gates cancel.

For IDLE QUBITS, state: which qubit waits, between which two of its own operations the wait \
falls, and what to do about it -- reorder so the wait disappears, or fill it -- without \
lengthening the chain of the busiest qubit.

IF THERE IS NOTHING TO REMOVE, SAY SO. A circuit can be over the threshold purely because of its \
size, with every operation contributing and nothing cancelling. Reporting "no removable \
redundancy" is a correct and useful answer: it tells the pipeline this circuit is not repairable \
without changing its behaviour. Do NOT invent a redundancy to justify the classification you \
were given -- a fabricated prescription makes the next agent produce a circuit that is no longer \
equivalent to the original, and the failure gets attributed to it instead of to you.

Respond ONLY with the required structured object, with no extra text outside it."""


_EXPECTED_OUTPUT = (
    "A structured object with a single field, report_details: an actionable prescription naming "
    "the concrete lines and operations to change, or an explicit statement that no removable "
    "redundancy exists."
)


class DetectorAgent(IDetectorAgent):
    """Misura con la facade, decide con le soglie, e fa prescrivere la riparazione all'LLM."""

    def __init__(self, llm: BaseLLM, facade: IQiskitFacade) -> None:
        self._facade = facade
        self._llm = llm
        self._agent = Agent(
            role="Quantum Circuit Repair Planner",
            goal=(
                "Given a circuit and its exact measured smell metrics, prescribe precisely which "
                "operations to change so that the metrics fall below their thresholds."
            ),
            backstory=(
                "You are a specialist in quantum circuit optimization. You do not classify "
                "circuits -- that is done exactly, by measurement, before you are called. You "
                "read code and decide what to change, and you say plainly when nothing can be "
                "changed without altering the circuit's behaviour."
            ),
            llm=llm,
            verbose=False,
        )

    def detect_smell(self, code: str) -> SmellReportDTO:
        """Misura il circuito, decide con le soglie e prescrive solo se c'e' qualcosa da riparare.

        L'LLM viene invocato SOLO sui circuiti smelly: su un circuito pulito non c'e' nulla da
        prescrivere, e il report deterministico riporta i valori misurati con le rispettive
        soglie -- piu' informativo di una frase generata, oltre che gratis.
        """
        metrics = self._facade.calculate_smell_metrics(code)
        long_circuit, idle_qubits = metrics["longCircuit"], metrics["idleQubits"]

        detected_smells = []
        if is_long_circuit(long_circuit["value"]):
            detected_smells.append(QuantumSmellType.LONG_CIRCUIT.value)
        if has_idle_qubits(idle_qubits["value"]):
            detected_smells.append(QuantumSmellType.IDLE_QUBITS.value)

        if not detected_smells:
            return SmellReportDTO(
                has_smells=False,
                report_details=_clean_report(long_circuit, idle_qubits),
                detected_smells=[],
            )

        prescription = self._run_prescription_crew(
            code, _format_measurements(long_circuit, idle_qubits, detected_smells)
        )
        return SmellReportDTO(
            has_smells=True,
            report_details=prescription.report_details,
            detected_smells=detected_smells,
        )

    def _run_prescription_crew(self, code: str, measurements: str) -> _SmellPrescriptionSchema:
        """Esegue il Crew e ritorna l'output strutturato; isola la chiamata all'LLM.

        Punto di mock nei test unitari: tutta la logica a valle passa di qui, cosi' i test non
        istanziano mai un vero Agent/Crew/LLM.
        """
        task = Task(
            description=_TASK_DESCRIPTION_TEMPLATE.format(
                mechanics=_MECHANICS,
                examples=_EXAMPLES,
                code=code,
                measurements=measurements,
            ),
            expected_output=_EXPECTED_OUTPUT,
            agent=self._agent,
            output_pydantic=_SmellPrescriptionSchema,
        )
        crew = Crew(agents=[self._agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()

        parsed = result.pydantic
        if not isinstance(parsed, _SmellPrescriptionSchema):
            raise RuntimeError(
                "Il DetectorAgent non ha prodotto un output conforme a "
                f"_SmellPrescriptionSchema. Output grezzo ricevuto dal modello: {result.raw!r}"
            )
        return parsed


def _format_measurements(long_circuit: dict, idle_qubits: dict, detected_smells: list) -> str:
    """Compone il blocco di misure iniettato nel prompt.

    Include il BERSAGLIO gia' calcolato (quale l porta l*c sotto soglia) invece di lasciarlo
    derivare al modello: e' l'aritmetica che ha gia' sbagliato in passato, e qui e' esatta.
    """
    lines = [
        f"- l (max operations on one qubit) = {long_circuit['maxOpsPerQubit']}",
        f"- c (max operations in one time step) = {long_circuit['maxParallelOps']}",
        f"- l * c = {long_circuit['value']}  (threshold: smelly at >= {LC_PRODUCT_CUTOFF})",
        f"- qubits currently holding the maximum l: {long_circuit['maxOpsQubits']}",
        f"- IdQ (longest wait) = {idle_qubits['value']}  "
        f"(threshold: smelly at > {IDLE_QUBITS_CUTOFF})",
    ]
    if idle_qubits["worstQubit"] is not None:
        lines.append(f"- qubit with the longest wait: q{idle_qubits['worstQubit']}")
    lines.append(f"- classified as: {', '.join(detected_smells)}")

    if QuantumSmellType.LONG_CIRCUIT.value in detected_smells:
        parallel = long_circuit["maxParallelOps"]
        target_l = (LC_PRODUCT_CUTOFF - 1) // parallel if parallel else 0
        to_remove = max(long_circuit["maxOpsPerQubit"] - target_l, 0)
        lines.append(
            f"- TARGET: with c = {parallel}, l must drop to {target_l} or below, so {to_remove} "
            f"operations must be removed from EACH qubit in {long_circuit['maxOpsQubits']}"
        )
    return "\n".join(lines)


def _clean_report(long_circuit: dict, idle_qubits: dict) -> str:
    """Report deterministico per i circuiti puliti: nessuna chiamata LLM."""
    return (
        f"No smell detected. l*c = {long_circuit['value']} "
        f"(threshold {LC_PRODUCT_CUTOFF}), IdQ = {idle_qubits['value']} "
        f"(threshold {IDLE_QUBITS_CUTOFF})."
    )
