"""ReviewerAgent: terzo agente del MAS, contestualizza gli errori di validazione via CrewAI.

Stesso pattern di DetectorAgent/RefactorerAgent (Agent CrewAI incapsulato, Task per chiamata,
output strutturato) e stesso schema di output SEMPLICE del RefactorerAgent: nessun campo di
ragionamento (chain-of-thought), da aggiungere solo se l'e2e mostrera' problemi concreti.

A differenza degli altri due agenti l'output esposto e' una stringa libera, non un DTO: e' il
feedback testuale che il MASEngine iniettera' come review_feedback nel tentativo successivo del
RefactorerAgent (vedi sezione "Gli Agenti e le loro Responsabilita'" della tesi).

Il prompt e' in inglese: qwen2.5-coder e' orientato al codice e piu' affidabile su prompt tecnici
in inglese; il feedback generato ne eredita la lingua, coerentemente con report_details del
DetectorAgent che finisce nello stesso prompt del RefactorerAgent.
"""

from crewai import Agent, BaseLLM, Crew, Process, Task
from pydantic import BaseModel

from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO
from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO
from qscsop_pipeline.qscsop.mas.interfaces.i_reviewer_agent import IReviewerAgent


class _ReviewSchema(BaseModel):
    """Schema Pydantic interno: solo tramite per l'output strutturato di CrewAI (non esposto)."""

    contextualized_feedback: str


# Sottostringa cercata (case-insensitive) per riconoscere un fallimento di equivalenza funzionale
# TRA i due casi con new_metrics=None (compilazione ed equivalenza sono indistinguibili sulla sola
# base strutturale del DTO, dato che entrambe azzerano new_metrics): copre le forme sia inglesi
# sia italiane ("equivalence", "equivalent", "equivalente"...) senza vincolarsi a un messaggio
# d'errore specifico.
_EQUIVALENCE_ERROR_MARKER = "equivalen"

# Blocco iniettato solo quando l'errore sembra riguardare l'equivalenza funzionale. Il sospetto
# suggerito non e' teorico: con il modello locale si e' osservato empiricamente un refactoring
# Idle Qubits che, oltre a rimuovere correttamente il qubit inerte, perdeva per strada gate
# legittimi sugli altri qubit (vedi docs/report_qscsop_refactoring_equivalence.md, sezione 5).
_EQUIVALENCE_HINT_SECTION = """
ADDITIONAL HINT FOR THIS SPECIFIC FAILURE

This error indicates a loss of FUNCTIONAL EQUIVALENCE: the refactored circuit no longer computes \
the same result as the original one. A very common cause is that the previous attempt altered or \
dropped operations that were NOT part of the smell being fixed -- for example, while removing an \
idle qubit or collapsing a redundant gate sequence, it also silently deleted legitimate gates \
belonging to the other qubits. Explicitly consider this possibility in your feedback, and tell \
the next attempt to touch ONLY what the smell requires and to leave every other operation of the \
original circuit exactly as it was.
"""

# Blocco iniettato quando il fallimento e' il nodo "Migliori?" (compilazione ed equivalenza
# superate, ma nessuna metrica fisica migliorata rispetto alla baseline). Riconoscibile in modo
# STRUTTURALE (ValidationResultDTO.new_metrics != None), non testuale: e' l'unico dei tre
# fallimenti in cui il ValidationService calcola con successo le metriche del new_code prima di
# respingerlo, vedi ValidationService._is_improvement.
_METRICS_NOT_IMPROVED_HINT_SECTION = """
ADDITIONAL HINT FOR THIS SPECIFIC FAILURE

This error means the circuit compiled and remained functionally equivalent, but did not satisfy \
the improvement criterion: EVERY metric (physical gate count, physical depth, logical qubit \
count) must be less than or equal to the baseline, AND at least one must be strictly less. The \
raw failure details above include the exact baseline and refactored values for all three \
metrics -- read them carefully and determine which specific situation applies:

CASE 1 -- NO metric improved at all (all three equal, or worse, than baseline): the attempt was \
too conservative and left the circuit essentially unchanged. Tell the next attempt it MUST make \
a real, measurable structural change that removes or simplifies the reported anomaly.

CASE 2 -- SOME metric(s) improved but at least one got WORSE than the baseline (a 'near miss'): \
the previous attempt (shown above) was substantially rewritten in the right direction, but \
regressed on one specific dimension. Identify EXACTLY which metric got worse by comparing the \
numbers in the raw failure details. Tell the next attempt to KEEP the overall structure and \
approach of the previous attempt (it was mostly correct), and fix SPECIFICALLY the regressed \
dimension -- do not discard the previous attempt's approach and start over from the original \
circuit.
"""

_TASK_DESCRIPTION_TEMPLATE = """A refactoring attempt on a Qiskit quantum circuit has just FAILED \
validation. Your job is to turn the raw technical failure into feedback that the next refactoring \
attempt can actually act on.

WHAT THE REFACTORING WAS TRYING TO FIX

A detector analyzed the original circuit and reported the following Quantum Code Smell(s):
{report_details}

The refactoring attempt was supposed to remove or simplify exactly that anomaly, and nothing else, \
while keeping the circuit functionally equivalent to the original.

RAW FAILURE DETAILS (verbatim, from the deterministic validation step)
{raw_error_details}

This raw text may be a Python traceback, a short diagnostic message, or any other technical \
output: interpret whatever form it takes, and do not assume a fixed format.

THE CODE THAT WAS ATTEMPTED (and failed)
{failed_code}

This is the exact code the previous refactoring attempt produced. Refer to it explicitly in your \
feedback when relevant -- do not describe the failure only in the abstract.
{hint_section}
YOUR TASK

Do NOT simply restate the raw error. Explain, in clear and actionable terms:
1. WHAT went wrong with respect to the original intent of the fix (removing/simplifying the \
anomaly described above);
2. WHAT the next refactoring attempt should do differently to avoid the same error, while still \
producing a valid correction of the original smell.

OUTPUT FORMAT

Return ONLY the feedback text in the contextualized_feedback field. No preamble such as "Here is \
the feedback:", no Markdown headings or code fences, no restating of these instructions. The text \
is injected verbatim into the next refactoring prompt, so write it as a direct instruction to \
whoever will perform that next attempt."""

_EXPECTED_OUTPUT = (
    "A structured object with a single field contextualized_feedback containing ONLY the "
    "actionable feedback text: what went wrong with respect to the intended fix, and what the "
    "next refactoring attempt should do differently. No preamble and no Markdown formatting."
)


class ReviewerAgent(IReviewerAgent):
    """Contestualizza gli errori di validazione in feedback azionabile via un Agent CrewAI."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm
        self._agent = Agent(
            role="Quantum Refactoring Review Specialist",
            goal=(
                "Turn the raw technical error of a failed refactoring validation into clear, "
                "actionable feedback for the next refactoring attempt."
            ),
            backstory=(
                "You are a reviewer of quantum circuit refactorings who reads low-level "
                "validation failures and explains them in terms of the original optimization "
                "intent, so that the next attempt corrects the mistake without losing sight of "
                "the smell it had to remove."
            ),
            llm=llm,
            verbose=False,
        )

    def review(
        self,
        validation_result: ValidationResultDTO,
        original_smell: SmellReportDTO,
        failed_code: str,
    ) -> str:
        """Contestualizza l'esito di validazione rispetto allo smell originale; ritorna il feedback."""
        result = self._run_review_crew(validation_result, original_smell, failed_code)
        return result.contextualized_feedback

    def _run_review_crew(
        self,
        validation_result: ValidationResultDTO,
        original_smell: SmellReportDTO,
        failed_code: str,
    ) -> _ReviewSchema:
        """Esegue il Crew di review e ritorna l'output strutturato; isola la chiamata all'LLM.

        Punto di mock nei test unitari: cosi' i test non istanziano mai un vero Agent/Crew/LLM.
        """
        raw_error_details = validation_result.get_raw_error_data()

        # is_valid e' gia' False in ogni chiamata a review() (invocato solo su fallimento): la
        # sola presenza di new_metrics distingue quindi strutturalmente il fallimento "Migliori?"
        # (metriche calcolate con successo ma non migliorative, vedi Modifica 1 in
        # ValidationService) dagli altri due tipi di fallimento (compilazione, equivalenza), dove
        # new_metrics resta None per costruzione.
        is_metrics_not_improved_failure = validation_result.get_new_metrics() is not None

        if is_metrics_not_improved_failure:
            hint_section = _METRICS_NOT_IMPROVED_HINT_SECTION
        elif _EQUIVALENCE_ERROR_MARKER in raw_error_details.lower():
            # Tra i due fallimenti con new_metrics=None (compilazione, equivalenza), solo il
            # contenuto testuale li distingue: il suggerimento sui gate persi si inietta solo se
            # l'errore riguarda l'equivalenza, mai su un errore di compilazione (fuorviante).
            hint_section = _EQUIVALENCE_HINT_SECTION
        else:
            hint_section = ""

        task = Task(
            description=_TASK_DESCRIPTION_TEMPLATE.format(
                report_details=original_smell.get_report_details(),
                raw_error_details=raw_error_details,
                failed_code=failed_code,
                hint_section=hint_section,
            ),
            expected_output=_EXPECTED_OUTPUT,
            agent=self._agent,
            output_pydantic=_ReviewSchema,
        )
        crew = Crew(agents=[self._agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()

        parsed = result.pydantic
        if not isinstance(parsed, _ReviewSchema):
            raise RuntimeError(
                "Il ReviewerAgent non ha prodotto un output conforme a _ReviewSchema. "
                f"Output grezzo ricevuto dal modello: {result.raw!r}"
            )
        return parsed
