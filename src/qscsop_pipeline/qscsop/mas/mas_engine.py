"""Implementazione concreta dell'orchestratore del MAS (sezione 1.4.3, Activity Diagram Fig. 1.6)."""

import logging

from qscsop_pipeline.qscsop.entities.circuit_version import CircuitVersion
from qscsop_pipeline.qscsop.entities.smell_metrics import SmellMetrics
from qscsop_pipeline.qscsop.entities.evaluation_status import EvaluationStatus
from qscsop_pipeline.qscsop.entities.quantum_program_entity import QuantumProgramEntity
from qscsop_pipeline.qscsop.mas.dto.failure_reason import FailureReason
from qscsop_pipeline.qscsop.mas.interfaces.i_detector_agent import IDetectorAgent
from qscsop_pipeline.qscsop.mas.interfaces.i_mas_engine import IMASEngine
from qscsop_pipeline.qscsop.mas.interfaces.i_refactorer_agent import IRefactorerAgent
from qscsop_pipeline.qscsop.mas.interfaces.i_reviewer_agent import IReviewerAgent
from qscsop_pipeline.qscsop.mas.interfaces.i_validation_service import IValidationService

logger = logging.getLogger(__name__)


class MASEngine(IMASEngine):
    """Coordina detect-refactor-validate-review senza mai assumere responsabilita' di calcolo."""

    def __init__(
        self,
        max_iterations: int,
        detector_agent: IDetectorAgent,
        refactorer_agent: IRefactorerAgent,
        validation_service: IValidationService,
        reviewer_agent: IReviewerAgent,
    ) -> None:
        self._max_iterations = max_iterations
        self._detector_agent = detector_agent
        self._refactorer_agent = refactorer_agent
        self._validation_service = validation_service
        self._reviewer_agent = reviewer_agent

    def process_entity(self, entity: QuantumProgramEntity) -> QuantumProgramEntity:
        """Muta entity fino a uno stato terminale e la ritorna (stessa istanza, non una copia).

        Segue il ciclo iterativo di sezione 1.4.3 e l'Activity Diagram Fig. 1.6: rilevamento
        smell, poi (se presenti) tentativi di refactoring-validazione-review fino a successo o
        esaurimento di max_iterations. Qualunque eccezione imprevista da un collaboratore
        produce comunque un'entita' con stato terminale, mai una propagazione (stessa garanzia
        di sezione 1.4.4, gia' applicata in ValidationService).
        """
        baseline_code = entity.get_baseline().get_source_code()
        try:
            smell_report = self._detector_agent.detect_smell(baseline_code)
            entity.get_evaluation().set_detected_smells(smell_report.get_detected_smells())

            if not smell_report.get_has_smells():
                entity.get_evaluation().set_status(EvaluationStatus.SMELL_FREE)
                return entity

            review_feedback = ""
            while True:
                entity.get_evaluation().increment_iteration_count()
                refactored_code = self._refactorer_agent.refactor(
                    baseline_code, smell_report, review_feedback
                )
                validation_result = self._validation_service.validate(
                    baseline_code, refactored_code
                )

                if validation_result.get_is_valid():
                    entity.set_refactored_version(
                        self._build_refactored_version(
                            refactored_code, validation_result.get_new_metrics()
                        )
                    )
                    entity.get_evaluation().update_result(
                        is_functionally_equivalent=True, status=EvaluationStatus.OPTIMIZED
                    )
                    return entity

                if entity.get_evaluation().get_iteration_count() < self._max_iterations:
                    review_feedback = self._reviewer_agent.review(
                        validation_result, smell_report, refactored_code
                    )
                    continue

                entity.get_evaluation().update_result(
                    is_functionally_equivalent=False, status=EvaluationStatus.OPT_FAILED
                )
                entity.get_evaluation().set_failure_reason(validation_result.get_failure_reason())
                return entity
        except Exception:
            logger.exception(
                "Errore imprevisto durante process_entity per circuito %s",
                entity.get_circuit_id(),
            )
            entity.get_evaluation().update_result(
                is_functionally_equivalent=False, status=EvaluationStatus.OPT_FAILED
            )
            entity.get_evaluation().set_failure_reason(FailureReason.UNEXPECTED_ERROR)
            return entity

    @staticmethod
    def _build_refactored_version(refactored_code: str, new_metrics: dict) -> CircuitVersion:
        """Costruisce la CircuitVersion refactored dal payload di misura gia' validato.

        new_metrics e' il payload di IQiskitFacade.calculate_smell_metrics, cioe' la stessa forma
        su cui ValidationService ha appena dato il verdetto: si legge da li' invece di rimisurare,
        perche' rimisurare significherebbe poter costruire una versione con numeri diversi da
        quelli che le hanno fatto superare il controllo.
        """
        return CircuitVersion(
            source_code=refactored_code,
            smell_metrics=SmellMetrics(
                max_ops_per_qubit=new_metrics["longCircuit"]["maxOpsPerQubit"],
                max_parallel_ops=new_metrics["longCircuit"]["maxParallelOps"],
                idle_qubits=new_metrics["idleQubits"]["value"],
            ),
        )
