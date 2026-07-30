from unittest.mock import Mock

import pytest

from qscsop_pipeline.qscsop.entities.circuit_metrics import CircuitMetrics
from qscsop_pipeline.qscsop.entities.circuit_version import CircuitVersion
from qscsop_pipeline.qscsop.entities.evaluation_status import EvaluationStatus
from qscsop_pipeline.qscsop.entities.quantum_program_entity import QuantumProgramEntity
from qscsop_pipeline.qscsop.mas.dto.failure_reason import FailureReason
from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO
from qscsop_pipeline.qscsop.mas.dto.validation_result_dto import ValidationResultDTO
from qscsop_pipeline.qscsop.mas.interfaces.i_detector_agent import IDetectorAgent
from qscsop_pipeline.qscsop.mas.interfaces.i_refactorer_agent import IRefactorerAgent
from qscsop_pipeline.qscsop.mas.interfaces.i_reviewer_agent import IReviewerAgent
from qscsop_pipeline.qscsop.mas.interfaces.i_validation_service import IValidationService
from qscsop_pipeline.qscsop.mas.mas_engine import MASEngine

BASELINE_CODE = "qc = QuantumCircuit(1)\nqc.h(0)\nqc.z(0)\nqc.h(0)\n"

NEW_METRICS = {
    "logicalQubits": 1,
    "abstractMetrics": {"gateCount": 1, "depth": 1},
    "physicalMetrics": {"gateCount": 2, "depth": 2},
}


def _make_entity() -> QuantumProgramEntity:
    baseline = CircuitVersion(
        source_code=BASELINE_CODE,
        logical_qubits=1,
        abstract_metrics=CircuitMetrics(gate_count=3, depth=3),
        physical_metrics=CircuitMetrics(gate_count=5, depth=5),
    )
    return QuantumProgramEntity(circuit_id="c1", dataset_source="ds1", baseline=baseline)


def _make_collaborators() -> dict:
    return {
        "detector_agent": Mock(spec=IDetectorAgent),
        "refactorer_agent": Mock(spec=IRefactorerAgent),
        "validation_service": Mock(spec=IValidationService),
        "reviewer_agent": Mock(spec=IReviewerAgent),
    }


@pytest.mark.unit
def test_process_entity_smell_free_skips_refactoring_cycle() -> None:
    collaborators = _make_collaborators()
    collaborators["detector_agent"].detect_smell.return_value = SmellReportDTO(
        has_smells=False, report_details="nessuno smell rilevato", detected_smells=[]
    )
    engine = MASEngine(max_iterations=3, **collaborators)
    entity = _make_entity()

    result = engine.process_entity(entity)

    assert result.get_evaluation().get_status() == EvaluationStatus.SMELL_FREE
    assert result.get_refactored() is None
    assert result.get_evaluation().get_detected_smells() == []
    assert result.get_evaluation().get_failure_reason() is None
    collaborators["refactorer_agent"].refactor.assert_not_called()
    collaborators["validation_service"].validate.assert_not_called()
    collaborators["reviewer_agent"].review.assert_not_called()


@pytest.mark.unit
def test_process_entity_succeeds_on_first_attempt() -> None:
    collaborators = _make_collaborators()
    smell_report = SmellReportDTO(
        has_smells=True, report_details="long circuit", detected_smells=["long_circuit"]
    )
    collaborators["detector_agent"].detect_smell.return_value = smell_report
    collaborators["refactorer_agent"].refactor.return_value = "qc.x(0)\n"
    collaborators["validation_service"].validate.return_value = ValidationResultDTO(
        is_valid=True, raw_error_data=None, new_metrics=NEW_METRICS
    )
    engine = MASEngine(max_iterations=3, **collaborators)
    entity = _make_entity()

    result = engine.process_entity(entity)

    assert result.get_evaluation().get_status() == EvaluationStatus.OPTIMIZED
    assert result.get_evaluation().get_is_functionally_equivalent() is True
    assert result.get_evaluation().get_iteration_count() == 1
    assert result.get_evaluation().get_failure_reason() is None
    refactored = result.get_refactored()
    assert refactored is not None
    assert refactored.get_source_code() == "qc.x(0)\n"
    assert refactored.get_logical_qubits() == 1
    assert refactored.get_abstract_metrics().get_gate_count() == 1
    assert refactored.get_abstract_metrics().get_depth() == 1
    assert refactored.get_physical_metrics().get_gate_count() == 2
    assert refactored.get_physical_metrics().get_depth() == 2
    collaborators["reviewer_agent"].review.assert_not_called()


@pytest.mark.unit
def test_process_entity_fails_once_then_succeeds() -> None:
    collaborators = _make_collaborators()
    smell_report = SmellReportDTO(
        has_smells=True, report_details="long circuit", detected_smells=["long_circuit"]
    )
    collaborators["detector_agent"].detect_smell.return_value = smell_report
    collaborators["refactorer_agent"].refactor.side_effect = ["attempt_1", "attempt_2"]
    invalid_result = ValidationResultDTO(
        is_valid=False, raw_error_data="non equivalente", new_metrics=None
    )
    valid_result = ValidationResultDTO(
        is_valid=True, raw_error_data=None, new_metrics=NEW_METRICS
    )
    collaborators["validation_service"].validate.side_effect = [invalid_result, valid_result]
    collaborators["reviewer_agent"].review.return_value = "correggi l'equivalenza"
    engine = MASEngine(max_iterations=3, **collaborators)
    entity = _make_entity()

    result = engine.process_entity(entity)

    assert result.get_evaluation().get_status() == EvaluationStatus.OPTIMIZED
    assert result.get_evaluation().get_iteration_count() == 2
    assert result.get_evaluation().get_failure_reason() is None
    assert collaborators["refactorer_agent"].refactor.call_count == 2
    first_call_args = collaborators["refactorer_agent"].refactor.call_args_list[0].args
    second_call_args = collaborators["refactorer_agent"].refactor.call_args_list[1].args
    assert first_call_args[2] == ""
    assert second_call_args[2] == "correggi l'equivalenza"
    collaborators["reviewer_agent"].review.assert_called_once_with(
        invalid_result, smell_report, "attempt_1"
    )


@pytest.mark.unit
def test_process_entity_exhausts_iterations() -> None:
    collaborators = _make_collaborators()
    smell_report = SmellReportDTO(
        has_smells=True, report_details="long circuit", detected_smells=["long_circuit"]
    )
    collaborators["detector_agent"].detect_smell.return_value = smell_report
    collaborators["refactorer_agent"].refactor.return_value = "attempt"
    invalid_result = ValidationResultDTO(
        is_valid=False,
        raw_error_data="non equivalente",
        new_metrics=None,
        failure_reason=FailureReason.NOT_EQUIVALENT,
    )
    collaborators["validation_service"].validate.return_value = invalid_result
    collaborators["reviewer_agent"].review.return_value = "correggi"
    engine = MASEngine(max_iterations=2, **collaborators)
    entity = _make_entity()

    result = engine.process_entity(entity)

    assert collaborators["refactorer_agent"].refactor.call_count == 2
    assert collaborators["reviewer_agent"].review.call_count == 1
    assert result.get_evaluation().get_status() == EvaluationStatus.OPT_FAILED
    assert result.get_evaluation().get_is_functionally_equivalent() is False
    assert result.get_refactored() is None
    assert result.get_evaluation().get_iteration_count() == 2
    # failure_reason dell'entita' deve coincidere con quello dell'ULTIMO tentativo (quello che
    # ha fatto uscire dal loop per esaurimento iterazioni), non un valore hardcoded in MASEngine.
    assert result.get_evaluation().get_failure_reason() == invalid_result.get_failure_reason()
    assert result.get_evaluation().get_failure_reason() == FailureReason.NOT_EQUIVALENT


@pytest.mark.unit
def test_process_entity_never_propagates_unexpected_exception() -> None:
    collaborators = _make_collaborators()
    collaborators["detector_agent"].detect_smell.side_effect = RuntimeError("errore imprevisto")
    engine = MASEngine(max_iterations=3, **collaborators)
    entity = _make_entity()

    result = engine.process_entity(entity)

    assert result.get_evaluation().get_status() == EvaluationStatus.OPT_FAILED
    assert result.get_evaluation().get_is_functionally_equivalent() is False
    assert result.get_evaluation().get_failure_reason() == FailureReason.UNEXPECTED_ERROR


@pytest.mark.unit
def test_process_entity_returns_same_instance() -> None:
    collaborators = _make_collaborators()
    collaborators["detector_agent"].detect_smell.return_value = SmellReportDTO(
        has_smells=False, report_details="nessuno smell rilevato", detected_smells=[]
    )
    engine = MASEngine(max_iterations=3, **collaborators)
    entity = _make_entity()

    result = engine.process_entity(entity)

    assert result is entity
