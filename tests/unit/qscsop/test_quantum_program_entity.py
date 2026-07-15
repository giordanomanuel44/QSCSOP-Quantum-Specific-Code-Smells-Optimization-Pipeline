import pytest

from qscsop_pipeline.qscsop.entities.circuit_metrics import CircuitMetrics
from qscsop_pipeline.qscsop.entities.circuit_version import CircuitVersion
from qscsop_pipeline.qscsop.entities.evaluation_data import EvaluationData
from qscsop_pipeline.qscsop.entities.quantum_program_entity import QuantumProgramEntity


def make_version(source_code: str = "qc = QuantumCircuit(2)") -> CircuitVersion:
    return CircuitVersion(
        source_code=source_code,
        logical_qubits=2,
        abstract_metrics=CircuitMetrics(gate_count=2, depth=2),
        physical_metrics=CircuitMetrics(gate_count=4, depth=3),
    )


def make_entity() -> QuantumProgramEntity:
    return QuantumProgramEntity(
        circuit_id="bell_state",
        dataset_source="Bugs4Q",
        baseline=make_version(),
    )


@pytest.mark.unit
def test_creation_has_no_refactored_key_and_default_evaluation() -> None:
    entity = make_entity()

    result = entity.to_dict()

    assert "refactored" not in result
    assert entity.get_refactored() is None
    assert entity.get_evaluation() is not None
    assert entity.get_evaluation().get_status() == "PROCESSING"


@pytest.mark.unit
def test_setters_update_fields() -> None:
    entity = make_entity()
    new_baseline = make_version("qc = QuantumCircuit(3)")
    new_refactored = make_version("qc = QuantumCircuit(4)")
    new_evaluation = EvaluationData()

    entity.set_circuit_id("lc-smelly")
    entity.set_dataset_source("TheSmellyEight")
    entity.set_baseline(new_baseline)
    entity.set_refactored(new_refactored)
    entity.set_evaluation(new_evaluation)

    assert entity.get_circuit_id() == "lc-smelly"
    assert entity.get_dataset_source() == "TheSmellyEight"
    assert entity.get_baseline() is new_baseline
    assert entity.get_refactored() is new_refactored
    assert entity.get_evaluation() is new_evaluation


@pytest.mark.unit
def test_set_refactored_version_adds_refactored_key_to_dict() -> None:
    entity = make_entity()
    refactored = make_version("qc = QuantumCircuit(2)  # optimized")

    entity.set_refactored_version(refactored)

    result = entity.to_dict()
    assert "refactored" in result
    assert result["refactored"] == refactored.to_dict()
    assert entity.get_refactored() is refactored


@pytest.mark.unit
def test_two_instances_do_not_share_evaluation_instance() -> None:
    first = make_entity()
    second = make_entity()

    assert first.get_evaluation() is not second.get_evaluation()
