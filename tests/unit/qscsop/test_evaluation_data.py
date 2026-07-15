import pytest

from qscsop_pipeline.qscsop.entities.evaluation_data import EvaluationData


@pytest.mark.unit
def test_default_state() -> None:
    evaluation = EvaluationData()

    assert evaluation.get_status() == "PROCESSING"
    assert evaluation.get_is_functionally_equivalent() is None
    assert evaluation.get_iteration_count() == 0
    assert evaluation.get_detected_smells() == []


@pytest.mark.unit
def test_setters_update_fields() -> None:
    evaluation = EvaluationData()

    evaluation.set_status("OPT_FAILED")
    evaluation.set_iteration_count(4)
    evaluation.set_is_functionally_equivalent(False)
    evaluation.set_detected_smells(["LongCircuit"])

    assert evaluation.get_status() == "OPT_FAILED"
    assert evaluation.get_iteration_count() == 4
    assert evaluation.get_is_functionally_equivalent() is False
    assert evaluation.get_detected_smells() == ["LongCircuit"]


@pytest.mark.unit
def test_increment_iteration_count_three_times() -> None:
    evaluation = EvaluationData()

    evaluation.increment_iteration_count()
    evaluation.increment_iteration_count()
    evaluation.increment_iteration_count()

    assert evaluation.get_iteration_count() == 3


@pytest.mark.unit
def test_update_result_updates_only_equivalence_and_status() -> None:
    evaluation = EvaluationData()
    evaluation.set_iteration_count(2)
    evaluation.set_detected_smells(["IdleQubits"])

    evaluation.update_result(True, "OPTIMIZED")

    assert evaluation.get_is_functionally_equivalent() is True
    assert evaluation.get_status() == "OPTIMIZED"
    assert evaluation.get_iteration_count() == 2
    assert evaluation.get_detected_smells() == ["IdleQubits"]


@pytest.mark.unit
def test_two_instances_do_not_share_detected_smells_list() -> None:
    first = EvaluationData()
    second = EvaluationData()

    # Manipola direttamente la lista interna (non tramite setter/getter, che gia' copiano)
    # per accertarsi che __init__ non condivida un default mutabile a livello di classe.
    first._detected_smells.append("LongCircuit")

    assert second.get_detected_smells() == []


@pytest.mark.unit
def test_get_detected_smells_returns_a_copy() -> None:
    evaluation = EvaluationData()
    evaluation.set_detected_smells(["LongCircuit"])

    retrieved = evaluation.get_detected_smells()
    retrieved.append("IdleQubits")

    assert evaluation.get_detected_smells() == ["LongCircuit"]


@pytest.mark.unit
def test_set_detected_smells_copies_the_given_list() -> None:
    evaluation = EvaluationData()
    external_list = ["LongCircuit"]

    evaluation.set_detected_smells(external_list)
    external_list.append("IdleQubits")

    assert evaluation.get_detected_smells() == ["LongCircuit"]
