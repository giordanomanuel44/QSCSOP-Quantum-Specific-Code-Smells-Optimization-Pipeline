import pytest

from qscsop_pipeline.qscsop.mas.dto.smell_report_dto import SmellReportDTO


@pytest.mark.unit
def test_constructor_and_getters_with_smells_detected() -> None:
    dto = SmellReportDTO(has_smells=True, report_details="Rilevato Long Circuit su 12 gate.")

    assert dto.get_has_smells() is True
    assert dto.get_report_details() == "Rilevato Long Circuit su 12 gate."


@pytest.mark.unit
def test_constructor_and_getters_without_smells() -> None:
    dto = SmellReportDTO(has_smells=False, report_details="")

    assert dto.get_has_smells() is False
    assert dto.get_report_details() == ""


@pytest.mark.unit
def test_set_has_smells_updates_field() -> None:
    dto = SmellReportDTO(has_smells=False, report_details="")

    dto.set_has_smells(True)

    assert dto.get_has_smells() is True


@pytest.mark.unit
def test_set_report_details_updates_field() -> None:
    dto = SmellReportDTO(has_smells=True, report_details="iniziale")

    dto.set_report_details("aggiornata")

    assert dto.get_report_details() == "aggiornata"


@pytest.mark.unit
def test_detected_smells_defaults_to_empty_list() -> None:
    dto = SmellReportDTO(has_smells=False, report_details="")

    assert dto.get_detected_smells() == []


@pytest.mark.unit
def test_two_instances_do_not_share_the_same_detected_smells_list() -> None:
    # Anti mutable-default: due DTO costruiti separatamente non devono condividere la stessa lista.
    first = SmellReportDTO(has_smells=False, report_details="")
    second = SmellReportDTO(has_smells=False, report_details="")

    first.set_detected_smells(["long_circuit"])

    assert second.get_detected_smells() == []


@pytest.mark.unit
def test_get_detected_smells_returns_a_copy() -> None:
    dto = SmellReportDTO(has_smells=True, report_details="", detected_smells=["long_circuit"])

    retrieved = dto.get_detected_smells()
    retrieved.append("idle_qubits")

    assert dto.get_detected_smells() == ["long_circuit"]


@pytest.mark.unit
def test_set_detected_smells_stores_a_copy() -> None:
    dto = SmellReportDTO(has_smells=False, report_details="")
    external = ["long_circuit"]

    dto.set_detected_smells(external)
    external.append("idle_qubits")

    assert dto.get_detected_smells() == ["long_circuit"]


@pytest.mark.unit
def test_repairable_defaults_to_true() -> None:
    """Un circuito e' riparabile finche' il Detector non dichiara il contrario.

    Il default conta: se fosse False, un chiamante che non conosce il campo (il ramo pulito del
    DetectorAgent, i test, i costruttori esistenti) marcherebbe come irriparabili circuiti sui
    quali il ciclo non e' mai stato nemmeno interrogato.
    """
    dto = SmellReportDTO(has_smells=True, report_details="", detected_smells=["long_circuit"])

    assert dto.get_repairable() is True


@pytest.mark.unit
def test_repairable_is_settable_and_independent_from_has_smells() -> None:
    """Sono due cose diverse: has_smells dice SE c'e' uno smell, repairable se si puo' togliere."""
    dto = SmellReportDTO(
        has_smells=True, report_details="", detected_smells=["long_circuit"], repairable=False
    )

    assert dto.get_has_smells() is True
    assert dto.get_repairable() is False

    dto.set_repairable(True)
    assert dto.get_repairable() is True
