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
