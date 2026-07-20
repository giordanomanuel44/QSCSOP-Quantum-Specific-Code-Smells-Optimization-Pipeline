"""Integration test di ValidationService con QiskitFacade REALE (zero mock) su circuiti reali.

Verifica il cablaggio ValidationService -> QiskitFacade su codice con misure terminali (la
famiglia Idle Qubits), il caso che faceva crashare check_equivalence prima del fix. Non si
asserisce is_valid=True: idq-smelly e idq-fixed sono circuiti diversi, non necessariamente
equivalenti; si verifica solo che validate() completi senza eccezioni e produca un DTO coerente.
"""

from pathlib import Path

import pytest

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.mas.validation.validation_service import ValidationService

# tests/integration/qscsop/ -> risali a root repo, poi ai file reali del dataset Idle Qubits.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "thesmellyeight"
_IDQ_SMELLY_PATH = _DATA_DIR / "idq" / "idq-smelly.py"
_IDQ_FIXED_PATH = _DATA_DIR / "idq" / "idq-fixed.py"


@pytest.mark.integration
def test_validation_service_on_real_idle_qubits_circuits_does_not_crash() -> None:
    facade = QiskitFacade()
    service = ValidationService(facade=facade)

    baseline_code = _IDQ_SMELLY_PATH.read_text(encoding="utf-8")
    new_code = _IDQ_FIXED_PATH.read_text(encoding="utf-8")

    # Il punto del test: con circuiti che misurano (measure_all / measure per qubit) validate()
    # deve completare senza sollevare eccezioni, non piu' crashare in check_equivalence.
    result = service.validate(baseline_code, new_code)

    is_valid = result.get_is_valid()
    assert isinstance(is_valid, bool)

    # Coerenza tra esito e raw_error_data: se valido nessun errore, altrimenti errore presente.
    if is_valid:
        assert result.get_raw_error_data() is None
    else:
        assert result.get_raw_error_data() is not None
