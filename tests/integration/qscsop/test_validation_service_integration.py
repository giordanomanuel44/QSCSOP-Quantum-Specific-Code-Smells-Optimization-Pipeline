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

# Output REALE prodotto da RefactorerAgent nell'e2e Idle Qubits (qwen2.5-coder:7b, temperature=0),
# congelato qui per poter ispezionare check_equivalence sul caso vero senza dipendere da un LLM.
# Nota: e' gia' noto essere infedele al baseline (sezione 5 del report) — vedi il test diagnostico.
_REAL_REFACTORED_IDQ_CODE = """from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
qc = QuantumCircuit(qreg_q, creg_c)
qc.h(qreg_q[0])
qc.p(pi / 2, qreg_q[0])
qc.s(qreg_q[0])
qc.barrier()
qc.p(pi / 4, qreg_q[1])
qc.z(qreg_q[1])
qc.s(qreg_q[1])
qc.measure_all(add_bits=False)
"""


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


@pytest.mark.integration
def test_check_equivalence_on_real_refactorer_output_is_diagnostic_only() -> None:
    """Ispezione (non validazione) di check_equivalence sul refactoring reale a 2 qubit.

    Caso DIAGNOSTICO: non si forza alcun esito. Un False qui e' con ogni probabilita' CORRETTO —
    il refactoring reale prodotto dal 7B ha imperfezioni di fedelta' note (gate legittimi persi
    rispetto al baseline, sezione 5 del report), quindi il confronto via partial_trace deve
    rilevarle. Un False NON e' un fallimento del meccanismo a dimensioni diverse: quello e'
    verificato dagli unit test costruiti apposta.
    """
    facade = QiskitFacade()
    baseline_code = _IDQ_SMELLY_PATH.read_text(encoding="utf-8")

    baseline_qubits = facade.isolate_circuit(baseline_code).num_qubits
    refactored_qubits = facade.isolate_circuit(_REAL_REFACTORED_IDQ_CODE).num_qubits
    is_equivalent = facade.check_equivalence(baseline_code, _REAL_REFACTORED_IDQ_CODE)

    print(
        f"\n[DIAGNOSTICA] baseline={baseline_qubits} qubit, refactored={refactored_qubits} qubit "
        f"-> check_equivalence = {is_equivalent}"
    )

    # Unico assert: il confronto a dimensioni diverse deve concludere con un booleano, senza
    # sollevare. Il valore in se' e' materiale di ispezione, non un criterio di successo.
    assert isinstance(is_equivalent, bool)
