"""Unit test del DetectorAgent ibrido.

La facade e' MOCKATA (spec=IQiskitFacade), come nei test di ValidationService: qui si verifica
la logica di decisione dell'agente -- chi decide cosa, quando l'LLM viene invocato e cosa vede --
non il comportamento di Qiskit, gia' coperto dai test su QiskitFacade.

I test sono stati riscritti insieme all'agente. Prima verificavano la mappatura di un verdetto
prodotto dal MODELLO in SmellReportDTO; ora il verdetto non e' piu' del modello, e i test che
contano sono due: che l'LLM non venga sfiorato su un circuito pulito, e che detected_smells
derivi dalla misura e non da cio' che il modello scrive.
"""

from unittest.mock import Mock

import pytest
from crewai import BaseLLM

from qscsop_pipeline.common.qiskit_facade.interfaces.i_qiskit_facade import IQiskitFacade
from qscsop_pipeline.qscsop.mas.agents.detector_agent import (
    DetectorAgent,
    _SmellPrescriptionSchema,
)
from qscsop_pipeline.qscsop.mas.detection_thresholds import IDLE_QUBITS_CUTOFF, LC_PRODUCT_CUTOFF
from qscsop_pipeline.qscsop.mas.dto.quantum_smell_type import QuantumSmellType

_SAMPLE_CODE = "qc.h(0)\nqc.z(0)\nqc.h(0)\n"


def _metrics(
    *,
    lc: int,
    idq: int,
    max_ops: int = 7,
    max_parallel: int = 5,
    holders=(0,),
    timelines=("h, cx, h, cx", "cx, _, cx, _"),
) -> dict:
    return {
        "longCircuit": {
            "maxOpsPerQubit": max_ops,
            "maxParallelOps": max_parallel,
            "maxOpsQubits": list(holders),
            "timelinePerQubit": list(timelines),
            "value": lc,
            "gateError": 0.00485,
            "errorFreeProbability": 0.84,
        },
        "idleQubits": {"value": idq, "worstQubit": 0 if idq else None},
    }


def _make_agent(metrics: dict, mocker) -> DetectorAgent:
    """DetectorAgent con facade mockata; Agent di CrewAI neutralizzato (mai istanziato davvero)."""
    mocker.patch("qscsop_pipeline.qscsop.mas.agents.detector_agent.Agent")
    facade = Mock(spec=IQiskitFacade)
    facade.calculate_smell_metrics.return_value = metrics
    return DetectorAgent(llm=Mock(spec=BaseLLM), facade=facade)


# ------------------------------------------------------------------------------------------
# Chi decide: le soglie sulla misura, non il modello.
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_a_clean_circuit_never_reaches_the_llm(mocker) -> None:
    """IL TEST CHE GIUSTIFICA IL DISEGNO IBRIDO.

    Su un circuito pulito non c'e' nulla da prescrivere, quindi non c'e' motivo di pagare una
    chiamata al modello: sul dataset sintetico sono 24 circuiti su 72. Il report e' generato
    deterministicamente dai valori misurati, ed e' piu' informativo di una frase inventata.
    """
    agent = _make_agent(_metrics(lc=LC_PRODUCT_CUTOFF - 1, idq=IDLE_QUBITS_CUTOFF), mocker)
    crew = mocker.patch.object(agent, "_run_prescription_crew")

    report = agent.detect_smell(_SAMPLE_CODE)

    assert report.get_has_smells() is False
    assert report.get_detected_smells() == []
    crew.assert_not_called()
    assert str(LC_PRODUCT_CUTOFF - 1) in report.get_report_details()


@pytest.mark.unit
def test_detected_smells_comes_from_the_measurement_not_from_the_model(mocker) -> None:
    """Il modello scrive una prescrizione, non un verdetto.

    Qui la prescrizione non nomina alcuno smell: l'etichetta arriva comunque, perche' la
    decidono le soglie. Un output del modello che contraddica la misura non e' rappresentabile.
    """
    agent = _make_agent(_metrics(lc=35, idq=3), mocker)
    mocker.patch.object(
        agent,
        "_run_prescription_crew",
        return_value=_SmellPrescriptionSchema(
            report_details="Rimuovi le righe 4 e 5.", repairable=True
        ),
    )

    report = agent.detect_smell(_SAMPLE_CODE)

    assert report.get_has_smells() is True
    assert report.get_detected_smells() == [
        QuantumSmellType.LONG_CIRCUIT.value,
        QuantumSmellType.IDLE_QUBITS.value,
    ]
    assert report.get_report_details() == "Rimuovi le righe 4 e 5."


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lc", "idq", "expected"),
    [
        (35, 0, [QuantumSmellType.LONG_CIRCUIT.value]),
        (9, 3, [QuantumSmellType.IDLE_QUBITS.value]),
        (
            LC_PRODUCT_CUTOFF,
            IDLE_QUBITS_CUTOFF + 1,
            [QuantumSmellType.LONG_CIRCUIT.value, QuantumSmellType.IDLE_QUBITS.value],
        ),
    ],
    ids=["solo_long_circuit", "solo_idle_qubits", "entrambi_esattamente_al_taglio"],
)
def test_each_threshold_drives_its_own_label(mocker, lc, idq, expected) -> None:
    agent = _make_agent(_metrics(lc=lc, idq=idq), mocker)
    mocker.patch.object(
        agent,
        "_run_prescription_crew",
        return_value=_SmellPrescriptionSchema(report_details="...", repairable=True),
    )

    assert agent.detect_smell(_SAMPLE_CODE).get_detected_smells() == expected


# ------------------------------------------------------------------------------------------
# Cosa vede il modello: i numeri esatti piu' il bersaglio gia' calcolato.
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_prompt_carries_the_qubits_holding_the_maximum(mocker) -> None:
    """Senza questa lista la prescrizione e' inutile quasi una volta su due.

    Togliere operazioni da un qubit che non e' al massimo non abbassa l: il massimo resta,
    tenuto dagli altri. Sui 72 circuiti del dataset il massimo e' condiviso in 34 casi.
    """
    agent = _make_agent(
        _metrics(lc=40, idq=0, max_ops=10, max_parallel=4, holders=(0, 2, 3)), mocker
    )
    crew = mocker.patch.object(
        agent,
        "_run_prescription_crew",
        return_value=_SmellPrescriptionSchema(report_details="...", repairable=True),
    )

    agent.detect_smell(_SAMPLE_CODE)

    measurements = crew.call_args.args[1]
    assert "[0, 2, 3]" in measurements


@pytest.mark.unit
def test_the_prompt_carries_the_precomputed_target(mocker) -> None:
    """Il bersaglio e' calcolato qui, non lasciato derivare al modello.

    Con c = 4 serve l <= (20-1)//4 = 4, quindi da un qubit a l = 10 vanno tolte 6 operazioni.
    E' l'aritmetica che il modello ha gia' sbagliato in passato, e qui e' esatta.
    """
    agent = _make_agent(_metrics(lc=40, idq=0, max_ops=10, max_parallel=4), mocker)
    crew = mocker.patch.object(
        agent,
        "_run_prescription_crew",
        return_value=_SmellPrescriptionSchema(report_details="...", repairable=True),
    )

    agent.detect_smell(_SAMPLE_CODE)

    measurements = crew.call_args.args[1]
    assert "l must drop to 4 or below" in measurements
    assert "6 operations must be removed" in measurements


@pytest.mark.unit
def test_the_idle_qubit_pointer_is_passed_only_when_there_is_a_wait(mocker) -> None:
    agent = _make_agent(_metrics(lc=35, idq=0), mocker)
    crew = mocker.patch.object(
        agent,
        "_run_prescription_crew",
        return_value=_SmellPrescriptionSchema(report_details="...", repairable=True),
    )

    agent.detect_smell(_SAMPLE_CODE)

    assert "longest wait: q" not in crew.call_args.args[1]


@pytest.mark.unit
def test_the_measurement_is_taken_once_on_the_code_under_analysis(mocker) -> None:
    agent = _make_agent(_metrics(lc=35, idq=3), mocker)
    mocker.patch.object(
        agent,
        "_run_prescription_crew",
        return_value=_SmellPrescriptionSchema(report_details="...", repairable=True),
    )

    agent.detect_smell(_SAMPLE_CODE)

    agent._facade.calculate_smell_metrics.assert_called_once_with(_SAMPLE_CODE)


@pytest.mark.unit
def test_a_measurement_failure_is_not_swallowed(mocker) -> None:
    """calculate_smell_metrics ESEGUE il sorgente e puo' fallire: il MASEngine ha gia' la sua
    rete (process_entity cattura e produce uno stato terminale), quindi qui non si ingoia nulla.
    """
    mocker.patch("qscsop_pipeline.qscsop.mas.agents.detector_agent.Agent")
    facade = Mock(spec=IQiskitFacade)
    facade.calculate_smell_metrics.side_effect = ValueError("nessun QuantumCircuit assegnato")
    agent = DetectorAgent(llm=Mock(spec=BaseLLM), facade=facade)

    with pytest.raises(ValueError):
        agent.detect_smell(_SAMPLE_CODE)


# ------------------------------------------------------------------------------------------
# La sequenza eseguita e il verdetto di riparabilita'.
# ------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_prompt_carries_the_executed_operation_sequence(mocker) -> None:
    """IL PONTE FRA SORGENTE E CIRCUITO ESEGUITO.

    Ricevendo i soli conteggi, il modello immaginava uno srotolamento inesistente: su un
    sorgente di 8 righe che costruisce 21 operazioni con un for, prescriveva di rimuovere
    "lines 2-3 ... 24-25". La sequenza rende visibile cosa il circuito fa davvero.
    """
    agent = _make_agent(
        _metrics(
            lc=42, idq=1, max_ops=21, max_parallel=2, timelines=("h, cx, h, cx", "cx, _, cx, _")
        ),
        mocker,
    )
    crew = mocker.patch.object(
        agent,
        "_run_prescription_crew",
        return_value=_SmellPrescriptionSchema(report_details="...", repairable=True),
    )

    agent.detect_smell(_SAMPLE_CODE)

    measurements = crew.call_args.args[1]
    assert "q0: h, cx, h, cx" in measurements
    assert "q1: cx, _, cx, _" in measurements
    # E l'avvertenza che quelle operazioni possono venire da un loop.
    assert "loop" in measurements


@pytest.mark.unit
def test_an_unrepairable_circuit_keeps_its_smells_but_is_flagged(mocker) -> None:
    """repairable=False non e' "nessuno smell": lo smell c'e', misurato.

    E' un circuito sopra soglia per sola dimensione. Il flag arriva al MASEngine tramite il DTO,
    ed e' li' che decide di non entrare nel ciclo.
    """
    agent = _make_agent(_metrics(lc=40, idq=0), mocker)
    mocker.patch.object(
        agent,
        "_run_prescription_crew",
        return_value=_SmellPrescriptionSchema(
            report_details="Nessuna ridondanza rimovibile.", repairable=False
        ),
    )

    report = agent.detect_smell(_SAMPLE_CODE)

    assert report.get_has_smells() is True
    assert report.get_detected_smells() == [QuantumSmellType.LONG_CIRCUIT.value]
    assert report.get_repairable() is False


@pytest.mark.unit
def test_a_clean_circuit_is_reported_repairable_by_default(mocker) -> None:
    """Sul ramo pulito l'LLM non viene invocato, quindi nessuno valorizza il flag: resta True.

    Non ha conseguenze -- has_smells=False fa uscire il MASEngine prima di guardarlo -- ma il
    default non deve essere False, altrimenti un circuito sano verrebbe letto come irriparabile.
    """
    agent = _make_agent(_metrics(lc=4, idq=0), mocker)
    mocker.patch.object(agent, "_run_prescription_crew")

    assert agent.detect_smell(_SAMPLE_CODE).get_repairable() is True
