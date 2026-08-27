"""Ogni esempio mostrato agli agenti deve davvero esibire lo smell che pretende di illustrare.

E' la sentinella della regressione che ha reso necessaria la riscrittura dei tre prompt. Quando
il progetto si e' allineato a QSMELL, i few-shot sono rimasti quelli delle definizioni semantiche
precedenti, e nessuno se n'e' accorto per settimane: misurati con la facade, l'esempio portante
del DetectorAgent (H-Z-H) risultava l*c = 3 contro una soglia di 20, e due few-shot su tre del
RefactorerAgent insegnavano a riparare circuiti che non erano smelly affatto.

Un esempio sbagliato non rompe nulla e non fa fallire alcun test: insegna in silenzio la cosa
sbagliata. Questo file esiste per rendere quel guasto rumoroso.

La facade e' quella REALE: mockarla significherebbe verificare numeri inventati dal test stesso.
"""

import contextlib
import io

import pytest

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from qscsop_pipeline.qscsop.mas.agents import refactorer_agent
from qscsop_pipeline.qscsop.mas.detection_thresholds import has_idle_qubits, is_long_circuit


@pytest.fixture(scope="module")
def facade() -> QiskitFacade:
    return QiskitFacade()


def _measure(facade: QiskitFacade, source_code: str) -> dict:
    # isolate_circuit ESEGUE il sorgente e i circuiti reali stampano spesso il disegno.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return facade.calculate_smell_metrics(source_code)


@pytest.mark.unit
def test_the_refactorer_smelly_example_really_has_idle_qubits(facade: QiskitFacade) -> None:
    """E' l'unico few-shot rimasto nel RefactorerAgent, e deve restare valido.

    Gli altri tre sono stati rimossi proprio perche' questo controllo li bocciava.
    """
    metrics = _measure(facade, refactorer_agent._IDQ_SMELLY_EXAMPLE)

    assert has_idle_qubits(metrics["idleQubits"]["value"])


@pytest.mark.unit
def test_the_refactorer_fixed_example_really_resolves_it(facade: QiskitFacade) -> None:
    """Il "dopo" deve essere effettivamente guarito, altrimenti insegna una riparazione finta."""
    before = _measure(facade, refactorer_agent._IDQ_SMELLY_EXAMPLE)
    after = _measure(facade, refactorer_agent._IDQ_FIXED_EXAMPLE)

    assert has_idle_qubits(before["idleQubits"]["value"])
    assert not has_idle_qubits(after["idleQubits"]["value"])
    # E non deve peggiorare l'altra metrica: sarebbe il baratto che il criterio Pareto respinge.
    assert after["longCircuit"]["value"] <= before["longCircuit"]["value"]


@pytest.mark.unit
def test_the_refactorer_no_longer_ships_examples_that_are_not_smelly(facade: QiskitFacade) -> None:
    """Le costanti bocciate non devono rientrare dalla finestra.

    _LC_SMELLY_EXAMPLE misurava l*c = 3 (soglia 20) e _IDQ_ILLUSTRATIVE_SMELLY_EXAMPLE misurava
    IdQ = 0: insegnavano a riparare non-smell. Quest'ultimo insegnava inoltre a RIDURRE il numero
    di qubit, strategia che sotto QSMELL non abbassa IdQ.
    """
    for removed in (
        "_LC_SMELLY_EXAMPLE",
        "_LC_FIXED_EXAMPLE",
        "_IDQ_ILLUSTRATIVE_SMELLY_EXAMPLE",
        "_IDQ_ILLUSTRATIVE_FIXED_EXAMPLE",
    ):
        assert not hasattr(refactorer_agent, removed), (
            f"{removed} e' tornato: misuralo con la facade prima di rimetterlo in un prompt."
        )


@pytest.mark.unit
def test_every_detector_example_illustrates_the_rule_it_claims(facade: QiskitFacade) -> None:
    """I primi tre few-shot del DetectorAgent, quelli sull'aritmetica di l.

    Non sono costanti separate ma casi inline dentro _EXAMPLES: qui si ricostruiscono i circuiti
    e si verifica che i numeri citati nel testo del prompt siano quelli veri. Se qualcuno ritocca
    un esempio senza rimisurarlo, il prompt inizia a citare numeri falsi.

    Gli altri tre -- il 4 e i due basati su loop -- hanno un test ciascuno qui sotto.
    """
    header = "from qiskit import QuantumCircuit\n"

    # Esempio 1: ridondanza sul qubit al massimo -- la rimozione funziona.
    before = _measure(
        facade,
        header + "qc = QuantumCircuit(2)\nqc.x(0)\nqc.x(0)\nqc.h(0)\nqc.h(1)\nqc.cx(0, 1)\n",
    )["longCircuit"]
    assert (before["maxOpsPerQubit"], before["maxParallelOps"], before["value"]) == (4, 2, 8)
    assert before["maxOpsQubits"] == [0]

    # Esempio 2: ridondanza su un qubit NON al massimo -- la metrica non si muove.
    non_max = header + (
        "qc = QuantumCircuit(2)\nqc.h(0)\nqc.x(0)\nqc.x(0)\nqc.h(1)\nqc.z(1)\nqc.s(1)\nqc.t(1)\n"
    )
    removed = header + "qc = QuantumCircuit(2)\nqc.h(0)\nqc.h(1)\nqc.z(1)\nqc.s(1)\nqc.t(1)\n"
    assert _measure(facade, non_max)["longCircuit"]["value"] == 8
    assert _measure(facade, removed)["longCircuit"]["value"] == 8

    # Esempio 3: massimo condiviso -- togliere da uno solo non basta.
    shared = header + (
        "qc = QuantumCircuit(2)\nqc.x(0)\nqc.x(0)\nqc.h(0)\n"
        "qc.x(1)\nqc.x(1)\nqc.h(1)\nqc.cx(0, 1)\n"
    )
    one = header + "qc = QuantumCircuit(2)\nqc.h(0)\nqc.x(1)\nqc.x(1)\nqc.h(1)\nqc.cx(0, 1)\n"
    both = header + "qc = QuantumCircuit(2)\nqc.h(0)\nqc.h(1)\nqc.cx(0, 1)\n"
    assert _measure(facade, shared)["longCircuit"]["maxOpsQubits"] == [0, 1]
    assert _measure(facade, one)["longCircuit"]["value"] == 8
    assert _measure(facade, both)["longCircuit"]["value"] == 4


@pytest.mark.unit
def test_the_detector_example_without_redundancy_is_really_unrepairable(
    facade: QiskitFacade,
) -> None:
    """Il quarto few-shot: sopra soglia per sola dimensione, nulla da togliere.

    E' anche l'unico esempio in cui il massimo e' tenuto da TUTTI i qubit: gli esempi 5 e 6, che
    insegnano il divario sorgente/eseguito, lavorano entrambi su due qubit soli.

    E' il rimedio al rischio principale del disegno ibrido -- l'LLM riceve la classificazione
    gia' fatta e tende a razionalizzare, inventando una ridondanza che non c'e'. Se questo
    circuito diventasse riparabile, l'esempio smetterebbe di insegnare quel permesso.
    """
    code = (
        "from qiskit import QuantumCircuit\n"
        "from numpy import pi\n"
        "qc = QuantumCircuit(4, 4)\n"
        "for layer in range(3):\n"
        "    qc.cx(0, 1)\n    qc.cz(2, 3)\n"
        "    qc.h(0)\n    qc.x(1)\n    qc.y(2)\n    qc.z(3)\n"
        "    for q in range(4):\n        qc.p(pi / 4, q)\n"
        "for q in range(4):\n    qc.measure(q, q)\n"
    )

    long_circuit = _measure(facade, code)["longCircuit"]

    assert is_long_circuit(long_circuit["value"])
    assert (long_circuit["maxOpsPerQubit"], long_circuit["maxParallelOps"]) == (10, 4)
    # Tutti e quattro i qubit al massimo: la riparazione dovrebbe toccarli tutti.
    assert long_circuit["maxOpsQubits"] == [0, 1, 2, 3]


@pytest.mark.unit
def test_the_commutation_pair_of_examples_really_behaves_as_claimed(facade: QiskitFacade) -> None:
    """GLI ESEMPI 5 E 6 SONO LA COPPIA CENTRALE DEL PROMPT: stessa forma, esito opposto.

    Il prompt sostiene che `h, cx, h, cx, ...` non si riduce mentre `z, cx, z, cx, ...` collassa
    quasi del tutto, e che nessuna adiacenza distingue i due casi. Se quell'affermazione fosse
    falsa il prompt insegnerebbe la fisica sbagliata proprio dove il modello sbagliava: sul
    dataset sintetico un `z, cx` dichiarato irriparabile era riducibile da l*c=34 a 2.
    """
    header = "from qiskit import QuantumCircuit\n"
    non_commuta = header + (
        "qc = QuantumCircuit(2, 2)\nfor i in range(10):\n    qc.h(0)\n    qc.cx(0, 1)\n"
        "qc.measure(0, 0)\nqc.measure(1, 1)\n"
    )
    commuta = header + (
        "qc = QuantumCircuit(2, 2)\nfor i in range(8):\n    qc.z(0)\n    qc.cx(0, 1)\n"
        "qc.measure(0, 0)\nqc.measure(1, 1)\n"
    )
    solo_measure = header + "qc = QuantumCircuit(2, 2)\nqc.measure(0, 0)\nqc.measure(1, 1)\n"

    h_metrics = _measure(facade, non_commuta)["longCircuit"]
    z_metrics = _measure(facade, commuta)["longCircuit"]

    # I numeri citati nel testo dei due esempi.
    assert (h_metrics["maxOpsPerQubit"], h_metrics["maxParallelOps"], h_metrics["value"]) == (
        21,
        2,
        42,
    )
    assert (z_metrics["maxOpsPerQubit"], z_metrics["maxParallelOps"], z_metrics["value"]) == (
        17,
        2,
        34,
    )
    assert is_long_circuit(h_metrics["value"]) and is_long_circuit(z_metrics["value"])

    # Nessuna adiacenza distingue i due casi: e' il punto dell'accostamento.
    assert "h, h" not in h_metrics["timelinePerQubit"][0]
    assert "z, z" not in z_metrics["timelinePerQubit"][0]

    # E l'esito e' opposto: il caso che commuta collassa alle sole measure, l'altro no.
    assert facade.check_equivalence(commuta, solo_measure) is True
    assert _measure(facade, solo_measure)["longCircuit"]["value"] == 2
    assert facade.check_equivalence(non_commuta, solo_measure) is False


@pytest.mark.unit
def test_the_timeline_shows_the_idle_steps_the_examples_print(facade: QiskitFacade) -> None:
    """Gli esempi 5 e 6 stampano la riga di q1 come `_, cx, _, cx`: dev'essere quella vera.

    Nella prima versione la riga era appiattita in `cx, cx, cx` -- una falsa adiacenza che il
    modello leggeva come otto coppie da cancellare. Se il prompt tornasse a mostrare quella
    forma, insegnerebbe di nuovo l'errore che questi esempi esistono per correggere.
    """
    commuta = (
        "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2, 2)\n"
        "for i in range(8):\n    qc.z(0)\n    qc.cx(0, 1)\n"
        "qc.measure(0, 0)\nqc.measure(1, 1)\n"
    )

    timelines = _measure(facade, commuta)["longCircuit"]["timelinePerQubit"]

    assert timelines[0].startswith("z, cx, z, cx")
    assert timelines[1].startswith("cx, _, cx, _")
    assert "cx, cx" not in timelines[1]


@pytest.mark.unit
def test_the_loop_body_example_really_carries_a_cancelling_pair(facade: QiskitFacade) -> None:
    """Esempio 7: la riparazione e' nel CORPO del loop, non a una riga inventata."""
    cancelling_pair = (
        "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2, 2)\n"
        "for i in range(12):\n    qc.x(0)\n    qc.x(0)\n    qc.cx(0, 1)\n"
        "qc.measure(0, 0)\nqc.measure(1, 1)\n"
    )

    timeline = _measure(facade, cancelling_pair)["longCircuit"]["timelinePerQubit"][0]

    assert timeline.startswith("x, x, cx, x, x, cx")
