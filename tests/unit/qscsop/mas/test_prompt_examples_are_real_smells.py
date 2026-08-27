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
def test_the_loop_based_examples_measure_what_the_prompt_claims(facade: QiskitFacade) -> None:
    """Gli esempi 5 e 6: e' sul divario sorgente/eseguito che il modello sbagliava.

    Il 79% dei circuiti processati costruisce le proprie operazioni con un for. L'esempio 4 ha
    dei loop ma non commenta mai il divario, e nessun esempio mostrava la sequenza eseguita: su
    un sorgente di 8 righe che ne costruisce 21 operazioni il modello prescriveva di rimuovere
    "lines 2-3 ... 24-25". Questi due rendono il divario esplicito, e i numeri che citano devono
    essere quelli veri -- altrimenti il prompt insegna con dati falsi proprio l'aritmetica che il
    modello ha gia' dimostrato di non saper fare.
    """
    header = "from qiskit import QuantumCircuit\n"

    # Esempio 5: nulla da rimuovere. 7 righe di sorgente, 21 operazioni su q0.
    nothing_to_remove = header + (
        "qc = QuantumCircuit(2, 2)\n"
        "for i in range(10):\n    qc.h(0)\n    qc.cx(0, 1)\n"
        "qc.measure(0, 0)\nqc.measure(1, 1)\n"
    )
    metrics = _measure(facade, nothing_to_remove)["longCircuit"]
    assert len(nothing_to_remove.splitlines()) == 7
    assert (metrics["maxOpsPerQubit"], metrics["maxParallelOps"], metrics["value"]) == (21, 2, 42)
    assert is_long_circuit(metrics["value"])
    # Nessuna coppia adiacente identica: e' cio' che rende il circuito irriparabile.
    assert "h, h" not in metrics["operationsPerQubit"][0]

    # Esempio 6: la riparazione e' nel CORPO del loop, non a una riga inventata.
    cancelling_pair = header + (
        "qc = QuantumCircuit(2, 2)\n"
        "for i in range(12):\n    qc.x(0)\n    qc.x(0)\n    qc.cx(0, 1)\n"
        "qc.measure(0, 0)\nqc.measure(1, 1)\n"
    )
    sequence = _measure(facade, cancelling_pair)["longCircuit"]["operationsPerQubit"][0]
    assert sequence.startswith("x, x, cx, x, x, cx")
