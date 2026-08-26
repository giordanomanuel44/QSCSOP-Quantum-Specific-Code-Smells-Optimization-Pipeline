"""Unit test dei lotti di generazione e dei few-shot che li accompagnano.

I test non guardano il TESTO del prompt, che cambia spesso e non e' un contratto. Guardano le
poche proprieta' che invece lo sono, e che si romperebbero in silenzio:

1. gli esempi mostrati al modello devono GIRARE con la Qiskit del progetto. Il prompt chiede
   ormai una cosa sola -- codice che compila -- e mostra otto circuiti reali come modello: un
   esempio che non gira insegnerebbe l'opposto di quello che si chiede. E' il test piu'
   importante di questo file;
2. il prompt non deve tornare a parlare di metriche. La riscrittura nasce dal fatto che il
   modello non sa contare l ne' modellare il packing (vedi il docstring di prompts.py): rimettere
   l, c o IdQ nel prompt e' esattamente la regressione da impedire;
3. ogni lotto deve vedere per primo l'esempio piu' vicino alla struttura che gli si chiede,
   perche' in un prompt lungo la posizione pesa.

I test della versione precedente sono spariti con cio' che verificavano: gli intervalli l/c dei
lotti, il loro rapporto con la soglia LC, e l'allineamento fra esempio e bersaglio metrico.
"""

import contextlib
import io
import re

import pytest

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from scripts.synthetic_dataset.prompts import (
    _EXAMPLES,
    BATCH_THEMES,
    GeneratedCircuit,
    build_batch_prompt,
)


@pytest.fixture(scope="module")
def facade() -> QiskitFacade:
    return QiskitFacade()


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(_EXAMPLES), ids=lambda k: k)
def test_every_few_shot_example_actually_runs(facade: QiskitFacade, key: str) -> None:
    """Un esempio che non compila insegnerebbe al modello proprio cio' che gli si vieta.

    Vale anche come sentinella di versione: se un aggiornamento di Qiskit rimuovesse una delle
    API usate da questi circuiti reali (e' gia' successo con execute e con providers.aer), il
    prompt continuerebbe a mostrarle come buone senza che nulla lo segnali.
    """
    # compile_circuit ESEGUE il sorgente e alcuni circuiti reali stampano: senza soppressione
    # l'output del test diventa illeggibile.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        compiles, error = facade.compile_circuit(_EXAMPLES[key][1])

    assert compiles, f"l'esempio {key} non gira: {error}"


@pytest.mark.unit
@pytest.mark.parametrize("theme", BATCH_THEMES, ids=lambda t: t.theme)
def test_the_prompt_never_mentions_the_metrics(theme) -> None:
    """La regressione da impedire: rimettere nel prompt i conteggi che il modello sbaglia.

    Il primo giro di generazione ha riportato `l` = numero di qubit in tutti gli 8 record di un
    lotto e la stessa riga di conteggi in tutti i 10 di un altro. Il compito e' stato ridotto a
    "scrivi codice che gira" proprio per togliere di mezzo quel livello: se un giorno qualcuno lo
    reintroduce, questo test lo dice.
    """
    prompt = build_batch_prompt(theme)

    for forbidden in ("IdQ", "execution matrix", "l * c", "Long Circuit", "Idle Qubits"):
        assert forbidden not in prompt, f"{theme.theme}: il prompt e' tornato a parlare di metriche"


@pytest.mark.unit
@pytest.mark.parametrize("theme", BATCH_THEMES, ids=lambda t: t.theme)
def test_each_batch_sees_its_own_example_first(theme) -> None:
    """La posizione conta in un prompt lungo: l'esempio della struttura richiesta apre la lista."""
    prompt = build_batch_prompt(theme)
    headings = re.findall(r"^Example \d+ -- (.+):$", prompt, re.MULTILINE)

    assert len(headings) == len(theme.example_keys)
    assert headings[0] == _EXAMPLES[theme.example_keys[0]][0]


@pytest.mark.unit
@pytest.mark.parametrize("theme", BATCH_THEMES, ids=lambda t: t.theme)
def test_every_batch_states_its_qubit_range_and_count(theme) -> None:
    """Il numero di qubit e' l'unico vincolo numerico che il modello ha dimostrato di rispettare."""
    prompt = build_batch_prompt(theme)

    assert f"between {theme.qubit_range[0]} and {theme.qubit_range[1]} qubits" in prompt
    assert f"exactly {theme.count} circuits" in prompt


@pytest.mark.unit
def test_the_generated_circuit_schema_asks_for_source_code_alone() -> None:
    """Il contratto col modello: un campo solo.

    Ogni campo in piu' chiesto in passato (l'etichetta, il ragionamento, lo schizzo della
    matrice) e' stato una fonte di allucinazione che nessuno leggeva a valle.
    """
    assert list(GeneratedCircuit.model_fields) == ["source_code"]


@pytest.mark.unit
@pytest.mark.parametrize("theme", BATCH_THEMES, ids=lambda t: t.theme)
def test_the_prompt_carries_the_rules_derived_from_real_crashes(theme) -> None:
    """Le due regole nate dai fallimenti misurati nel primo giro, non da prudenza generica.

    Quattro circuiti su 55 sono morti per queste due cause: una measure su un circuito senza
    bit classici, e un gate a due qubit con lo stesso indice due volte.
    """
    prompt = build_batch_prompt(theme)

    assert "duplicate bit arguments" in prompt
    assert "QuantumCircuit(3)` followed by `qc.measure(0, 0)` CRASHES" in prompt
