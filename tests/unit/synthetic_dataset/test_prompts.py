"""Unit test dei lotti di generazione e dei few-shot che li accompagnano.

Questi test non guardano il testo del prompt (che cambia spesso e non e' un contratto) ma due
proprieta' che invece lo sono, e che si romperebbero in silenzio se qualcuno ritoccasse un
intervallo in BATCH_THEMES:

1. l'intervallo (l, c) di ogni lotto deve cadere INTERAMENTE da un lato di LC_PRODUCT_CUTOFF,
   altrimenti l'etichetta prodotta dal lotto diventa un caso fortunato invece che una
   conseguenza della forma richiesta;
2. l'esempio che ogni lotto vede per primo deve avere la forma che quel lotto chiede -- dove un
   esempio reale con quella forma esiste. Dove non esiste (le tre forme di Long Circuit assenti
   dal corpus) il disallineamento e' noto, misurato e ATTESO dal test, cosi' che resti visibile
   invece di essere dimenticato.

Le misure passano dalla QiskitFacade reale, come gia' fanno i test in tests/unit/qcep/: qui
mockarla significherebbe verificare i numeri che il test stesso si e' inventato.
"""

import contextlib
import io
import re

import pytest

from qscsop_pipeline.common.qiskit_facade.implementations.qiskit_facade import QiskitFacade
from scripts.synthetic_dataset import prompts
from scripts.synthetic_dataset.prompts import (
    LC_PRODUCT_CUTOFF,
    BATCH_THEMES,
    IdleTarget,
    build_batch_prompt,
)
from scripts.synthetic_dataset.verification import matches_batch_theme, measure_shape


@pytest.fixture(scope="module")
def facade() -> QiskitFacade:
    return QiskitFacade()


def _measure(facade: QiskitFacade, source_code: str) -> dict:
    # isolate_circuit ESEGUE il sorgente e i circuiti reali stampano spesso il disegno del
    # circuito: senza soppressione l'output del test diventa illeggibile.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return measure_shape(source_code, facade)


@pytest.mark.unit
@pytest.mark.parametrize("theme", BATCH_THEMES, ids=lambda t: t.theme)
def test_every_batch_range_falls_entirely_on_one_side_of_the_cutoff(theme) -> None:
    """Con un intervallo a cavallo della soglia il lotto produrrebbe etichette miste."""
    smallest = theme.l_range[0] * theme.c_range[0]
    largest = theme.l_range[1] * theme.c_range[1]

    always_long = smallest >= LC_PRODUCT_CUTOFF
    never_long = largest < LC_PRODUCT_CUTOFF

    assert always_long or never_long, (
        f"{theme.theme}: l*c va da {smallest} a {largest}, a cavallo della soglia "
        f"{LC_PRODUCT_CUTOFF} -- l'etichetta del lotto non sarebbe piu' determinata dalla forma."
    )
    assert theme.expects_long_circuit is always_long


@pytest.mark.unit
def test_exactly_one_batch_targets_clean_circuits() -> None:
    """I circuiti puliti sono la classe negativa: servono, ma un solo lotto deve produrli."""
    clean_batches = [theme for theme in BATCH_THEMES if theme.targets_clean_circuits]

    assert len(clean_batches) == 1
    assert clean_batches[0].idle_target == IdleTarget.NONE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("theme_name", "example"),
    [
        ("idq_short_wait", "_REAL_IDQ_SHORT_WAIT"),
        ("idq_long_wait", "_REAL_IDQ_LONG_WAIT"),
        ("both_smells", "_REAL_BOTH_SMELLS"),
        ("clean_mixed", "_REAL_CLEAN_SEQUENTIAL"),
    ],
)
def test_batch_is_shown_an_example_that_satisfies_its_own_constraint(
    facade: QiskitFacade, theme_name: str, example: str
) -> None:
    """Quattro lotti su sette hanno un esempio reale che centra il loro bersaglio: resti cosi'."""
    theme = next(t for t in BATCH_THEMES if t.theme == theme_name)
    shape = _measure(facade, getattr(prompts, example))

    assert matches_batch_theme(shape, theme), (
        f"{theme_name}: l'esempio misura l={shape['l']} c={shape['c']} IdQ={shape['idq']}, "
        f"fuori dal bersaglio l{theme.l_range} c{theme.c_range} {theme.idle_target.value}."
    )


@pytest.mark.unit
def test_the_deep_narrow_example_misses_only_on_magnitude(facade: QiskitFacade) -> None:
    """Disallineamento NOTO: il corpus non ha un LC profondo con l fra 20 e 40, solo uno da 1002.

    Il prompt lo compensa con un avviso esplicito sulla magnitudine. Il test lo fissa perche' se
    un giorno comparisse un esempio della taglia giusta, quell'avviso andrebbe tolto.
    """
    theme = next(t for t in BATCH_THEMES if t.theme == "lc_deep_narrow")
    shape = _measure(facade, prompts._REAL_LC_DEEP_NARROW)

    assert theme.c_range[0] <= shape["c"] <= theme.c_range[1]
    assert shape["idq"] == 0
    assert shape["l"] > theme.l_range[1]


@pytest.mark.unit
def test_the_balanced_example_misses_only_on_idling(facade: QiskitFacade) -> None:
    """Disallineamento NOTO: l'unico LC bilanciato reale porta anche Idle Qubits.

    Il prompt lo compensa chiedendo esplicitamente "quella forma, senza il suo idling".
    """
    theme = next(t for t in BATCH_THEMES if t.theme == "lc_balanced")
    shape = _measure(facade, prompts._REAL_BOTH_SMELLS)

    assert theme.l_range[0] <= shape["l"] <= theme.l_range[1]
    assert theme.c_range[0] <= shape["c"] <= theme.c_range[1]
    assert shape["idq"] > 0


@pytest.mark.unit
@pytest.mark.parametrize("theme", BATCH_THEMES, ids=lambda t: t.theme)
def test_each_batch_sees_its_own_profile_first(theme) -> None:
    """La posizione conta in un prompt lungo: l'esempio del bersaglio non deve finire in fondo."""
    prompt = build_batch_prompt(theme)
    first_example = prompt[
        prompt.index("Example 1 -- ") : prompt.index("\n", prompt.index("Example 1 -- "))
    ]

    if theme.targets_clean_circuits:
        assert "CLEAN" in first_example
    elif theme.expects_long_circuit and theme.idle_target == IdleTarget.PRESENT:
        assert "LONG CIRCUIT + IDLE QUBITS" in first_example
    elif theme.expects_long_circuit:
        assert first_example.startswith("Example 1 -- LONG CIRCUIT alone")
    else:
        assert "IDLE QUBITS alone" in first_example


@pytest.mark.unit
def test_smelly_batches_are_not_shown_clean_circuits_as_examples() -> None:
    """La regressione da cui nasce build_batch_prompt: un prompt smelly pieno di esempi puliti."""
    for theme in BATCH_THEMES:
        if theme.targets_clean_circuits:
            continue
        prompt = build_batch_prompt(theme)
        # Solo le INTESTAZIONI: la stringa "Example N" compare anche dentro le spiegazioni e
        # nelle istruzioni di lotto, che vi rimandano.
        headings = re.findall(r"^Example \d+ -- (.+):$", prompt, re.MULTILINE)

        assert len(headings) == 4
        assert not any(heading.startswith("CLEAN") for heading in headings)
        assert "TWO WAYS TO MISS THE TARGET" in prompt
