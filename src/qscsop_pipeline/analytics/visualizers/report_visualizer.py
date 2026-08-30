"""Implementazione concreta del generatore di artefatti visivi di Analytics (sezione 1.5.3)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # nessun backend GUI: evita popup in ambienti headless/di test.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from qscsop_pipeline.analytics.interfaces.i_report_visualizer import IReportVisualizer
from qscsop_pipeline.qscsop.entities.evaluation_status import EvaluationStatus

# Palette validata dalla skill dataviz del progetto, riusata senza modifiche (nessuna nuova
# combinazione di colori introdotta, quindi nessuna nuova validazione richiesta).
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_YELLOW = "#eda100"
_STATUS_GOOD = "#0ca30c"
_STATUS_CRITICAL = "#d03b3b"
_MUTED = "#898781"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRIDLINE = "#e1e0d9"
_SURFACE = "#fcfcfb"

# Categorico: assegnazione fissa, mai ciclica. Al piu' 4 slot usati (i 4 valori di FailureReason).
_CATEGORICAL = [_BLUE, _ORANGE, _AQUA, _YELLOW]

# Palette di stato (job semantico, non identita' generica): OPTIMIZED/OPT_FAILED sono
# rispettivamente l'esito "buono" e "critico" del ciclo di ottimizzazione; SMELL_FREE non e'
# ne' un successo ne' un fallimento del MASEngine (non e' mai entrato nel ciclo), quindi resta
# neutro.
_STATUS_COLORS = {
    EvaluationStatus.OPTIMIZED.value: _STATUS_GOOD,
    EvaluationStatus.OPT_FAILED.value: _STATUS_CRITICAL,
    EvaluationStatus.SMELL_FREE.value: _MUTED,
}

_NO_DATA_TEXT = "Nessun dato disponibile"


class ReportVisualizer(IReportVisualizer):
    """Genera Report_Metriche.pdf: un unico PDF multi-pagina (sezione 1.5.3 della tesi)."""

    def __init__(self, output_path: str) -> None:
        self._output_path = Path(output_path)

    def visualize(self, df: pd.DataFrame, metrics: dict) -> None:
        """Genera le sei pagine del report e le esporta in un unico file PDF.

        df resta nella firma (fa parte di IReportVisualizer) ma non e' piu' letto: tutte le
        pagine attingono al dizionario del MetricsCalculator, che e' l'unica fonte di verita'
        sui sottoinsiemi e sui filtri di stato. Prima la distribuzione delle iterazioni
        ri-derivava il proprio filtro dal DataFrame, duplicando qui una regola che vive di
        diritto nel calcolatore.
        """
        sns.set_context("paper")
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(self._output_path) as pdf:
            self._page_core_rates(pdf, metrics)
            self._page_status_distribution(pdf, metrics)
            self._page_failure_reason_distribution(pdf, metrics)
            self._page_smell_reduction(pdf, metrics)
            self._page_threshold_recovery(pdf, metrics)
            self._page_iteration_count_distribution(pdf, metrics)

    # ---- helper condivisi ----

    @staticmethod
    def _new_figure(title: str) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(_SURFACE)
        ReportVisualizer._style_axis(ax, title)
        return fig, ax

    @staticmethod
    def _style_axis(ax: plt.Axes, title: str) -> None:
        """Stile comune a ogni asse del report, estratto per essere riusato dai pannelli
        affiancati della pagina di riduzione, che non passano da _new_figure."""
        ax.set_facecolor(_SURFACE)
        ax.set_title(title, color=_INK_PRIMARY, fontsize=13, pad=12)
        ax.tick_params(colors=_INK_SECONDARY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(_GRIDLINE)
        ax.spines["bottom"].set_color(_GRIDLINE)

    def _placeholder_page(self, pdf: PdfPages, title: str) -> None:
        """Pagina di cortesia per un sottoinsieme vuoto: mai un grafico vuoto o un crash.

        Riusata su ogni pagina il cui dato sorgente puo' essere vuoto sui pochi circuiti reali
        del dataset (non solo boxplot/histogram di Categoria 2 come minimo richiesto).
        """
        fig, ax = self._new_figure(title)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            _NO_DATA_TEXT,
            ha="center",
            va="center",
            color=_INK_SECONDARY,
            fontsize=12,
            transform=ax.transAxes,
        )
        self._save(pdf, fig)

    @staticmethod
    def _save(pdf: PdfPages, fig: plt.Figure) -> None:
        pdf.savefig(fig)
        plt.close(fig)

    @staticmethod
    def _style_count_axis(ax: plt.Axes, ylabel: str) -> None:
        ax.set_ylabel(ylabel, color=_INK_SECONDARY)
        ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)

    @staticmethod
    def _label_bars(ax: plt.Axes, bars, labels: list[str]) -> None:
        for bar, label in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                color=_INK_PRIMARY,
                fontsize=10,
            )

    # ---- pagine ----

    def _page_core_rates(self, pdf: PdfPages, metrics: dict) -> None:
        # "Equivalenza Funzionale" non compare piu': era vera per costruzione su ogni OPTIMIZED
        # (il ValidationService non ne emette senza che check_equivalence sia passato), quindi
        # non misurava nulla. Il dato utile e' il conteggio di not_equivalent nella pagina della
        # tassonomia dei fallimenti.
        candidates = [
            ("Tasso di\nRisoluzione", metrics["tasso_risoluzione"]),
            ("Tasso di Successo\nGlobale", metrics["tasso_successo_globale"]),
        ]
        available = [(label, value) for label, value in candidates if value is not None]
        title = "Metriche di Correttezza e Successo"
        if not available:
            self._placeholder_page(pdf, title)
            return

        fig, ax = self._new_figure(title)
        labels = [label for label, _ in available]
        values = [value * 100 for _, value in available]
        bars = ax.bar(labels, values, color=_BLUE, width=0.5)
        ax.set_ylim(0, 100)
        self._style_count_axis(ax, "%")
        self._label_bars(ax, bars, [f"{value:.1f}%" for value in values])
        self._save(pdf, fig)

    def _page_status_distribution(self, pdf: PdfPages, metrics: dict) -> None:
        distribution = metrics["distribuzione_stati"]
        title = "Distribuzione degli Stati Finali"
        if not distribution:
            self._placeholder_page(pdf, title)
            return

        fig, ax = self._new_figure(title)
        labels = list(distribution.keys())
        values = [distribution[label] for label in labels]
        colors = [_STATUS_COLORS.get(label, _MUTED) for label in labels]
        bars = ax.bar(labels, values, color=colors, width=0.5)
        self._style_count_axis(ax, "Numero di circuiti")
        self._label_bars(ax, bars, [str(value) for value in values])
        self._save(pdf, fig)

    def _page_failure_reason_distribution(self, pdf: PdfPages, metrics: dict) -> None:
        distribution = metrics["distribuzione_failure_reason"]
        title = "Distribuzione dei Motivi di Fallimento (OPT_FAILED)"
        if not distribution:
            self._placeholder_page(pdf, title)
            return

        fig, ax = self._new_figure(title)
        labels = list(distribution.keys())
        values = [distribution[label] for label in labels]
        colors = [_CATEGORICAL[index % len(_CATEGORICAL)] for index in range(len(labels))]
        bars = ax.bar(labels, values, color=colors, width=0.5)
        self._style_count_axis(ax, "Numero di circuiti")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        self._label_bars(ax, bars, [str(value) for value in values])
        self._save(pdf, fig)

    def _page_smell_reduction(self, pdf: PdfPages, metrics: dict) -> None:
        """Le due riduzioni affiancate su una sola pagina, una barra per circuito.

        Stanno insieme perche' rispondono alla stessa domanda su due smell diversi, e affiancate
        si leggono come un confronto invece che come due risultati scollegati. Le due scale non
        sono pero' comparabili -- percentuale a sinistra, colonne di attesa a destra -- ed e' il
        motivo per cui restano due assi distinti con due etichette esplicite, anziche' una serie
        doppia sullo stesso asse.

        Una barra per circuito, etichettata col circuitId, invece di un boxplot o di un
        istogramma: i sottoinsiemi contano pochissime unita' e i quartili di un boxplot su cinque
        punti non avrebbero significato statistico. Con le barre ogni osservazione resta
        rintracciabile nel dataset e citabile nel testo.
        """
        long_circuit = metrics["long_circuit_reduction_pct"]
        idle_qubits = metrics["idle_qubits_reduction"]
        title = "Riduzione delle Metriche di Smell (OPTIMIZED)"

        if not long_circuit["values"] and not idle_qubits["values"]:
            self._placeholder_page(pdf, title)
            return

        fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(11, 5))
        fig.patch.set_facecolor(_SURFACE)

        self._reduction_panel(
            left_ax,
            "Long Circuit",
            long_circuit,
            "Riduzione % di l * c",
            "{:.1f}%",
        )
        self._reduction_panel(
            right_ax,
            "Idle Qubits",
            idle_qubits,
            "Colonne di attesa eliminate",
            "{:.0f}",
        )

        fig.suptitle(title, color=_INK_PRIMARY, fontsize=14)
        fig.tight_layout()
        self._save(pdf, fig)

    def _reduction_panel(
        self, ax: plt.Axes, title: str, summary: dict, ylabel: str, value_format: str
    ) -> None:
        """Un pannello della pagina di riduzione: barre per circuito piu' la linea della media."""
        values = summary["values"]
        self._style_axis(ax, title)
        if not values:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                _NO_DATA_TEXT,
                ha="center",
                va="center",
                color=_INK_SECONDARY,
                fontsize=11,
                transform=ax.transAxes,
            )
            return

        # Le etichette possono mancare (dizionario costruito a mano nei test, o colonna circuitId
        # assente): in quel caso si numerano le barre, senza far fallire il disegno.
        labels = summary.get("labels") or [str(index + 1) for index in range(len(values))]
        bars = ax.bar(range(len(values)), values, color=_BLUE, width=0.6)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

        mean = sum(values) / len(values)
        ax.axhline(mean, color=_ORANGE, linestyle="--", linewidth=1.4, zorder=2)
        ax.text(
            0.99,
            mean,
            f" media {value_format.format(mean)}",
            color=_ORANGE,
            va="bottom",
            ha="right",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
        )

        self._style_count_axis(ax, f"{ylabel}  (n = {len(values)})")
        self._label_bars(ax, bars, [value_format.format(value) for value in values])

    def _page_threshold_recovery(self, pdf: PdfPages, metrics: dict) -> None:
        """Quota di circuiti che dopo il refactoring non sono piu' classificati come smelly.

        Distinta dalle due pagine di riduzione, e non ridondante con esse: un circuito puo'
        ridurre l*c del 48% e restare sopra la soglia. Qui la domanda non e' "di quanto e'
        migliorato" ma "lo smell e' stato rimosso", che e' quella a cui la tesi deve rispondere.
        """
        recovery = metrics["rientro_sotto_soglia"]
        labels = [
            ("l * c\nsotto soglia", recovery["long_circuit"]),
            ("IdQ\nazzerato", recovery["idle_qubits"]),
            ("Nessuno smell\nresiduo", recovery["smell_free"]),
        ]
        available = [(label, share) for label, share in labels if share["rate"] is not None]
        title = "Rientro sotto la Soglia di Rilevamento (OPTIMIZED)"
        if not available:
            self._placeholder_page(pdf, title)
            return

        fig, ax = self._new_figure(title)
        bars = ax.bar(
            [label for label, _ in available],
            [share["rate"] * 100 for _, share in available],
            color=_AQUA,
            width=0.5,
        )
        ax.set_ylim(0, 100)
        self._style_count_axis(ax, "%")
        # L'etichetta porta anche il rapporto grezzo: su denominatori di 5 o 6 circuiti una
        # percentuale da sola ("60%") nasconde quanto sia piccolo il campione che la produce.
        self._label_bars(
            ax,
            bars,
            [f"{share['count']}/{share['total']}" for _, share in available],
        )
        self._save(pdf, fig)

    def _page_iteration_count_distribution(self, pdf: PdfPages, metrics: dict) -> None:
        """Conteggio dei circuiti per numero di iterazioni impiegate a convergere.

        E' un BAR CHART di conteggi, non piu' un istogramma: la variabile e' un intero limitato
        superiormente da maxIterations (con maxIterations=3 assume i soli valori 1, 2, 3), quindi
        non c'e' alcuna scelta di bin da fare e un istogramma inventerebbe una continuita' che il
        dato non ha. Il sottoinsieme e' ora quello dei soli OPTIMIZED, deciso a monte dal
        MetricsCalculator: gli OPT_FAILED si fermano tutti a maxIterations per costruzione e
        avrebbero prodotto una barra terminale che dice solo quanti fallimenti ci sono stati.
        """
        distribution = metrics["iterazioni_alla_convergenza"]["distribution"]
        title = "Iterazioni Impiegate alla Convergenza (OPTIMIZED)"
        if not distribution:
            self._placeholder_page(pdf, title)
            return

        fig, ax = self._new_figure(title)
        iterations = sorted(distribution)
        counts = [distribution[iteration] for iteration in iterations]
        bars = ax.bar([str(iteration) for iteration in iterations], counts, color=_BLUE, width=0.5)
        ax.set_xlabel("Numero di iterazioni", color=_INK_SECONDARY)
        self._style_count_axis(ax, "Numero di circuiti")
        self._label_bars(ax, bars, [str(count) for count in counts])
        self._save(pdf, fig)
