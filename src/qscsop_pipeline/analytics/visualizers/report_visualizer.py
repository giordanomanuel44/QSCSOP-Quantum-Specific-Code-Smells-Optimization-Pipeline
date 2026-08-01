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

_PROCESSED_STATUSES = [EvaluationStatus.OPTIMIZED.value, EvaluationStatus.OPT_FAILED.value]

_NO_DATA_TEXT = "Nessun dato disponibile"


class ReportVisualizer(IReportVisualizer):
    """Genera Report_Metriche.pdf: un unico PDF multi-pagina (sezione 1.5.3 della tesi)."""

    def __init__(self, output_path: str) -> None:
        self._output_path = Path(output_path)

    def visualize(self, df: pd.DataFrame, metrics: dict) -> None:
        """Genera le sette pagine del report e le esporta in un unico file PDF."""
        sns.set_context("paper")
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(self._output_path) as pdf:
            self._page_core_rates(pdf, metrics)
            self._page_status_distribution(pdf, metrics)
            self._page_failure_reason_distribution(pdf, metrics)
            self._page_long_circuit_reduction(pdf, metrics)
            self._page_idle_qubits_reduction(pdf, metrics)
            self._page_iteration_count_distribution(pdf, df)
            self._page_dataset_source_comparison(pdf, metrics)

    # ---- helper condivisi ----

    @staticmethod
    def _new_figure(title: str) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(_SURFACE)
        ax.set_facecolor(_SURFACE)
        ax.set_title(title, color=_INK_PRIMARY, fontsize=13, pad=12)
        ax.tick_params(colors=_INK_SECONDARY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(_GRIDLINE)
        ax.spines["bottom"].set_color(_GRIDLINE)
        return fig, ax

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
        candidates = [
            ("Tasso di\nRisoluzione", metrics["tasso_risoluzione"]),
            ("Equivalenza\nFunzionale", metrics["equivalenza_funzionale"]),
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

    def _page_long_circuit_reduction(self, pdf: PdfPages, metrics: dict) -> None:
        gate_values = metrics["long_circuit_gate_reduction_pct"]["values"]
        depth_values = metrics["long_circuit_depth_reduction_pct"]["values"]
        title = "Riduzione % Metriche Fisiche (Long Circuit, OPTIMIZED)"
        if not gate_values and not depth_values:
            self._placeholder_page(pdf, title)
            return

        fig, ax = self._new_figure(title)
        box = ax.boxplot(
            [gate_values, depth_values],
            tick_labels=["Gate Count", "Depth"],
            patch_artist=True,
            widths=0.4,
        )
        for patch, color in zip(box["boxes"], (_BLUE, _ORANGE)):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        self._style_count_axis(ax, "Riduzione %")
        self._save(pdf, fig)

    def _page_idle_qubits_reduction(self, pdf: PdfPages, metrics: dict) -> None:
        values = metrics["idle_qubits_reduction"]["values"]
        title = "Riduzione Qubit Logici (Idle Qubits, OPTIMIZED)"
        if not values:
            self._placeholder_page(pdf, title)
            return

        fig, ax = self._new_figure(title)
        ax.hist(
            values,
            bins=range(int(min(values)), int(max(values)) + 2),
            color=_BLUE,
            edgecolor=_SURFACE,
        )
        ax.set_xlabel("Qubit logici risparmiati (baseline - refactored)", color=_INK_SECONDARY)
        self._style_count_axis(ax, "Numero di circuiti")
        self._save(pdf, fig)

    def _page_iteration_count_distribution(self, pdf: PdfPages, df: pd.DataFrame) -> None:
        title = "Distribuzione del Numero di Iterazioni"
        values = self._processed_iteration_counts(df)
        if not values:
            self._placeholder_page(pdf, title)
            return

        fig, ax = self._new_figure(title)
        ax.hist(
            values,
            bins=range(int(min(values)), int(max(values)) + 2),
            color=_BLUE,
            edgecolor=_SURFACE,
        )
        ax.set_xlabel("Numero di iterazioni", color=_INK_SECONDARY)
        self._style_count_axis(ax, "Numero di circuiti")
        self._save(pdf, fig)

    @staticmethod
    def _processed_iteration_counts(df: pd.DataFrame) -> list[int]:
        """iterationCount sul sottoinsieme OPTIMIZED/OPT_FAILED (esclude SMELL_FREE=0 di default,
        stessa esclusione del numero_medio_iterazioni di MetricsCalculator, altrimenti il picco
        artificiale a 0 comprimerebbe la distribuzione reale)."""
        if "evaluation.status" not in df.columns:
            return []
        processed = df[df["evaluation.status"].isin(_PROCESSED_STATUSES)]
        if processed.empty:
            return []
        return processed["evaluation.iterationCount"].dropna().tolist()

    def _page_dataset_source_comparison(self, pdf: PdfPages, metrics: dict) -> None:
        title = "Tasso di Risoluzione per Dataset di Provenienza"
        available = {
            source: values["tasso_risoluzione"]
            for source, values in metrics["metriche_per_dataset_source"].items()
            if values["tasso_risoluzione"] is not None
        }
        if not available:
            self._placeholder_page(pdf, title)
            return

        fig, ax = self._new_figure(title)
        labels = list(available.keys())
        values = [available[label] * 100 for label in labels]
        colors = _CATEGORICAL[: len(labels)]
        bars = ax.bar(labels, values, color=colors, width=0.4)
        ax.set_ylim(0, 100)
        self._style_count_axis(ax, "%")
        self._label_bars(ax, bars, [f"{value:.1f}%" for value in values])
        self._save(pdf, fig)
