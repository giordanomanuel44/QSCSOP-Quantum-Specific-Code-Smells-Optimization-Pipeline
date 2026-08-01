"""
Script di esecuzione del modulo Analytics sui risultati prodotti da QSCSOP.

Istanzia le dipendenze concrete (DataLoader, MetricsCalculator, ReportVisualizer) e le
inietta in AnalyticsMain, poi lancia il ciclo carica-calcola-visualizza end-to-end su
data/interim/risultati_dataset.jsonl. Produce data/processed/Report_Metriche.pdf.

Uso (dalla root del progetto, con il venv attivo):
    python scripts/run_analytics.py
"""

from pathlib import Path

from qscsop_pipeline.analytics.analytics_main import AnalyticsMain
from qscsop_pipeline.analytics.calculators.metrics_calculator import MetricsCalculator
from qscsop_pipeline.analytics.loaders.data_loader import DataLoader
from qscsop_pipeline.analytics.visualizers.report_visualizer import ReportVisualizer

# scripts/run_analytics.py -> risale alla root del progetto, stesso pattern di
# run_qcep.py/run_qscsop.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "risultati_dataset.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "Report_Metriche.pdf"


if __name__ == "__main__":
    data_loader = DataLoader(filepath=str(INPUT_PATH))
    metrics_calculator = MetricsCalculator()
    report_visualizer = ReportVisualizer(output_path=str(OUTPUT_PATH))

    analytics_main = AnalyticsMain(
        data_loader=data_loader,
        metrics_calculator=metrics_calculator,
        report_visualizer=report_visualizer,
    )
    analytics_main.run()

    print(f"Analytics completato. Report scritto in {OUTPUT_PATH}")
